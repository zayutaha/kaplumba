use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use half::bf16;
use metal::Buffer;
use thiserror::Error;

use crate::metal::{DEFAULT_RESIDENT_LIMIT_BYTES, DEFAULT_SEGMENT_BYTES, MetalContext, MetalError};
use crate::safetensors::{DType, TextTensorSelection};

const COPY_CHUNK_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum WeightError {
    #[error("failed to open {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("failed to map {path}: {message}")]
    Mmap { path: PathBuf, message: String },
    #[error("tensor range is outside mapped shard {shard}")]
    Range { shard: String },
    #[error("unsupported source dtype {dtype:?} for tensor {name}")]
    UnsupportedDtype { name: String, dtype: DType },
    #[error("resident allocation would require {required} bytes, limit is {limit}")]
    AllocationLimit { required: u64, limit: u64 },
    #[error("weight byte arithmetic overflow")]
    ArithmeticOverflow,
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("invalid weight arena segment size {0}")]
    InvalidSegmentSize(u64),
    #[error("invalid weight span for tensor {name}")]
    InvalidSpan { name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightStorage {
    PlainF16,
    PackedExl3,
    Raw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSpan {
    pub segment: usize,
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone)]
pub struct LoadedTensor {
    pub name: String,
    pub source_dtype: DType,
    pub storage_dtype: DType,
    pub shape: Vec<usize>,
    pub spans: Vec<ArenaSpan>,
    pub storage: WeightStorage,
    pub converted_from_bf16: bool,
}

impl LoadedTensor {
    pub fn byte_len(&self) -> Result<u64, WeightError> {
        self.spans.iter().try_fold(0_u64, |total, span| {
            total
                .checked_add(span.len)
                .ok_or(WeightError::ArithmeticOverflow)
        })
    }
}

struct BufferSegment {
    buffer: Buffer,
    len: u64,
}

pub struct WeightArena {
    context: MetalContext,
    segments: Vec<BufferSegment>,
    current_segment: Option<usize>,
    current_used: u64,
    logical_bytes: u64,
    capacity_bytes: u64,
    limit_bytes: u64,
    segment_bytes: u64,
}

impl WeightArena {
    pub fn new(
        context: MetalContext,
        limit_bytes: u64,
        segment_bytes: u64,
    ) -> Result<Self, WeightError> {
        if segment_bytes == 0 {
            return Err(WeightError::InvalidSegmentSize(segment_bytes));
        }
        if segment_bytes > context.max_buffer_length() {
            return Err(WeightError::InvalidSegmentSize(segment_bytes));
        }
        Ok(Self {
            context,
            segments: Vec::new(),
            current_segment: None,
            current_used: 0,
            logical_bytes: 0,
            capacity_bytes: 0,
            limit_bytes,
            segment_bytes,
        })
    }

    pub fn context(&self) -> &MetalContext {
        &self.context
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }

    pub fn segment_bytes(&self) -> u64 {
        self.segment_bytes
    }

    pub fn buffer(&self, segment: usize) -> Option<&Buffer> {
        self.segments.get(segment).map(|value| &value.buffer)
    }

    pub fn allocate(&mut self, length: u64) -> Result<Vec<ArenaSpan>, WeightError> {
        if length == 0 {
            return Ok(Vec::new());
        }
        let required = self.required_capacity(length)?;
        if required > self.limit_bytes {
            return Err(WeightError::AllocationLimit {
                required,
                limit: self.limit_bytes,
            });
        }
        let mut spans = Vec::new();
        let mut remaining = length;
        while remaining > 0 {
            if self.current_segment.is_none() {
                let segment_index = self.segments.len();
                let buffer = self.context.allocate_shared(self.segment_bytes)?;
                self.segments.push(BufferSegment {
                    buffer,
                    len: self.segment_bytes,
                });
                self.capacity_bytes = self
                    .capacity_bytes
                    .checked_add(self.segment_bytes)
                    .ok_or(WeightError::ArithmeticOverflow)?;
                self.current_segment = Some(segment_index);
                self.current_used = 0;
            }
            let segment_index = self
                .current_segment
                .ok_or(WeightError::ArithmeticOverflow)?;
            let available = self.segment_bytes - self.current_used;
            let amount = remaining.min(available);
            spans.push(ArenaSpan {
                segment: segment_index,
                offset: self.current_used,
                len: amount,
            });
            self.current_used += amount;
            self.logical_bytes = self
                .logical_bytes
                .checked_add(amount)
                .ok_or(WeightError::ArithmeticOverflow)?;
            remaining -= amount;
            if self.current_used == self.segment_bytes {
                self.current_segment = None;
            }
        }
        Ok(spans)
    }

    pub fn write_at(
        &self,
        spans: &[ArenaSpan],
        data_offset: u64,
        data: &[u8],
    ) -> Result<(), WeightError> {
        let data_end = data_offset
            .checked_add(data.len() as u64)
            .ok_or(WeightError::ArithmeticOverflow)?;
        let total = span_total(spans)?;
        if data_end > total {
            return Err(WeightError::ArithmeticOverflow);
        }
        let mut span_offset = 0_u64;
        for span in spans {
            let span_end = span_offset + span.len;
            if data_end <= span_offset || data_offset >= span_end {
                span_offset = span_end;
                continue;
            }
            let start = data_offset.saturating_sub(span_offset) as usize;
            let end = data_end.min(span_end).saturating_sub(span_offset) as usize;
            let segment = self
                .segments
                .get(span.segment)
                .ok_or(WeightError::ArithmeticOverflow)?;
            if span
                .offset
                .checked_add(span.len)
                .ok_or(WeightError::ArithmeticOverflow)?
                > segment.len
            {
                return Err(WeightError::ArithmeticOverflow);
            }
            let contents = crate::metal::shared_contents(&segment.buffer)?;
            unsafe {
                let destination = (contents as *mut u8).add((span.offset + start as u64) as usize);
                std::ptr::copy_nonoverlapping(data[start..end].as_ptr(), destination, end - start);
            }
            span_offset = span_end;
        }
        Ok(())
    }

    pub fn read(&self, spans: &[ArenaSpan]) -> Result<Vec<u8>, WeightError> {
        let total = span_total(spans)?;
        let total_usize = usize::try_from(total).map_err(|_| WeightError::ArithmeticOverflow)?;
        let mut output = vec![0_u8; total_usize];
        let mut output_offset = 0_usize;
        for span in spans {
            let segment = self
                .segments
                .get(span.segment)
                .ok_or(WeightError::ArithmeticOverflow)?;
            if span
                .offset
                .checked_add(span.len)
                .ok_or(WeightError::ArithmeticOverflow)?
                > segment.len
            {
                return Err(WeightError::ArithmeticOverflow);
            }
            let contents = crate::metal::shared_contents(&segment.buffer)?;
            let len = span.len as usize;
            unsafe {
                let source = (contents as *const u8).add(span.offset as usize);
                std::ptr::copy_nonoverlapping(source, output[output_offset..].as_mut_ptr(), len);
            }
            output_offset += len;
        }
        if output_offset != output.len() {
            return Err(WeightError::ArithmeticOverflow);
        }
        Ok(output)
    }

    fn required_capacity(&self, additional: u64) -> Result<u64, WeightError> {
        let mut capacity = self.capacity_bytes;
        let mut remaining = additional;
        let mut current_used = self.current_used;
        let mut has_current = self.current_segment.is_some();
        while remaining > 0 {
            if !has_current {
                capacity = capacity
                    .checked_add(self.segment_bytes)
                    .ok_or(WeightError::ArithmeticOverflow)?;
                current_used = 0;
            }
            let available = self.segment_bytes - current_used;
            if remaining <= available {
                break;
            }
            remaining -= available;
            current_used = self.segment_bytes;
            has_current = false;
        }
        Ok(capacity)
    }
}

fn span_total(spans: &[ArenaSpan]) -> Result<u64, WeightError> {
    spans.iter().try_fold(0_u64, |total, span| {
        total
            .checked_add(span.len)
            .ok_or(WeightError::ArithmeticOverflow)
    })
}

pub struct TextWeights {
    arena: WeightArena,
    tensors: BTreeMap<String, LoadedTensor>,
    source_bytes: u64,
}

impl TextWeights {
    pub fn load(
        model_dir: &Path,
        selection: &TextTensorSelection,
        context: MetalContext,
        limit_bytes: u64,
    ) -> Result<Self, WeightError> {
        Self::load_with_segment_size(
            model_dir,
            selection,
            context,
            limit_bytes,
            DEFAULT_SEGMENT_BYTES,
        )
    }

    pub fn load_with_segment_size(
        model_dir: &Path,
        selection: &TextTensorSelection,
        context: MetalContext,
        limit_bytes: u64,
        segment_bytes: u64,
    ) -> Result<Self, WeightError> {
        let mut maps = BTreeMap::new();
        for (shard_name, shard) in &selection.shards {
            if shard.tensors.is_empty() {
                continue;
            }
            let path = model_dir.join(shard_name);
            maps.insert(shard_name.clone(), Mmap::open(&path)?);
        }
        let mut arena = WeightArena::new(context, limit_bytes, segment_bytes)?;
        let mut tensors = BTreeMap::new();
        let mut source_bytes = 0_u64;
        for (name, info) in &selection.tensors {
            let (storage, storage_dtype, converted) = classify_tensor(name, info.dtype)?;
            let source_len = info.byte_len();
            source_bytes = source_bytes
                .checked_add(source_len)
                .ok_or(WeightError::ArithmeticOverflow)?;
            let spans = arena.allocate(source_len)?;
            let shard_name = selection
                .shard_for(name)
                .ok_or_else(|| WeightError::Range {
                    shard: name.clone(),
                })?;
            let shard = selection
                .shards
                .get(shard_name)
                .ok_or_else(|| WeightError::Range {
                    shard: shard_name.to_string(),
                })?;
            let range = shard
                .absolute_range(name)
                .ok_or_else(|| WeightError::Range {
                    shard: shard_name.to_string(),
                })?;
            let map = maps.get(shard_name).ok_or_else(|| WeightError::Range {
                shard: shard_name.to_string(),
            })?;
            let (start, end) = range;
            let source = map.slice(start..end)?;
            copy_tensor(&arena, &spans, source, converted)?;
            tensors.insert(
                name.clone(),
                LoadedTensor {
                    name: name.clone(),
                    source_dtype: info.dtype,
                    storage_dtype,
                    shape: info.shape.clone(),
                    spans,
                    storage,
                    converted_from_bf16: converted,
                },
            );
        }
        Ok(Self {
            arena,
            tensors,
            source_bytes,
        })
    }

    pub fn load_default(
        model_dir: &Path,
        selection: &TextTensorSelection,
    ) -> Result<Self, WeightError> {
        let context = MetalContext::new()?;
        Self::load(model_dir, selection, context, DEFAULT_RESIDENT_LIMIT_BYTES)
    }

    pub fn tensor(&self, name: &str) -> Option<&LoadedTensor> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> &BTreeMap<String, LoadedTensor> {
        &self.tensors
    }

    pub fn arena(&self) -> &WeightArena {
        &self.arena
    }

    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }

    pub fn resident_bytes(&self) -> u64 {
        self.arena.capacity_bytes()
    }

    pub fn logical_bytes(&self) -> u64 {
        self.arena.logical_bytes()
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightError::InvalidSpan {
                name: name.to_string(),
            })?;
        self.arena.read(&tensor.spans)
    }

    pub fn packed_u16(&self, name: &str) -> Result<Vec<u16>, WeightError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightError::InvalidSpan {
                name: name.to_string(),
            })?;
        if tensor.storage != WeightStorage::PackedExl3 {
            return Err(WeightError::InvalidSpan {
                name: name.to_string(),
            });
        }
        let bytes = self.arena.read(&tensor.spans)?;
        if !bytes.len().is_multiple_of(2) {
            return Err(WeightError::ArithmeticOverflow);
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect())
    }
}

fn classify_tensor(name: &str, dtype: DType) -> Result<(WeightStorage, DType, bool), WeightError> {
    let packed = is_exl3_component(name);
    match dtype {
        DType::BF16 if !packed => Ok((WeightStorage::PlainF16, DType::F16, true)),
        DType::F16 if packed => Ok((WeightStorage::PackedExl3, DType::F16, false)),
        DType::F16 => Ok((WeightStorage::PlainF16, DType::F16, false)),
        DType::I8
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::I32
        | DType::U32
        | DType::I64
        | DType::U64
            if packed =>
        {
            Ok((WeightStorage::PackedExl3, dtype, false))
        }
        DType::I8
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::I32
        | DType::U32
        | DType::I64
        | DType::U64 => Ok((WeightStorage::Raw, dtype, false)),
        _ => Err(WeightError::UnsupportedDtype {
            name: name.to_string(),
            dtype,
        }),
    }
}

fn is_exl3_component(name: &str) -> bool {
    [".trellis", ".mcg", ".su", ".suh", ".sv", ".svh"]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn copy_tensor(
    arena: &WeightArena,
    spans: &[ArenaSpan],
    source: &[u8],
    convert_bf16: bool,
) -> Result<(), WeightError> {
    if !convert_bf16 {
        for (offset, chunk) in source.chunks(COPY_CHUNK_BYTES).enumerate() {
            let data_offset = (offset * COPY_CHUNK_BYTES) as u64;
            arena.write_at(spans, data_offset, chunk)?;
        }
        return Ok(());
    }
    if !source.len().is_multiple_of(2) {
        return Err(WeightError::ArithmeticOverflow);
    }
    let mut converted = Vec::new();
    for (offset, chunk) in source.chunks(COPY_CHUNK_BYTES).enumerate() {
        if chunk.len() % 2 != 0 {
            return Err(WeightError::ArithmeticOverflow);
        }
        converted.clear();
        converted.resize(chunk.len(), 0);
        for (source_bytes, destination_bytes) in
            chunk.chunks_exact(2).zip(converted.chunks_exact_mut(2))
        {
            let bits = u16::from_le_bytes([source_bytes[0], source_bytes[1]]);
            let value = half::f16::from_f32(bf16::from_bits(bits).to_f32()).to_bits();
            destination_bytes.copy_from_slice(&value.to_le_bytes());
        }
        arena.write_at(spans, (offset * COPY_CHUNK_BYTES) as u64, &converted)?;
    }
    Ok(())
}

struct Mmap {
    pointer: NonNull<u8>,
    length: usize,
    path: PathBuf,
}

impl Mmap {
    fn open(path: &Path) -> Result<Self, WeightError> {
        let file = File::open(path).map_err(|source| WeightError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let length = file
            .metadata()
            .map_err(|source| WeightError::Io {
                path: path.to_path_buf(),
                source,
            })?
            .len();
        let length = usize::try_from(length).map_err(|_| WeightError::Mmap {
            path: path.to_path_buf(),
            message: "file does not fit in process address space".to_string(),
        })?;
        if length == 0 {
            return Err(WeightError::Mmap {
                path: path.to_path_buf(),
                message: "cannot map an empty shard".to_string(),
            });
        }
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(WeightError::Mmap {
                path: path.to_path_buf(),
                message: "mmap failed".to_string(),
            });
        }
        Ok(Self {
            pointer: NonNull::new(pointer.cast::<u8>()).ok_or_else(|| WeightError::Mmap {
                path: path.to_path_buf(),
                message: "mmap returned null".to_string(),
            })?,
            length,
            path: path.to_path_buf(),
        })
    }

    fn slice(&self, range: Range<u64>) -> Result<&[u8], WeightError> {
        let start = usize::try_from(range.start).map_err(|_| WeightError::Range {
            shard: self.path.display().to_string(),
        })?;
        let end = usize::try_from(range.end).map_err(|_| WeightError::Range {
            shard: self.path.display().to_string(),
        })?;
        if start > end || end > self.length {
            return Err(WeightError::Range {
                shard: self.path.display().to_string(),
            });
        }
        Ok(unsafe { std::slice::from_raw_parts(self.pointer.as_ptr().add(start), end - start) })
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use half::bf16;
    use serde_json::json;

    use super::*;
    use crate::safetensors::{inspect_model_dir, read_index};

    fn temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "kaplumba-weights-{name}-{}-{stamp}",
            std::process::id()
        ));
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
            let dtype_name = match dtype {
                DType::BF16 => "BF16",
                DType::F16 => "F16",
                DType::I16 => "I16",
                DType::I32 => "I32",
                DType::F32 => "F32",
                other => panic!("unsupported test dtype {other:?}"),
            };
            header.insert(
                name.to_string(),
                json!({
                    "dtype": dtype_name,
                    "shape": shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut file = fs::File::create(path).unwrap();
        use std::io::Write;
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&payload).unwrap();
    }

    fn selection_for(
        dir: &Path,
        entries: Vec<(&str, DType, Vec<usize>, Vec<u8>)>,
    ) -> TextTensorSelection {
        let total: usize = entries.iter().map(|(_, _, _, bytes)| bytes.len()).sum();
        let shard = dir.join("model.safetensors");
        write_shard(&shard, entries.clone());
        let index_path = dir.join("model.safetensors.index.json");
        let weight_map: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        let mut map = weight_map;
        let names: Vec<String> = vec![
            "model.language_model.norm.weight".to_string(),
            "model.language_model.layers.0.self_attn.q_proj.trellis".to_string(),
            "model.vision_tower.weight".to_string(),
        ];
        let used_names = names
            .into_iter()
            .filter(|name| entries.iter().any(|(entry, _, _, _)| *entry == name))
            .collect::<Vec<_>>();
        for name in used_names {
            map.insert(name, json!("model.safetensors"));
        }
        fs::write(
            &index_path,
            serde_json::to_vec(&json!({
                "metadata": {"total_size": total},
                "weight_map": map
            }))
            .unwrap(),
        )
        .unwrap();
        let index = read_index(&index_path).unwrap();
        crate::safetensors::load_text_selection(dir, index).unwrap()
    }

    fn context_or_skip() -> Option<MetalContext> {
        MetalContext::new().ok()
    }

    #[test]
    fn converts_plain_bf16_without_changing_size() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("bf16");
        let one = bf16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        let minus_two = bf16::from_f32(-2.0).to_bits().to_le_bytes().to_vec();
        let mut bytes = one.clone();
        bytes.extend_from_slice(&minus_two);
        let selection = selection_for(
            &dir,
            vec![(
                "model.language_model.norm.weight",
                DType::BF16,
                vec![2],
                bytes,
            )],
        );
        let weights =
            TextWeights::load_with_segment_size(&dir, &selection, context, 1024 * 1024, 64)
                .unwrap();
        let tensor = weights.tensor("model.language_model.norm.weight").unwrap();
        assert_eq!(tensor.storage, WeightStorage::PlainF16);
        assert_eq!(tensor.storage_dtype, DType::F16);
        assert!(tensor.converted_from_bf16);
        let output = weights
            .read_tensor("model.language_model.norm.weight")
            .unwrap();
        assert_eq!(u16::from_le_bytes([output[0], output[1]]), 0x3c00);
        assert_eq!(u16::from_le_bytes([output[2], output[3]]), 0xc000);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn retains_packed_exl3_bytes() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("packed");
        let selection = selection_for(
            &dir,
            vec![(
                "model.language_model.layers.0.self_attn.q_proj.trellis",
                DType::I16,
                vec![1, 1, 2],
                vec![0x34, 0x12, 0x78, 0x56],
            )],
        );
        let weights =
            TextWeights::load_with_segment_size(&dir, &selection, context, 1024 * 1024, 64)
                .unwrap();
        let tensor = weights
            .tensor("model.language_model.layers.0.self_attn.q_proj.trellis")
            .unwrap();
        assert_eq!(tensor.storage, WeightStorage::PackedExl3);
        assert_eq!(
            weights
                .read_tensor("model.language_model.layers.0.self_attn.q_proj.trellis")
                .unwrap(),
            vec![0x34, 0x12, 0x78, 0x56]
        );
        assert_eq!(
            weights
                .packed_u16("model.language_model.layers.0.self_attn.q_proj.trellis")
                .unwrap(),
            vec![0x1234, 0x5678]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn excludes_vision_from_source_and_resident_bytes() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("vision");
        let selection = selection_for(
            &dir,
            vec![
                (
                    "model.language_model.norm.weight",
                    DType::BF16,
                    vec![1],
                    vec![0, 0],
                ),
                (
                    "model.vision_tower.weight",
                    DType::BF16,
                    vec![2],
                    vec![0, 0, 0, 0],
                ),
            ],
        );
        let weights =
            TextWeights::load_with_segment_size(&dir, &selection, context, 1024 * 1024, 64)
                .unwrap();
        assert_eq!(weights.source_bytes(), 2);
        assert!(weights.tensor("model.vision_tower.weight").is_none());
        assert!(weights.resident_bytes() >= weights.logical_bytes());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn copies_tensors_across_multiple_arena_segments() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("spans");
        let bytes = vec![0xA5_u8; 130];
        let selection = selection_for(
            &dir,
            vec![(
                "model.language_model.norm.weight",
                DType::F16,
                vec![65],
                bytes.clone(),
            )],
        );
        let weights =
            TextWeights::load_with_segment_size(&dir, &selection, context, 1024, 64).unwrap();
        let tensor = weights.tensor("model.language_model.norm.weight").unwrap();
        assert_eq!(tensor.spans.len(), 3);
        assert_eq!(weights.read_tensor(tensor.name.as_str()).unwrap(), bytes);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fails_before_exceeding_resident_limit() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut arena = WeightArena::new(context, 3, 64).unwrap();
        let error = arena.allocate(4).unwrap_err().to_string();
        assert!(error.contains("resident allocation"));
    }

    #[test]
    #[ignore = "requires KAPLUMBA_TEST_MODEL"]
    fn validates_real_text_selection_without_allocating_weights() {
        let model_dir = PathBuf::from(
            std::env::var("KAPLUMBA_TEST_MODEL")
                .expect("KAPLUMBA_TEST_MODEL must point to the checkpoint"),
        );
        let metadata = inspect_model_dir(&model_dir).unwrap();
        let mut source_bytes = 0_u64;
        for (name, info) in &metadata.selection.tensors {
            classify_tensor(name, info.dtype).unwrap();
            source_bytes += info.byte_len();
        }
        assert_eq!(source_bytes, 14_876_563_172);
        assert!(metadata.selection.tensors.len() > 1_000);
        assert!(!metadata.selection.excluded_vision.is_empty());
    }

    #[test]
    fn rejects_unsupported_plain_dtype() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("dtype");
        let selection = selection_for(
            &dir,
            vec![(
                "model.language_model.norm.weight",
                DType::F32,
                vec![1],
                vec![0, 0, 0, 0],
            )],
        );
        let error =
            match TextWeights::load_with_segment_size(&dir, &selection, context, 1024 * 1024, 64) {
                Ok(_) => panic!("unsupported dtype unexpectedly loaded"),
                Err(error) => error.to_string(),
            };
        assert!(error.contains("unsupported source dtype"));
        fs::remove_dir_all(dir).unwrap();
    }
}
