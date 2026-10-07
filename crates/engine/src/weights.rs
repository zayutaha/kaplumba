use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use metal::Buffer;
use rayon::prelude::*;
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
    /// Shard this tensor's bytes live in, and their `[start, end)` range.
    pub shard: String,
    pub start: u64,
    pub end: u64,
    /// Source byte length, recorded because zero-copy tensors have no arena spans.
    pub source_len: u64,
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

    pub fn allocate_dedicated(&mut self, length: u64) -> Result<Vec<ArenaSpan>, WeightError> {
        if length == 0 {
            return Ok(Vec::new());
        }
        if length > self.context.max_buffer_length() {
            return Err(WeightError::InvalidSegmentSize(length));
        }
        let required = self
            .capacity_bytes
            .checked_add(length)
            .ok_or(WeightError::ArithmeticOverflow)?;
        if required > self.limit_bytes {
            return Err(WeightError::AllocationLimit {
                required,
                limit: self.limit_bytes,
            });
        }
        let buffer = self.context.allocate_shared(length)?;
        let segment = self.segments.len();
        self.segments.push(BufferSegment {
            buffer,
            len: length,
        });
        self.capacity_bytes = required;
        self.logical_bytes = self
            .logical_bytes
            .checked_add(length)
            .ok_or(WeightError::ArithmeticOverflow)?;
        self.current_segment = None;
        self.current_used = 0;
        Ok(vec![ArenaSpan {
            segment,
            offset: 0,
            len: length,
        }])
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
            let copy_start = data_offset.max(span_offset) - data_offset;
            let copy_end = data_end.min(span_end) - data_offset;
            let destination_offset = data_offset.max(span_offset) - span_offset;
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
            let copy_start =
                usize::try_from(copy_start).map_err(|_| WeightError::ArithmeticOverflow)?;
            let copy_end =
                usize::try_from(copy_end).map_err(|_| WeightError::ArithmeticOverflow)?;
            let destination_offset =
                usize::try_from(destination_offset).map_err(|_| WeightError::ArithmeticOverflow)?;
            unsafe {
                let destination =
                    (contents as *mut u8).add((span.offset as usize) + destination_offset);
                std::ptr::copy_nonoverlapping(
                    data[copy_start..copy_end].as_ptr(),
                    destination,
                    copy_end - copy_start,
                );
            }
            span_offset = span_end;
        }
        Ok(())
    }

    /// Read `len` bytes starting at `file_offset` directly into the destination
    /// window covered by `spans`.
    ///
    /// A tensor may straddle several arena segments; each intersecting piece is
    /// read straight into its own buffer, so no staging copy of the tensor is
    /// ever materialised in host memory.
    pub fn read_into(
        &self,
        file: &File,
        spans: &[ArenaSpan],
        file_offset: u64,
        len: u64,
    ) -> Result<(), WeightError> {
        let total = span_total(spans)?;
        if len > total {
            return Err(WeightError::ArithmeticOverflow);
        }
        let mut remaining = len;
        let mut file_position = file_offset;
        for span in spans {
            if remaining == 0 {
                break;
            }
            let take = u64::min(span.len, remaining);
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
            let destination = unsafe { (contents as *mut u8).add(span.offset as usize) };
            read_exact_at(file, destination, take, file_position)?;
            file_position += take;
            remaining -= take;
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
    /// Shard mappings stay resident for the lifetime of the weights because
    /// zero-copy Metal buffers alias these pages; dropping them would leave the
    /// GPU reading freed memory.
    shards: BTreeMap<String, ShardReader>,
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
        Self::load_internal(
            model_dir,
            selection,
            context,
            limit_bytes,
            segment_bytes,
            false,
        )
    }

    pub fn load_dedicated(
        model_dir: &Path,
        selection: &TextTensorSelection,
        context: MetalContext,
        limit_bytes: u64,
    ) -> Result<Self, WeightError> {
        Self::load_internal(
            model_dir,
            selection,
            context,
            limit_bytes,
            DEFAULT_SEGMENT_BYTES,
            true,
        )
    }

    fn load_internal(
        model_dir: &Path,
        selection: &TextTensorSelection,
        context: MetalContext,
        limit_bytes: u64,
        segment_bytes: u64,
        dedicated: bool,
    ) -> Result<Self, WeightError> {
        let mut shards = BTreeMap::new();
        for (shard_name, shard) in &selection.shards {
            if shard.tensors.is_empty() {
                continue;
            }
            let path = model_dir.join(shard_name);
            shards.insert(shard_name.clone(), ShardReader::open(&path)?);
        }
        let phase_start = std::time::Instant::now();
        let mut arena = WeightArena::new(context, limit_bytes, segment_bytes)?;
        let mut source_bytes = 0_u64;

        // Phase 1: allocate every destination up front (sequential, because the
        // arena needs `&mut`). Allocation is pure bookkeeping plus Metal buffer
        // creation; the expensive part is the data copy, which happens in
        // parallel in phase 2.
        struct PendingCopy {
            name: String,
            shard: String,
            file_offset: u64,
            len: u64,
            spans: Vec<ArenaSpan>,
            storage: WeightStorage,
            storage_dtype: DType,
            source_dtype: DType,
            shape: Vec<usize>,
            converted: bool,
            /// EXL3 trellis bytes are consumed straight from the shard mapping,
            /// so they never enter the arena and cost no anonymous memory.
            skip_copy: bool,
        }

        let mut pending = Vec::with_capacity(selection.tensors.len());
        for (name, info) in &selection.tensors {
            let (storage, storage_dtype, converted) = classify_tensor(name, info.dtype)?;
            let source_len = info.byte_len();
            source_bytes = source_bytes
                .checked_add(source_len)
                .ok_or(WeightError::ArithmeticOverflow)?;
            // Resident by default. Decode streams every layer weight on every
            // token, so file-backing the 11 GB of layers let the OS evict them:
            // that build measured 3.9 GiB of swap at ~1.4 tok/s. The two
            // offloaded tensors are the exception, and only because they are
            // cheap to serve from the file cache:
            //
            //   embed_tokens (2.82 GB) - exactly one row is read per token, a
            //     ~10 KiB working set, so its pages stay hot.
            //   lm_head (1.06 GB) - read in full each token, but 1.06 GB against
            //     11 GB of layers is a small enough share to stay resident in the
            //     file cache.
            //
            // Both are read as 16-bit values, so the 2-byte-aligned file offsets
            // are fine. Layer trellis is NOT safe to offload: see
            // `exl3_code_at` in linear.metal for the 4-byte alignment trap.
            // Both are mapped by default, and the two switches exist because
            // the tensors behave completely differently. `phys_footprint` via
            // `/usr/bin/time -l` (never `ru_maxrss`: it misses Metal pages
            // entirely), output byte-identical at `--temperature 0`:
            //
            //   all resident (baseline)          15.12 GiB
            //   lm_head.trellis mapped           14.12 GiB   (-1.00)
            //   embed_tokens mapped (in-kernel)  12.45 GiB   (-2.67)
            //   both mapped (default)            11.51 GiB   (-3.61)
            //
            //   embed_tokens (2.82 GB) - one row is read per token, so mapping
            //     it leaves a ~10 KiB working set and saves the full 2.82 GB.
            //
            //   lm_head.trellis (1.06 GB) - read *in full* on every token, which
            //     sounds like it should be the worst candidate to map, but
            //     measurement says otherwise: mapping it saves 1.00 GiB.
            //
            // CORRECTION (2026-09-30). An earlier revision of this comment said
            // the opposite here: "read *in full* on every token, so mapping it
            // just pages every byte back in, and file-backed resident pages
            // cost *more* than arena pages. Measured 12.99 -> 14.14 GiB, i.e.
            // actively harmful. Left resident." **That was wrong and the sign of
            // the result was inverted.** The 12.99 GiB it used as the baseline
            // was a run that had already failed (`lm_head.suh has 0 values`) and
            // so never allocated KV or scratch -- it was artificially LOW, and
            // comparing a healthy run against it made the healthy run look like
            // a 1.15 GiB regression. The 14.14 figure was not invented; it is
            // the correct measurement of the lm-head-only configuration. The
            // mistake was the baseline it was paired with. (Recorded in
            // `spec/resident-memory-12gib/implementation_notes.md`, which had
            // already flagged 12.99 as an artifact.)
            //
            // "Left resident" was also false about the code three lines below,
            // which maps the tensor, and it contradicted the paragraph twelve
            // lines above in this same block, which had it right. The paging
            // worry is real in principle but does not bite at this size: the
            // 1.06 GB stays file-backed-resident in the page cache rather than
            // costing anonymous memory, and 1.06 GB fits comfortably in the
            // reclaimable cache left over after the ~11.3 GiB arena. A
            // 2500-token decode re-reads those 1.06 GB 2500 times (~2.6 TB of
            // reads) and grows swap by zero pages, at 0 swaps in
            // `/usr/bin/time -l` and 0 MiB in `vm.swapusage` sampled every 5 s.
            // See the swap watch (T9) in
            // `spec/resident-memory-12gib/implementation_notes.md`.
            //
            // `lm_head.suh` / `lm_head.svh` / `lm_head.mcg` are sign side-tensors
            // read on the CPU via `read_f16` and always need their arena copy.
            // Clean file-backed mapping by default for packed EXL3 weights.
            // Copying 13.4 GiB of weights into dirty anonymous Metal buffers
            // causes macOS to exhaust available RAM at 16k context (where KV cache
            // adds ~3 GiB) and swap out 12.5 GiB to SSD, collapsing decode to 0.03 tok/s.
            // Clean file-backed mmap pages cannot be compressed or swapped.
            let force_resident = std::env::var_os("KAPLUMBA_RESIDENT_WEIGHTS").is_some();
            let offload_embedding = !force_resident;
            let offload_head = !force_resident;
            let offload_bf16_embedding =
                offload_embedding && name == "model.language_model.embed_tokens.weight";
            let offload_head_trellis = offload_head && name == "lm_head.trellis";
            let offload_trellis = !force_resident && name.ends_with(".trellis");
            let skip_copy = offload_bf16_embedding
                || (!converted && (offload_trellis || offload_head_trellis));
            let spans = if skip_copy {
                Vec::new()
            } else if dedicated {
                arena.allocate_dedicated(source_len)?
            } else {
                arena.allocate(source_len)?
            };
            let shard_name = selection
                .shard_for(name)
                .ok_or_else(|| WeightError::Range {
                    shard: name.to_string(),
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
            let reader = shards.get(shard_name).ok_or_else(|| WeightError::Range {
                shard: shard_name.to_string(),
            })?;
            let (start, end) = range;
            reader.check_range(start, end - start)?;
            pending.push(PendingCopy {
                name: name.clone(),
                shard: shard_name.to_string(),
                file_offset: start,
                len: end - start,
                spans,
                storage,
                storage_dtype,
                source_dtype: info.dtype,
                shape: info.shape.clone(),
                converted,
                skip_copy,
            });
        }

        // Phase 2: copy shard bytes into the arena in parallel. Each tensor owns
        // a disjoint destination range, and `copy_tensor` only ever writes the
        // slice of that range it was given, so the writes never overlap and the
        // conversion of one tensor never observes another.
        let alloc_ms = phase_start.elapsed().as_millis();
        let copy_start = std::time::Instant::now();
        // Read in ascending file order so concurrent readers stay close to
        // sequential within each shard and the drive keeps full readahead.
        pending.sort_by(|left, right| {
            left.shard
                .cmp(&right.shard)
                .then(left.file_offset.cmp(&right.file_offset))
        });
        let arena_ref = &arena;
        let shards_ref = &shards;
        pending.par_iter().try_for_each(|item| {
            if item.skip_copy {
                return Ok(());
            }
            let reader = shards_ref
                .get(&item.shard)
                .ok_or_else(|| WeightError::Range {
                    shard: item.shard.clone(),
                })?;
            copy_tensor(
                arena_ref,
                &reader.file,
                &item.spans,
                item.file_offset,
                item.len,
                item.converted,
            )
        })?;
        let copy_ms = copy_start.elapsed().as_millis();
        if std::env::var_os("KAPLUMBA_LOAD_PROFILE").is_some() {
            eprintln!(
                "load profile: alloc={alloc_ms}ms read+convert={copy_ms}ms tensors={} bytes={}",
                pending.len(),
                source_bytes
            );
        }

        // Phase 3: register the loaded tensors in deterministic name order.
        let mut tensors = BTreeMap::new();
        for item in pending {
            tensors.insert(
                item.name.clone(),
                LoadedTensor {
                    name: item.name,
                    source_dtype: item.source_dtype,
                    storage_dtype: item.storage_dtype,
                    shape: item.shape,
                    spans: item.spans,
                    storage: item.storage,
                    converted_from_bf16: item.converted,
                    shard: item.shard,
                    start: item.file_offset,
                    end: item.file_offset + item.len,
                    source_len: item.len,
                },
            );
        }
        Ok(Self {
            arena,
            tensors,
            source_bytes,
            shards,
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

    /// Zero-copy Metal view of a tensor that needs no dtype conversion.
    ///
    /// EXL3 trellis and sign tensors are already stored in the layout the
    /// kernels read, so they are bound directly from the shard mapping instead
    /// of being copied into anonymous memory. Returns `None` for tensors that
    /// were converted at load time (BF16 to FP16), which do live in the arena.
    pub fn zero_copy_tensor_buffer(
        &self,
        name: &str,
    ) -> Result<Option<(Buffer, u64)>, WeightError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightError::InvalidSpan {
                name: name.to_string(),
            })?;
        if tensor.converted_from_bf16 {
            return Ok(None);
        }
        let device = metal::Device::system_default().ok_or(WeightError::Metal(
            crate::metal::MetalError::DeviceUnavailable,
        ))?;
        self.raw_view(&device, name)
    }

    /// Map a tensor straight out of the shard file, in its *source* dtype.
    ///
    /// Unlike [`Self::zero_copy_tensor_buffer`] this does not refuse tensors that
    /// the loader would otherwise convert (bf16 -> fp16). The caller is then
    /// responsible for converting in the kernel. Only safe for the embedding,
    /// whose working set is a single row per token.
    pub fn raw_zero_copy_tensor_buffer(
        &self,
        name: &str,
    ) -> Result<Option<(Buffer, u64)>, WeightError> {
        let device = metal::Device::system_default().ok_or(WeightError::Metal(
            crate::metal::MetalError::DeviceUnavailable,
        ))?;
        self.raw_view(&device, name)
    }

    fn raw_view(
        &self,
        device: &metal::Device,
        name: &str,
    ) -> Result<Option<(Buffer, u64)>, WeightError> {
        let tensor = self.tensors.get(name).ok_or(WeightError::InvalidSpan {
            name: name.to_string(),
        })?;
        let Some(reader) = self.shards.get(&tensor.shard) else {
            return Ok(None);
        };
        let (start, end) = (tensor.start, tensor.end);
        Ok(Some(reader.no_copy_view(device, start, end - start)?))
    }

    pub fn tensor_buffer(&self, name: &str) -> Result<Buffer, WeightError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightError::InvalidSpan {
                name: name.to_string(),
            })?;
        if tensor.spans.len() != 1 || tensor.spans[0].offset != 0 {
            return Err(WeightError::InvalidSpan {
                name: name.to_string(),
            });
        }
        self.arena
            .buffer(tensor.spans[0].segment)
            .cloned()
            .ok_or_else(|| WeightError::InvalidSpan {
                name: name.to_string(),
            })
    }

    pub fn read_f16(&self, name: &str) -> Result<Vec<u16>, WeightError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightError::InvalidSpan {
                name: name.to_string(),
            })?;
        if tensor.storage == WeightStorage::Raw || tensor.storage_dtype != DType::F16 {
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

/// Load one tensor's shard bytes into its arena spans.
///
/// Chunks are independent and run in parallel: each reads a disjoint file range
/// and writes a disjoint destination range, so no locking is needed and the work
/// spreads across all cores. Destination spans are fixed before this runs, so the
/// arena is only ever read here.
fn copy_tensor(
    arena: &WeightArena,
    file: &File,
    spans: &[ArenaSpan],
    file_offset: u64,
    len: u64,
    convert_bf16: bool,
) -> Result<(), WeightError> {
    let len = usize::try_from(len).map_err(|_| WeightError::ArithmeticOverflow)?;
    if !convert_bf16 {
        return arena.read_into(file, spans, file_offset, len as u64);
    }
    if !len.is_multiple_of(2) {
        return Err(WeightError::ArithmeticOverflow);
    }
    // bfloat16 needs rewriting to f16, so stage one chunk at a time and convert
    // in place: bfloat16 is the top 16 bits of an f32, so each step is a shift
    // plus f32 -> f16 rounding.
    let mut scratch = vec![0_u8; COPY_CHUNK_BYTES];
    let mut offset = 0_usize;
    while offset < len {
        let take = COPY_CHUNK_BYTES.min(len - offset);
        read_exact_at(
            file,
            scratch.as_mut_ptr(),
            take as u64,
            file_offset + offset as u64,
        )?;
        for pair in scratch[..take].chunks_exact_mut(2) {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let value = half::f16::from_f32(f32::from_bits(u32::from(bits) << 16)).to_bits();
            pair.copy_from_slice(&value.to_le_bytes());
        }
        arena.write_at(spans, offset as u64, &scratch[..take])?;
        offset += take;
    }
    Ok(())
}

/// `pread` exactly `len` bytes into `destination`, retrying short reads.
fn read_exact_at(
    file: &File,
    destination: *mut u8,
    len: u64,
    offset: u64,
) -> Result<(), WeightError> {
    let total = usize::try_from(len).map_err(|_| WeightError::ArithmeticOverflow)?;
    let mut filled = 0_usize;
    while filled < total {
        let slice =
            unsafe { std::slice::from_raw_parts_mut(destination.add(filled), total - filled) };
        let read = file
            .read_at(slice, offset + filled as u64)
            .map_err(|source| WeightError::Io {
                path: PathBuf::from("<shard>"),
                source,
            })?;
        if read == 0 {
            return Err(WeightError::ArithmeticOverflow);
        }
        filled += read;
    }
    Ok(())
}

/// A safetensors shard opened for positional reads.
///
/// Bulk tensor bytes are pulled with `read_at` straight into their destination
/// Metal buffers. Reading through an mmap instead costs one minor fault per page;
/// with several threads faulting unrelated file offsets at once the kernel's
/// readahead collapses and a full text load drops to roughly a quarter of the
/// drive's sequential rate, while `read_at` keeps the same data moving at full
/// sequential speed.
struct ShardReader {
    file: File,
    pointer: NonNull<u8>,
    length: usize,
    length_u64: u64,
    path: PathBuf,
}

impl ShardReader {
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
        if length == 0 {
            return Err(WeightError::Mmap {
                path: path.to_path_buf(),
                message: "cannot read an empty shard".to_string(),
            });
        }
        let length = usize::try_from(length).map_err(|_| WeightError::Mmap {
            path: path.to_path_buf(),
            message: "shard does not fit in the address space".to_string(),
        })?;
        // Map the shard read-only so large tensors can be bound straight from the
        // page cache instead of being copied into anonymous memory.
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
        let pointer = NonNull::new(pointer.cast::<u8>()).ok_or_else(|| WeightError::Mmap {
            path: path.to_path_buf(),
            message: "mmap returned null".to_string(),
        })?;
        // Pre-fault clean pages sequentially to avoid minor faults on hot path.
        unsafe {
            libc::madvise(pointer.as_ptr().cast(), length, libc::MADV_WILLNEED);
        }
        Ok(Self {
            file,
            pointer,
            length,
            length_u64: length as u64,
            path: path.to_path_buf(),
        })
    }

    /// Wrap `[offset, offset+len)` of this shard in a Metal buffer without
    /// copying, returning the buffer and the tensor's byte offset inside it.
    ///
    /// The mapping is page aligned, but a safetensors tensor offset is not, so
    /// the buffer starts at the page-aligned address below the tensor and the
    /// caller binds at the returned offset.
    fn no_copy_view(
        &self,
        device: &metal::Device,
        offset: u64,
        len: u64,
    ) -> Result<(Buffer, u64), WeightError> {
        self.check_range(offset, len)?;
        let base = self.pointer.as_ptr() as usize;
        let start = base
            .checked_add(offset as usize)
            .ok_or(WeightError::ArithmeticOverflow)?;
        let aligned = start & !(PAGE_ALIGN - 1);
        let delta = start - aligned;
        let end = start
            .checked_add(len as usize)
            .ok_or(WeightError::ArithmeticOverflow)?;
        if end > base + self.length {
            return Err(WeightError::ArithmeticOverflow);
        }
        let buffer = device.new_buffer_with_bytes_no_copy(
            aligned as *const std::ffi::c_void,
            (end - aligned) as u64,
            metal::MTLResourceOptions::StorageModeShared
                | metal::MTLResourceOptions::HazardTrackingModeUntracked,
            None,
        );
        Ok((buffer, delta as u64))
    }

    fn check_range(&self, offset: u64, len: u64) -> Result<(), WeightError> {
        let end = offset
            .checked_add(len)
            .ok_or(WeightError::ArithmeticOverflow)?;
        if end > self.length_u64 {
            return Err(WeightError::Range {
                shard: self.path.display().to_string(),
            });
        }
        Ok(())
    }
}

// The mapping is PROT_READ/MAP_PRIVATE and never written, so sharing the reader
// across threads (and handing those pages to the GPU) is sound.
unsafe impl Sync for ShardReader {}
unsafe impl Send for ShardReader {}

impl Drop for ShardReader {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length);
        }
    }
}

const PAGE_ALIGN: usize = 4096;

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
        // Default packed trellis is zero-copy file-backed to prevent swap thrashing.
        // KAPLUMBA_RESIDENT_WEIGHTS forces arena copy.
        let force_resident = std::env::var_os("KAPLUMBA_RESIDENT_WEIGHTS").is_some();
        if force_resident {
            assert!(
                !tensor.spans.is_empty(),
                "forced-resident trellis must be resident in the arena"
            );
            let buffer = weights
                .tensor_buffer("model.language_model.layers.0.self_attn.q_proj.trellis")
                .unwrap();
            let byte_len = (tensor.end - tensor.start) as usize;
            let mapped =
                unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, byte_len) };
            assert_eq!(mapped, &[0x34, 0x12, 0x78, 0x56]);
        } else {
            assert!(
                tensor.spans.is_empty(),
                "default zero-copy trellis should not occupy arena space"
            );
        }
        let view = weights
            .zero_copy_tensor_buffer("model.language_model.layers.0.self_attn.q_proj.trellis")
            .unwrap()
            .expect("trellis must expose a zero-copy view");
        let (buffer, offset) = view;
        let mapped = unsafe {
            std::slice::from_raw_parts((buffer.contents() as *const u8).add(offset as usize), 4)
        };
        assert_eq!(mapped.to_vec(), vec![0x34, 0x12, 0x78, 0x56]);
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
    fn dedicated_tensors_are_single_addressable_buffers() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("dedicated");
        let bytes = vec![0x5A_u8; 8 * 1024 * 1024 + 2];
        let selection = selection_for(
            &dir,
            vec![(
                "model.language_model.norm.weight",
                DType::F16,
                vec![bytes.len() / 2],
                bytes.clone(),
            )],
        );
        let weights =
            TextWeights::load_dedicated(&dir, &selection, context, bytes.len() as u64).unwrap();
        let tensor = weights.tensor("model.language_model.norm.weight").unwrap();
        assert_eq!(tensor.spans.len(), 1);
        assert_eq!(tensor.spans[0].offset, 0);
        assert_eq!(
            weights
                .tensor_buffer(tensor.name.as_str())
                .unwrap()
                .length(),
            bytes.len() as u64
        );
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
