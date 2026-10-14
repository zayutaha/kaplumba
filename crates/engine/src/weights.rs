use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Mutex;

use metal::foreign_types::ForeignType;
use metal::Buffer;
use rayon::prelude::*;
use thiserror::Error;

use crate::metal::{DEFAULT_RESIDENT_LIMIT_BYTES, DEFAULT_SEGMENT_BYTES, MetalContext, MetalError};
use crate::model::EXL3_PACKED_ALIGNMENT;
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

    pub fn create_residency_set(&self) -> Result<crate::metal::ResidencySet, WeightError> {
        self.create_residency_set_with(&[])
    }

    /// [`Self::create_residency_set`] plus extra allocations.
    ///
    /// R4: the zero-copy weight views are `StorageModeShared` buffers over the
    /// shard mappings and must sit in the same `MTLResidencySet` as the arena
    /// segments, so the driver's GPU working set covers *all* weights -- a
    /// resident arena with an unregistered mapped trellis would still let the
    /// GPU re-fault those pages token after token.
    pub fn create_residency_set_with(
        &self,
        extra: &[Buffer],
    ) -> Result<crate::metal::ResidencySet, WeightError> {
        let set = crate::metal::ResidencySet::new(
            self.context.device(),
            self.segments.len() + extra.len(),
        )
        .map_err(WeightError::Metal)?;
        for seg in &self.segments {
            set.add_allocation(&seg.buffer);
        }
        for buffer in extra {
            set.add_allocation(buffer);
        }
        set.commit();
        set.request_residency();
        Ok(set)
    }
}

fn span_total(spans: &[ArenaSpan]) -> Result<u64, WeightError> {
    spans.iter().try_fold(0_u64, |total, span| {
        total
            .checked_add(span.len)
            .ok_or(WeightError::ArithmeticOverflow)
    })
}

fn checked_page_round_up(value: usize, page: usize) -> Result<usize, WeightError> {
    if page == 0 || !page.is_power_of_two() {
        return Err(WeightError::InvalidSegmentSize(page as u64));
    }
    value
        .checked_add(page - 1)
        .map(|rounded| rounded & !(page - 1))
        .ok_or(WeightError::ArithmeticOverflow)
}

/// Outward-rounded page range for a byte interval inside a mapped shard.
/// `mapping_len` may end in a partial file page; the kernel still maps that
/// final page, so clamp to the rounded mapping extent rather than raw EOF.
fn page_rounded_mapping_range(
    mapping_base: usize,
    mapping_len: usize,
    start_offset: usize,
    end_offset: usize,
    page: usize,
) -> Result<Option<(usize, usize)>, WeightError> {
    if end_offset < start_offset || end_offset > mapping_len {
        return Err(WeightError::Range {
            shard: "page range outside mapped shard".to_string(),
        });
    }
    if start_offset == end_offset {
        return Ok(None);
    }
    let mapping_end = mapping_base
        .checked_add(mapping_len)
        .ok_or(WeightError::ArithmeticOverflow)?;
    let mapping_page_end = checked_page_round_up(mapping_end, page)?;
    let absolute_start = mapping_base
        .checked_add(start_offset)
        .ok_or(WeightError::ArithmeticOverflow)?;
    let absolute_end = mapping_base
        .checked_add(end_offset)
        .ok_or(WeightError::ArithmeticOverflow)?;
    let start = absolute_start & !(page - 1);
    let end = checked_page_round_up(absolute_end, page)?.min(mapping_page_end);
    if end <= start {
        Ok(None)
    } else {
        Ok(Some((start, end)))
    }
}

/// Merge touching/overlapping page intervals, rejecting malformed boundaries.
fn merge_page_ranges(ranges: &[(usize, usize)]) -> Result<Vec<(usize, usize)>, WeightError> {
    let mut sorted = ranges.to_vec();
    if sorted.iter().any(|(start, end)| end < start) {
        return Err(WeightError::InvalidSpan {
            name: "reversed lock range".to_string(),
        });
    }
    sorted.retain(|(start, end)| start != end);
    sorted.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(sorted.len());
    for (start, end) in sorted {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    Ok(merged)
}

fn unique_page_bytes(ranges: &[(usize, usize)]) -> Result<u64, WeightError> {
    merge_page_ranges(ranges)?
        .into_iter()
        .try_fold(0_u64, |total, (start, end)| {
            let length = u64::try_from(end - start).map_err(|_| WeightError::ArithmeticOverflow)?;
            total
                .checked_add(length)
                .ok_or(WeightError::ArithmeticOverflow)
        })
}

fn locked_tensor_prefix_bytes(
    tensor_start: usize,
    tensor_end: usize,
    locked_until: usize,
) -> Result<usize, WeightError> {
    if tensor_end < tensor_start {
        return Err(WeightError::InvalidSpan {
            name: "reversed tensor range".to_string(),
        });
    }
    Ok(tensor_end.min(locked_until).saturating_sub(tensor_start))
}

/// Per-model report of which mapped packed-weight pages were selected and
/// successfully held. Tensor byte fields can differ from page-byte fields
/// because VM locking rounds outward to complete pages.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MlockSummary {
    pub model: String,
    pub eligible_page_bytes: u64,
    pub requested_page_bytes: u64,
    pub attempted_page_bytes: u64,
    pub locked_page_bytes: u64,
    pub failed_page_bytes: u64,
    pub evictable_page_bytes: u64,
    pub disabled_page_bytes: u64,
    /// Source bytes skipped because they are arena-copied, misaligned for the
    /// zero-copy packed path, or unavailable in a retained shard mapping.
    pub excluded_tensor_bytes: u64,
    /// Source bytes already copied into arena spans. This is a subset of the
    /// excluded source bytes above.
    pub copied_tensor_bytes: u64,
    pub failure: Option<String>,
}

fn format_mlock_summary(summary: &MlockSummary) -> String {
    format!(
        "mlock summary model={} eligible_page_bytes={} requested_page_bytes={} attempted_page_bytes={} locked_page_bytes={} failed_page_bytes={} evictable_page_bytes={} disabled_page_bytes={} excluded_tensor_bytes={} copied_tensor_bytes={} failure={}",
        summary.model,
        summary.eligible_page_bytes,
        summary.requested_page_bytes,
        summary.attempted_page_bytes,
        summary.locked_page_bytes,
        summary.failed_page_bytes,
        summary.evictable_page_bytes,
        summary.disabled_page_bytes,
        summary.excluded_tensor_bytes,
        summary.copied_tensor_bytes,
        summary.failure.as_deref().unwrap_or("none")
    )
}

pub struct TextWeights {
    arena: WeightArena,
    tensors: BTreeMap<String, LoadedTensor>,
    source_bytes: u64,
    /// Shard mappings stay resident for the lifetime of the weights because
    /// zero-copy Metal buffers alias these pages; dropping them would leave the
    /// GPU reading freed memory.
    shards: BTreeMap<String, ShardReader>,
    /// Every distinct zero-copy Metal view handed out by [`Self::raw_view`],
    /// kept alive here and registered with the residency set by
    /// [`Self::create_residency_set`] (R4). The views alias shard mappings that
    /// [`Self::shards`] keeps alive for exactly as long as these buffers can be
    /// touched.
    zero_copy_views: Mutex<Vec<Buffer>>,
    /// What [`Self::pin_trellis_pages`] actually wired: `(tensor name, bytes)`
    /// for every `.trellis` range a [`LlamaMlock`] locked successfully. Probes
    /// assert from this set that `embed_tokens` is never wired (R3) and that the
    /// wired total tracks the trellis size, without re-deriving the ranges.
    pinned: Mutex<Vec<(String, u64)>>,
    /// Unique page bytes successfully wired across disjoint shard runs.
    pinned_unique_bytes: Mutex<u64>,
    /// Full per-model mlock outcome, including the warning-and-continue path.
    mlock_summary: Mutex<MlockSummary>,
    /// R3: `Some(ok)` after [`Self::advise_embedding_random`] ran on a *mapped*
    /// embedding (`ok` = `madvise(MADV_RANDOM)` succeeded), `None` when the
    /// table is arena-resident and the advice does not apply.
    embedding_madvise_random: Mutex<Option<bool>>,
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
        let enabled = pin_trellis();
        Self::load_internal_with_mlock(
            model_dir,
            selection,
            context,
            limit_bytes,
            segment_bytes,
            dedicated,
            enabled,
            &mut system_mlock_range,
        )
    }

    fn load_internal_with_mlock(
        model_dir: &Path,
        selection: &TextTensorSelection,
        context: MetalContext,
        limit_bytes: u64,
        segment_bytes: u64,
        dedicated: bool,
        mlock_enabled: bool,
        mlock_range: &mut dyn FnMut(*const std::ffi::c_void, usize) -> io::Result<()>,
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
            let offload_bf16_embedding =
                offload_embedding && name == "model.language_model.embed_tokens.weight";
            // NOT gated on `!dedicated` -- see the regression note above in
            // `pin_trellis_pages`. `dedicated` is true for every production load
            // (`model.rs` uses `load_dedicated`), so the guard as written sent
            // all 12.04 GiB of trellis into dirty anonymous Metal shared buffers
            // and collapsed long-context decode.
            let offload_trellis = !force_resident && name.ends_with(".trellis");
            let skip_copy = offload_bf16_embedding || offload_trellis;
            let spans = if skip_copy {
                Vec::new()
            } else if dedicated || name.ends_with(".trellis") {
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
        // Retain only the shards that contain tensors with zero-copy views (e.g. embed_tokens).
        // Shards whose tensors are all copied into the arena are dropped immediately, unmapping
        // gigabytes of unused file memory from the process address space.
        let needed_shards: std::collections::BTreeSet<&str> = tensors
            .values()
            .filter(|t| t.spans.is_empty())
            .map(|t| t.shard.as_str())
            .collect();
        shards.retain(|name, _| needed_shards.contains(name.as_str()));

        let weights = Self {
            arena,
            tensors,
            source_bytes,
            shards,
            zero_copy_views: Mutex::new(Vec::new()),
            pinned: Mutex::new(Vec::new()),
            pinned_unique_bytes: Mutex::new(0),
            mlock_summary: Mutex::new(MlockSummary::default()),
            embedding_madvise_random: Mutex::new(None),
        };
        // R3 first: the embedding must never be wired, only advised for
        // single-row access. Then R2: wire the trellis.
        weights.advise_embedding_random();
        weights.pin_trellis_pages(model_dir, mlock_enabled, mlock_range);
        Ok(weights)
    }

    /// Pin the aligned mapped `.trellis` pages and report this model's outcome.
    ///
    /// # Why this exists -- and why `mlock` is back after `e1b05c1` removed it
    ///
    /// `e1b05c1` ("green memory residency") deleted this method and routed the
    /// 12.04 GiB of `.trellis` weights into dirty anonymous
    /// `MTLResourceStorageModeShared` Metal buffers, on the theory that "clean
    /// file-backed pages are the VM's easiest reclaim" and that dirty anonymous
    /// Metal buffers are not. **Measured on `Gxma-repacked`, 16 438-context
    /// `--chat --temperature 0`, that inverted the failure:**
    ///
    /// | context | step | tok/s |
    /// |---|---:|---:|
    /// | 16 438, trellis file-backed + pinned (833c2ce) | 76.3 ms | 13.1 |
    /// | 16 438, trellis in the shared arena (e1b05c1) | **2 721 ms** | **0.37** |
    ///
    /// The tell is that host encode stayed at 1.8 ms while the GPU wait went to
    /// 2 488 ms, and the machine showed **swap 4.35 GiB, compressor 8.1 GiB,
    /// 26 % free** -- against the spec's own claim of "36 % free, SOLID GREEN,
    /// zero swap growth". The spec's telemetry never measured *decode* at 17.5k;
    /// it measured memory at 17.5k and throughput on a 19-token prompt. So its
    /// R5 (">= 13.0 tok/s at 17 500 context") was violated by 35x and shipped
    /// anyway.
    ///
    /// **A dirty anonymous page has exactly two possible fates under pressure --
    /// compression or swap -- and both are paid per token.** `KAPLUMBA_RESIDENT_
    /// WEIGHTS=1` measured the same 0.41 tok/s collapse in `833c2ce` for the same
    /// reason. `madvise(MADV_WILLNEED)` faults pages in once and does not stop
    /// either (see `ShardReader::open`). `mlock` is the only property that
    /// actually holds, and it is deliberately scoped to the trellis:
    ///
    /// * The trellis is 12.04 GiB of the ~17.7 GiB hot set (the rest is ~3.5 GiB
    ///   of KV at 16.5k, 1.06 GiB of `lm_head`, ~1.1 GiB of arena), and it is
    ///   the bytes every token streams. Pinning the whole mapping instead put
    ///   the process at 21 GiB wired, which is not a safe place for this box.
    /// * `embed_tokens` (2.82 GB) stays offloaded exactly as before: one row is
    ///   read per token through the zero-copy in-kernel path, so eviction costs
    ///   a 10.7 KB page-in, not a re-stream.
    ///
    /// Pages remain read-only file-backed pages. `mlock` can fail against
    /// `RLIMIT_MEMLOCK` or the kernel's `user_wire_limit`; failure does not
    /// change correctness, and this method reports each load's shortfall.
    ///
    /// **Amends `spec/green-memory-residency` R1** ("the engine must not call
    /// `libc::mlock` under any circumstances"), which is unsatisfiable alongside
    /// its own R5 on this box: a 17.7 GiB hot set on 24 GiB will be compressed
    /// unless those pages are wired. R5 was the user-facing requirement; R1 was
    /// a means. The conflict and the numbers above are recorded there too.
    fn pin_trellis_pages(
        &self,
        model_dir: &Path,
        enabled: bool,
        mlock_range: &mut dyn FnMut(*const std::ffi::c_void, usize) -> io::Result<()>,
    ) {
        // Group the mapped `.trellis` ranges by shard, then merge only ranges
        // that *touch* after the outward rounding: one `LlamaMlock` per merged
        // run, grown incrementally once per tensor -- the same driver pattern
        // as `llama-model-loader.cpp:1662`. A gap between two ranges means
        // something else lives in between, so runs never bridge a gap; that is
        // what keeps R3 ("`embed_tokens` is never wired") true even when the
        // embedding shares a shard with trellis tensors.
        let mut by_shard: BTreeMap<String, Vec<(usize, usize, String)>> = BTreeMap::new();
        let mut excluded_tensor_bytes = 0_u64;
        let mut copied_tensor_bytes = 0_u64;
        for (name, tensor) in &self.tensors {
            if !name.ends_with(".trellis") {
                continue;
            }
            if !tensor.spans.is_empty() {
                excluded_tensor_bytes = excluded_tensor_bytes
                    .checked_add(tensor.source_len)
                    .expect("excluded packed-weight bytes fit u64");
                copied_tensor_bytes = copied_tensor_bytes
                    .checked_add(tensor.source_len)
                    .expect("copied packed-weight bytes fit u64");
                continue;
            }
            let Some(reader) = self.shards.get(&tensor.shard) else {
                excluded_tensor_bytes = excluded_tensor_bytes
                    .checked_add(tensor.source_len)
                    .expect("excluded packed-weight bytes fit u64");
                continue;
            };
            // `mlock` wants a page-aligned start, and a safetensors tensor
            // offset is not page-aligned, so the range is rounded *outward*.
            // That can pin a page shared with an adjacent side-tensor; those
            // pages are read every token too, so the overshoot is harmless.
            let base = reader.pointer.as_ptr() as usize;
            let start_offset = usize::try_from(tensor.start)
                .expect("validated tensor offset fits the mapped address space");
            let end_offset = usize::try_from(tensor.end)
                .expect("validated tensor end fits the mapped address space");
            // The OS rounds a file mapping to full pages even when its length
            // ends mid-page. Keep the final page lockable, including a shard
            // smaller than one page.
            let Some((start, end)) = page_rounded_mapping_range(
                base,
                reader.length,
                start_offset,
                end_offset,
                page_size(),
            )
            .expect("tensor range was validated when the shard was loaded") else {
                continue;
            };
            // A tensor whose absolute shard offset is not a multiple of
            // `EXL3_PACKED_ALIGNMENT` (4) cannot be bound zero-copy:
            // `align_packed_view` (`model.rs:5689`) copies it into an anonymous
            // realigned buffer and the GPU reads THAT for the whole session,
            // never these file pages. Wiring them would wire a dead second
            // copy. Measured on `Gxma`: 311 of 411 tensors, 8.61 GiB of wired
            // file pages living beside 8.61 GiB of live anonymous copies --
            // 19.8 GiB of process for an 11.2 GiB model, with the *live*
            // copies the side the compressor ate (9040 MB compressed, decode
            // crawled). `pin_realigned_copy` wires the live copy instead, and
            // this tensor's file range is then never faulted at all.
            if tensor.start % EXL3_PACKED_ALIGNMENT != 0 {
                excluded_tensor_bytes = excluded_tensor_bytes
                    .checked_add(tensor.source_len)
                    .expect("excluded packed-weight bytes fit u64");
                continue;
            }
            by_shard
                .entry(tensor.shard.clone())
                .or_default()
                .push((start, end, name.clone()));
        }

        let eligible_ranges: Vec<_> = by_shard
            .values()
            .flat_map(|ranges| ranges.iter().map(|(start, end, _)| (*start, *end)))
            .collect();
        let eligible_page_bytes = unique_page_bytes(&eligible_ranges)
            .expect("validated eligible page ranges fit the address space");
        let mut summary = MlockSummary {
            model: model_dir.display().to_string(),
            eligible_page_bytes,
            requested_page_bytes: if enabled { eligible_page_bytes } else { 0 },
            disabled_page_bytes: if enabled { 0 } else { eligible_page_bytes },
            excluded_tensor_bytes,
            copied_tensor_bytes,
            ..MlockSummary::default()
        };

        let mut pinned_bytes = 0_u64;
        let mut pinned_tensors = 0_usize;
        let mut record = self
            .pinned
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut unique_bytes_this_load = 0_u64;
        for (_shard, mut ranges) in by_shard {
            if !enabled {
                continue;
            }
            ranges.sort();
            let mut runs: Vec<Vec<(usize, usize, String)>> = Vec::new();
            for range in ranges {
                match runs.last_mut() {
                    Some(run) if range.0 <= run.last().map_or(usize::MAX, |last| last.1) => {
                        run.push(range);
                    }
                    _ => runs.push(vec![range]),
                }
            }
            for mut run in runs {
                // `ranges` was sorted by start, so the first element is the
                // run's lowest address -- the lock's base.
                let run_start = run.first().map_or(0, |first| first.0);
                run.sort_by_key(|range| range.1);
                let run_page_ranges: Vec<_> =
                    run.iter().map(|(start, end, _)| (*start, *end)).collect();
                let planned_unique_bytes = unique_page_bytes(&run_page_ranges)
                    .expect("validated page ranges fit the address space");
                let mut lock = LlamaMlock::init(run_start as *const std::ffi::c_void);
                for range in &run {
                    lock.grow_to_with(range.1 - run_start, mlock_range);
                }
                // A failed grow leaves earlier pages wired. `size()` is the
                // exact successful prefix and counts overlapping tensor page
                // windows once.
                debug_assert!(
                    u64::try_from(lock.size()).expect("lock size fits u64") <= planned_unique_bytes
                );
                let actual_locked_bytes = u64::try_from(lock.size())
                    .expect("successful lock size fits the accounting type");
                let attempted_bytes = u64::try_from(lock.attempted_bytes())
                    .expect("attempted lock size fits the accounting type");
                summary.attempted_page_bytes = summary
                    .attempted_page_bytes
                    .checked_add(attempted_bytes)
                    .expect("attempted lock bytes fit u64");
                if summary.failure.is_none() {
                    summary.failure = lock.failure().map(|failure| failure.to_string());
                }
                unique_bytes_this_load = unique_bytes_this_load
                    .checked_add(actual_locked_bytes)
                    .expect("mapped lock bytes fit the address space");
                let locked_upto = run_start
                    .checked_add(lock.size())
                    .expect("successful lock prefix fits the address space");
                let mapping_base = self.shards[&_shard].pointer.as_ptr() as usize;
                for (_start, _end, name) in run {
                    let tensor = &self.tensors[&name];
                    let tensor_start_offset = usize::try_from(tensor.start)
                        .expect("validated tensor start fits the mapped address space");
                    let tensor_end_offset = usize::try_from(tensor.end)
                        .expect("validated tensor end fits the mapped address space");
                    let tensor_start = mapping_base
                        .checked_add(tensor_start_offset)
                        .expect("validated tensor start fits the address space");
                    let tensor_end = mapping_base
                        .checked_add(tensor_end_offset)
                        .expect("validated tensor end fits the address space");
                    let bytes = locked_tensor_prefix_bytes(tensor_start, tensor_end, locked_upto)
                        .expect("validated tensor range is ordered");
                    if bytes > 0 {
                        let bytes =
                            u64::try_from(bytes).expect("tensor prefix fits the accounting type");
                        pinned_bytes = pinned_bytes
                            .checked_add(bytes)
                            .expect("tensor lock accounting fits u64");
                        pinned_tensors = pinned_tensors
                            .checked_add(1)
                            .expect("locked tensor count fits usize");
                        record.push((name, bytes));
                    }
                }
            }
        }
        drop(record);
        summary.locked_page_bytes = unique_bytes_this_load;
        summary.failed_page_bytes = summary
            .attempted_page_bytes
            .saturating_sub(summary.locked_page_bytes);
        summary.evictable_page_bytes = summary
            .eligible_page_bytes
            .saturating_sub(summary.locked_page_bytes);
        *self
            .pinned_unique_bytes
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = unique_bytes_this_load;
        *self
            .mlock_summary
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = summary.clone();
        if summary.disabled_page_bytes > 0
            || summary.locked_page_bytes < summary.requested_page_bytes
            || summary.excluded_tensor_bytes > 0
        {
            eprintln!(
                "warning: {}; continuing with evictable or excluded packed weights. Increase vm.user_wire_limit / vm.global_user_wire_limit or RLIMIT_MEMLOCK if the lock failure is unexpected",
                format_mlock_summary(&summary)
            );
        } else {
            eprintln!("{}", format_mlock_summary(&summary));
        }
        if pinned_tensors > 0 && std::env::var_os("KAPLUMBA_LOAD_PROFILE").is_some() {
            eprintln!(
                "load profile: pinned {pinned_tensors} trellis tensors, {:.2} GiB wired",
                pinned_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }

    /// R3: keep `embed_tokens` an *unwired* clean mapping and turn off
    /// readahead on it.
    ///
    /// The table serves one row per token (~10.7 KB), so readahead would pull
    /// an order of magnitude more per token for nothing -- and, decisively,
    /// the pages must stay reclaimable to keep the process under Darwin's
    /// `vm.user_wire_limit` (~18.24 GiB) alongside the wired trellis. This is
    /// deliberately the opposite treatment from `pin_trellis_pages`, which is
    /// why the two live in separate passes over different name sets.
    fn advise_embedding_random(&self) {
        const EMBED: &str = "model.language_model.embed_tokens.weight";
        let Some(tensor) = self.tensors.get(EMBED) else {
            return;
        };
        if !tensor.spans.is_empty() {
            // Resident in the arena; there is no file mapping to advise.
            return;
        }
        let result = self
            .shards
            .get(&tensor.shard)
            .ok_or_else(|| WeightError::Range {
                shard: tensor.shard.clone(),
            })
            .and_then(|reader| {
                reader.advise_random(tensor.start, tensor.end - tensor.start)
            });
        let ok = match result {
            Ok(ok) => ok,
            Err(_) => false,
        };
        if !ok {
            eprintln!(
                "warning: madvise(MADV_RANDOM) on {EMBED} failed; the mapping is \
                 correct but the VM will read ahead on a one-row-per-token table"
            );
        }
        *self
            .embedding_madvise_random
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(ok);
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

    pub fn retained_shards(&self) -> usize {
        self.shards.len()
    }

    /// Arena segments **plus** every zero-copy view handed out so far (R4).
    ///
    /// Called by `TextModel::finish_construction` (`model.rs`) after the layer
    /// build, which is after every trellis view and the embedding view exist,
    /// so one set covers the whole weight footprint.
    pub fn create_residency_set(&self) -> Result<crate::metal::ResidencySet, WeightError> {
        let views = self
            .zero_copy_views
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.arena.create_residency_set_with(&views)
    }

    /// Number of distinct zero-copy views registered so far (R4 probe hook).
    pub fn zero_copy_view_count(&self) -> usize {
        self.zero_copy_views
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len()
    }

    /// `(name, bytes)` of every `.trellis` range actually wired by
    /// [`Self::pin_trellis_pages`] (R2/R3 probe hook).
    pub fn pinned_tensors(&self) -> Vec<(String, u64)> {
        self.pinned
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// Sum of source tensor bytes covered by successful locks. Overlapping
    /// tensor byte windows can be counted more than once; use
    /// [`Self::pinned_unique_bytes`] for the unique page total.
    pub fn pinned_bytes(&self) -> u64 {
        self.pinned
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .map(|(_, bytes)| *bytes)
            .sum()
    }

    /// Unique page bytes successfully wired by this load (R2/R3 probe hook).
    pub fn pinned_unique_bytes(&self) -> u64 {
        *self
            .pinned_unique_bytes
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// Lock outcomes for this specific model load. Each `TextWeights` owns its
    /// own summary so a previous load cannot suppress a later warning.
    pub fn mlock_summary(&self) -> MlockSummary {
        self.mlock_summary
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// R3 probe hook: see [`Self::embedding_madvise_random`].
    pub fn embedding_madvise_random(&self) -> Option<bool> {
        *self
            .embedding_madvise_random
            .lock()
            .unwrap_or_else(|error| error.into_inner())
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
        let (buffer, offset) = reader.no_copy_view(device, start, end - start)?;
        // R4 ledger: remember the view so `create_residency_set` can register
        // it (and so its lifetime is pinned to this struct's, independent of
        // whether the caller keeps the buffer it received).
        {
            let mut views = self
                .zero_copy_views
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !views
                .iter()
                .any(|existing| existing.as_ptr() == buffer.as_ptr())
            {
                views.push(buffer.clone());
            }
        }
        Ok(Some((buffer, offset)))
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

/// Whether `.trellis` pages should be pinned with `mlock` at load.
///
/// On by default; `KAPLUMBA_NO_PIN_TRELLIS` turns it off, which is the escape
/// hatch for a box whose `vm.user_wire_limit` cannot take the ~11 GiB. Read once
/// into a `OnceLock` so the answer cannot change between the two shards of one
/// checkpoint, and so the hot paths that ask for it pay a branch rather than a
/// `getenv`.
fn pin_trellis() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("KAPLUMBA_NO_PIN_TRELLIS").is_none())
}

/// Port of `llama.cpp`'s `llama_mlock` (`reference/llama.cpp/src/llama-mmap.cpp:694`),
/// the memory-locking engine behind llama.cpp's `--mlock` (R2).
///
/// Semantics kept 1:1 with `llama_mlock::impl`:
///
/// * [`Self::lock_granularity`] is `sysconf(_SC_PAGESIZE)` (16384 on this
///   box), and [`Self::grow_to`] rounds the target *up* to it before locking.
/// * Locking is **incremental**: only `[addr + size, target)` is passed to
///   `mlock`, so a later grow wires just the new slice -- the property
///   `llama-model-loader.cpp:1662` relies on when it grows one mapping once
///   per tensor (`lmlock->grow_to(weight->offs + n_size)`).
/// * A failed `mlock` (`ENOMEM` from `vm.user_wire_limit`/`RLIMIT_MEMLOCK`, or
///   `EPERM`) sets `failed_already`: every later `grow_to` on this range is a
///   no-op, the successful prefix stays recorded, and the model load continues.
///   The caller reports the failure in a per-model summary.
///
/// Deliberate divergences from llama.cpp:
///
/// * **No `Drop` unlock.** `llama_mlock` owns memory it releases; here the
///   region belongs to `ShardReader`'s `mmap`, and unlocking on drop would undo
///   `pin_trellis_pages` the moment it returned. `munmap` in `ShardReader`s
///   `Drop` releases the wire.
pub struct LlamaMlock {
    addr: usize,
    size: usize,
    attempted_bytes: usize,
    failed_already: bool,
    failure: Option<String>,
}

impl LlamaMlock {
    /// Bind the lock to `addr`; nothing is wired until [`Self::grow_to`].
    pub fn init(addr: *const std::ffi::c_void) -> Self {
        Self {
            addr: addr as usize,
            size: 0,
            attempted_bytes: 0,
            failed_already: false,
            failure: None,
        }
    }

    /// `sysconf(_SC_PAGESIZE)`, llama.cpp's POSIX `lock_granularity()`.
    pub fn lock_granularity() -> usize {
        page_size()
    }

    /// Wire `[addr, target_size)` if that reaches past what is already locked.
    ///
    /// `saturating_add` instead of llama.cpp's plain `+` before the rounding:
    /// an absurd target (the soft-failure case `probe_llama_mlock` exercises)
    /// must reach `mlock` and come back `ENOMEM`, not overflow-panic in a debug
    /// build.
    pub fn grow_to(&mut self, target_size: usize) {
        self.grow_to_with(target_size, &mut system_mlock_range);
    }

    fn grow_to_with(
        &mut self,
        target_size: usize,
        mlock_range: &mut dyn FnMut(*const std::ffi::c_void, usize) -> io::Result<()>,
    ) {
        if self.failed_already {
            return;
        }
        let granularity = Self::lock_granularity();
        let target_size = target_size.saturating_add(granularity - 1) & !(granularity - 1);
        if target_size > self.size {
            let slice = (self.addr + self.size) as *const std::ffi::c_void;
            let len = target_size - self.size;
            self.attempted_bytes = self
                .attempted_bytes
                .checked_add(len)
                .expect("attempted mlock bytes fit usize");
            match mlock_range(slice, len) {
                Ok(()) => self.size = target_size,
                Err(source) => {
                    self.failed_already = true;
                    self.failure = Some(match source.raw_os_error() {
                        Some(code) => format!("errno={code}: {source}"),
                        None => source.to_string(),
                    });
                }
            }
        }
    }

    /// Bytes currently wired by this lock (a multiple of the granularity after
    /// every successful `grow_to`).
    pub fn size(&self) -> usize {
        self.size
    }

    /// `true` once any `grow_to` failed; later calls are no-ops (R2: graceful
    /// soft clamp instead of an aborted model load).
    pub fn failed(&self) -> bool {
        self.failed_already
    }

    fn attempted_bytes(&self) -> usize {
        self.attempted_bytes
    }

    fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }
}

fn system_mlock_range(address: *const std::ffi::c_void, length: usize) -> io::Result<()> {
    if unsafe { libc::mlock(address, length) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
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

/// A safetensors shard opened for positional reads and zero-copy Metal binding.
///
/// Bulk tensor bytes are pulled with `read_at` straight into their destination
/// Metal arena buffers. Zero-copy tensors (trellis and embed_tokens) are bound
/// directly out of page-rounded tensor views of the shard.
struct ShardReader {
    metal_views: Mutex<BTreeMap<(u64, u64), Buffer>>,
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
        // On Darwin, set F_NOCACHE to disable unified page cache pollution during pread
        #[cfg(target_os = "macos")]
        unsafe {
            libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
        }
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
        // Keep the shard read-only and shared so locking its file-backed pages
        // does not create private COW copies. Large tensors bind straight from
        // this mapping instead of being copied into anonymous memory.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_SHARED,
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
        Ok(Self {
            metal_views: Mutex::new(BTreeMap::new()),
            file,
            pointer,
            length,
            length_u64: length as u64,
            path: path.to_path_buf(),
        })
    }

    /// Wrap `[offset, offset+len)` of this shard in a Metal buffer without
    /// copying. Metal wires a bound resource's full span, so exposing the whole
    /// shard would also wire unrelated embedding/multimodal pages. Cache only
    /// the outward page-rounded range and return the tensor's local offset.
    fn no_copy_view(
        &self,
        device: &metal::Device,
        offset: u64,
        len: u64,
    ) -> Result<(Buffer, u64), WeightError> {
        self.check_range(offset, len)?;
        if len == 0 {
            return Err(WeightError::InvalidSegmentSize(0));
        }
        let page = page_size() as u64;
        let start = offset / page * page;
        let end = offset.checked_add(len).ok_or(WeightError::ArithmeticOverflow)?;
        let rounded_end = end.checked_add(page - 1)
            .ok_or(WeightError::ArithmeticOverflow)? / page * page;
        let length = rounded_end - start;
        if length > device.max_buffer_length() {
            return Err(WeightError::InvalidSegmentSize(length));
        }
        let mut views = self.metal_views.lock().unwrap_or_else(|e| e.into_inner());
        let buffer = views.entry((start, length)).or_insert_with(|| {
            device.new_buffer_with_bytes_no_copy(
                unsafe { self.pointer.as_ptr().add(start as usize) }.cast(),
                length,
                metal::MTLResourceOptions::StorageModeShared
                    | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                None,
            )
        });
        Ok((buffer.clone(), offset - start))
    }

    /// R3: `madvise(MADV_RANDOM)` over the tensor's outward-rounded page range
    /// (no readahead for a one-row-at-a-time region). Returns whether the
    /// advice was accepted; a `false`/`Err` never fails the load, because the
    /// mapping is correct either way -- only the readahead waste differs.
    fn advise_random(&self, offset: u64, len: u64) -> Result<bool, WeightError> {
        self.check_range(offset, len)?;
        let page = page_size();
        let base = self.pointer.as_ptr() as usize;
        let start = (base + offset as usize) & !(page - 1);
        let map_end = base + self.length;
        let end = (base + offset as usize + len as usize)
            .next_multiple_of(page)
            .min(map_end);
        if end <= start {
            return Err(WeightError::Range {
                shard: self.path.display().to_string(),
            });
        }
        let rc = unsafe {
            libc::madvise(
                start as *mut std::ffi::c_void,
                end - start,
                libc::MADV_RANDOM,
            )
        };
        Ok(rc == 0)
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

// The mapping is PROT_READ/MAP_SHARED and never written, so sharing the reader
// across threads (and handing those pages to the GPU) is sound.
unsafe impl Sync for ShardReader {}
unsafe impl Send for ShardReader {}

impl Drop for ShardReader {
    fn drop(&mut self) {
        self.metal_views.get_mut().unwrap_or_else(|e| e.into_inner()).clear();
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length);
        }
    }
}

/// Dynamic page size query (16 KiB on Apple Silicon, sysconf(_SC_PAGESIZE)).
pub fn page_size() -> usize {
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if value > 0 {
        value as usize
    } else {
        16384
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
                DType::U8 => "U8",
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
        let mut map: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        for (name, _, _, _) in &entries {
            map.insert(name.to_string(), json!("model.safetensors"));
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
    fn unique_page_accounting_merges_disjoint_touching_and_overlapping_ranges() {
        assert_eq!(unique_page_bytes(&[]).unwrap(), 0, "empty lock set");
        assert_eq!(unique_page_bytes(&[(0, 10), (20, 30)]).unwrap(), 20);
        assert_eq!(unique_page_bytes(&[(0, 16), (16, 32)]).unwrap(), 32);
        assert_eq!(unique_page_bytes(&[(0, 16), (8, 24)]).unwrap(), 24);
        assert_eq!(
            merge_page_ranges(&[(0, 16), (16, 32)]).unwrap(),
            vec![(0, 32)]
        );
        assert_eq!(
            merge_page_ranges(&[(0, 16), (8, 24)]).unwrap(),
            vec![(0, 24)]
        );
        assert_eq!(
            merge_page_ranges(&[(8, 8)]).unwrap(),
            Vec::<(usize, usize)>::new()
        );
        assert!(
            merge_page_ranges(&[(9, 8)]).is_err(),
            "reversed interval must fail"
        );
    }

    #[test]
    fn page_rounding_handles_eof_empty_ranges_and_arithmetic_edges() {
        let base = 0x10_000;
        let page = 0x1000;
        // The file mapping ends 1 byte into its last page. The last tensor
        // locks the whole VM page that the kernel mapped for it.
        assert_eq!(
            page_rounded_mapping_range(base, 0x3001, 0x2ff0, 0x3001, page).unwrap(),
            Some((base + 0x2000, base + 0x4000))
        );
        assert_eq!(
            page_rounded_mapping_range(base, 4, 2, 2, page).unwrap(),
            None,
            "empty byte range must not lock a page"
        );
        assert!(page_rounded_mapping_range(base, 4, 2, 5, page).is_err());
        assert!(page_rounded_mapping_range(base, 4, 3, 2, page).is_err());
        assert!(checked_page_round_up(usize::MAX, page).is_err());
        assert!(page_rounded_mapping_range(usize::MAX - 1, 4, 0, 1, page).is_err());
    }

    #[test]
    fn partial_lock_prefix_accounts_only_tensor_bytes_it_reaches() {
        assert_eq!(locked_tensor_prefix_bytes(100, 200, 150).unwrap(), 50);
        assert_eq!(locked_tensor_prefix_bytes(100, 200, 200).unwrap(), 100);
        assert_eq!(locked_tensor_prefix_bytes(100, 200, 99).unwrap(), 0);
        assert_eq!(locked_tensor_prefix_bytes(100, 100, 150).unwrap(), 0);
        assert!(locked_tensor_prefix_bytes(200, 100, 150).is_err());
    }

    const LOCK_SELECTION_CHILD: &str = "KAPLUMBA_LOCK_SELECTION_CHILD";

    #[test]
    fn complete_disabled_or_copied_loads_report_their_lock_state() {
        for mode in ["complete", "disabled", "copied"] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "weights::tests::child_lock_selection_is_truthful",
                    "--nocapture",
                ])
                .env(LOCK_SELECTION_CHILD, "1")
                .env_remove("KAPLUMBA_NO_PIN_TRELLIS")
                .env_remove("KAPLUMBA_RESIDENT_WEIGHTS")
                .env_remove("KAPLUMBA_LOAD_PROFILE");
            if mode == "copied" {
                command.env("KAPLUMBA_RESIDENT_WEIGHTS", "1");
            } else if mode == "disabled" {
                command.env("KAPLUMBA_NO_PIN_TRELLIS", "1");
            }
            let output = command
                .output()
                .expect("failed to run lock-selection child");
            assert!(
                output.status.success(),
                "lock-selection child failed (mode={mode}): {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(
                stderr.matches("mlock summary model=").count(),
                1,
                "each {mode} load must emit one lock summary without profiling: {stderr}"
            );
            let page = page_size();
            match mode {
                "complete" => {
                    assert!(!stderr.contains("warning: mlock summary"));
                    assert!(stderr.contains(&format!("requested_page_bytes={page}")));
                    assert!(stderr.contains(&format!("locked_page_bytes={page}")));
                    assert!(stderr.contains("failure=none"));
                }
                "disabled" => {
                    assert!(stderr.contains("warning: mlock summary"));
                    assert!(stderr.contains(&format!("disabled_page_bytes={page}")));
                }
                "copied" => {
                    assert!(stderr.contains("warning: mlock summary"));
                    assert!(stderr.contains("copied_tensor_bytes=128"));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn child_lock_selection_is_truthful() {
        if std::env::var_os(LOCK_SELECTION_CHILD).is_none() {
            return;
        }
        let context = context_or_skip().expect("lock-selection child requires Metal");
        let name = "model.language_model.layers.0.self_attn.q_proj.trellis";
        let payload = vec![0x5a_u8; 128];
        let (dir, selection) = (1..65)
            .find_map(|pad| {
                let dir = temp_dir(&format!("lock-selection-child-{pad}"));
                let selection = selection_for(
                    &dir,
                    vec![
                        ("pad.bytes", DType::U8, vec![pad], vec![0_u8; pad]),
                        (name, DType::I16, vec![64], payload.clone()),
                    ],
                );
                if selection
                    .absolute_range(name)
                    .unwrap()
                    .0
                    .is_multiple_of(EXL3_PACKED_ALIGNMENT as u64)
                {
                    Some((dir, selection))
                } else {
                    fs::remove_dir_all(dir).unwrap();
                    None
                }
            })
            .expect("child fixture must contain an aligned packed range");
        let weights = TextWeights::load_dedicated(&dir, &selection, context, 1024 * 1024)
            .expect("fixture must still load when locking is disabled or weights are copied");
        let tensor = weights.tensor(name).unwrap();
        let copied = std::env::var_os("KAPLUMBA_RESIDENT_WEIGHTS").is_some();
        let disabled = std::env::var_os("KAPLUMBA_NO_PIN_TRELLIS").is_some();
        assert_eq!(!tensor.spans.is_empty(), copied);
        let summary = weights.mlock_summary();
        if copied {
            assert_eq!(weights.pinned_bytes(), 0);
            assert_eq!(weights.pinned_unique_bytes(), 0);
            assert_eq!(summary.copied_tensor_bytes, payload.len() as u64);
            assert_eq!(summary.excluded_tensor_bytes, payload.len() as u64);
            assert_eq!(summary.eligible_page_bytes, 0);
        } else if disabled {
            assert_eq!(weights.pinned_bytes(), 0);
            assert_eq!(weights.pinned_unique_bytes(), 0);
            assert_eq!(summary.eligible_page_bytes, page_size() as u64);
            assert_eq!(summary.requested_page_bytes, 0);
            assert_eq!(summary.disabled_page_bytes, page_size() as u64);
            assert_eq!(summary.evictable_page_bytes, page_size() as u64);
        } else {
            assert_eq!(weights.pinned_bytes(), payload.len() as u64);
            assert_eq!(weights.pinned_unique_bytes(), page_size() as u64);
            assert_eq!(summary.requested_page_bytes, page_size() as u64);
            assert_eq!(summary.locked_page_bytes, page_size() as u64);
            assert_eq!(summary.evictable_page_bytes, 0);
        }
        if copied {
            assert_eq!(weights.read_tensor(name).unwrap(), payload);
        } else {
            let (buffer, offset) = weights.zero_copy_tensor_buffer(name).unwrap().unwrap();
            let actual = unsafe {
                std::slice::from_raw_parts(
                    (buffer.contents() as *const u8).add(offset as usize),
                    payload.len(),
                )
            };
            assert_eq!(actual, payload.as_slice());
        }
        drop(weights);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mlock_failure_latches_and_retains_successful_prefix() {
        let page = LlamaMlock::lock_granularity();
        let mut first_failure = LlamaMlock::init(0x10_000 as *const std::ffi::c_void);
        let mut first_calls = 0;
        first_failure.grow_to_with(page, &mut |_, length| {
            first_calls += 1;
            assert_eq!(length, page);
            Err(io::Error::from_raw_os_error(libc::ENOMEM))
        });
        assert_eq!(first_failure.size(), 0);
        assert_eq!(first_failure.attempted_bytes(), page);
        assert!(first_failure.failed());
        assert!(
            first_failure
                .failure()
                .unwrap()
                .contains(&format!("errno={}", libc::ENOMEM))
        );
        first_failure.grow_to_with(2 * page, &mut |_, _| {
            panic!("grow after failure must not call mlock")
        });
        assert_eq!(first_calls, 1);

        let mut later_failure = LlamaMlock::init(0x20_000 as *const std::ffi::c_void);
        let mut calls = 0;
        let mut injected = |_, length| {
            calls += 1;
            if calls == 1 {
                assert_eq!(length, page);
                Ok(())
            } else {
                assert_eq!(length, 2 * page);
                Err(io::Error::from_raw_os_error(libc::EPERM))
            }
        };
        later_failure.grow_to_with(page, &mut injected);
        later_failure.grow_to_with(3 * page, &mut injected);
        later_failure.grow_to_with(4 * page, &mut injected);
        assert_eq!(later_failure.size(), page, "successful prefix must remain");
        assert_eq!(later_failure.attempted_bytes(), 3 * page);
        assert!(later_failure.failed());
        assert!(
            later_failure
                .failure()
                .unwrap()
                .contains(&format!("errno={}", libc::EPERM))
        );
        assert_eq!(calls, 2, "later grows after EPERM must be skipped");
    }

    #[test]
    fn partial_lock_loads_and_each_model_keeps_its_own_warning_summary() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "weights::tests::child_partial_lock_summaries_are_per_model",
                "--nocapture",
            ])
            .env("KAPLUMBA_PARTIAL_LOCK_SUMMARY_CHILD", "1")
            .output()
            .expect("failed to run partial-lock summary child");
        assert!(
            output.status.success(),
            "partial-lock child failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            stderr.matches("warning: mlock summary").count(),
            2,
            "each load must print a warning even after a previous load warned: {stderr}"
        );
        assert!(stderr.contains(&format!("errno={}", libc::EPERM)));
        assert!(stderr.contains(&format!("errno={}", libc::ENOMEM)));
    }

    #[test]
    fn child_partial_lock_summaries_are_per_model() {
        if std::env::var_os("KAPLUMBA_PARTIAL_LOCK_SUMMARY_CHILD").is_none() {
            return;
        }
        let context = context_or_skip().expect("failure-injection test requires Metal");
        let first = "model.language_model.layers.0.self_attn.q_proj.trellis";
        let second = "model.language_model.layers.0.self_attn.k_proj.trellis";
        let page = page_size();
        let first_payload = vec![0x31_u8; 128];
        let second_payload = vec![0x72_u8; 128];
        let (dir, selection) = (1..65)
            .find_map(|pad| {
                let dir = temp_dir(&format!("partial-lock-{pad}"));
                let selection = selection_for(
                    &dir,
                    vec![
                        ("headpad.bytes", DType::U8, vec![pad], vec![0_u8; pad]),
                        (first, DType::I16, vec![64], first_payload.clone()),
                        ("gap.bytes", DType::U8, vec![2 * page], vec![0_u8; 2 * page]),
                        (second, DType::I16, vec![64], second_payload.clone()),
                    ],
                );
                let first_start = selection.absolute_range(first).unwrap().0;
                let second_start = selection.absolute_range(second).unwrap().0;
                if first_start.is_multiple_of(EXL3_PACKED_ALIGNMENT as u64)
                    && second_start.is_multiple_of(EXL3_PACKED_ALIGNMENT as u64)
                {
                    Some((dir, selection))
                } else {
                    fs::remove_dir_all(dir).unwrap();
                    None
                }
            })
            .expect("fixture must produce two aligned ranges separated by a cold gap");

        let mut calls = 0;
        let mut fail_later = |_, _| {
            calls += 1;
            if calls == 1 {
                Ok(())
            } else {
                Err(io::Error::from_raw_os_error(libc::EPERM))
            }
        };
        let weights = TextWeights::load_internal_with_mlock(
            &dir,
            &selection,
            context,
            4 * 1024 * 1024,
            DEFAULT_SEGMENT_BYTES,
            true,
            true,
            &mut fail_later,
        )
        .expect("partial mlock must warn and continue loading");
        let summary = weights.mlock_summary();
        assert_eq!(summary.eligible_page_bytes, 2 * page as u64);
        assert_eq!(summary.requested_page_bytes, 2 * page as u64);
        assert_eq!(summary.attempted_page_bytes, 2 * page as u64);
        assert_eq!(summary.locked_page_bytes, page as u64);
        assert_eq!(summary.failed_page_bytes, page as u64);
        assert_eq!(summary.evictable_page_bytes, page as u64);
        assert!(
            summary
                .failure
                .as_deref()
                .unwrap()
                .contains(&format!("errno={}", libc::EPERM))
        );
        assert_eq!(weights.pinned_tensors().len(), 1);
        assert_eq!(weights.pinned_bytes(), first_payload.len() as u64);
        assert!(format_mlock_summary(&summary).contains(&summary.model));
        for (name, expected) in [(first, &first_payload), (second, &second_payload)] {
            let (buffer, offset) = weights.zero_copy_tensor_buffer(name).unwrap().unwrap();
            let actual = unsafe {
                std::slice::from_raw_parts(
                    (buffer.contents() as *const u8).add(offset as usize),
                    expected.len(),
                )
            };
            assert_eq!(
                actual,
                expected.as_slice(),
                "partial lock changed tensor data"
            );
        }
        drop(weights);

        let context = context_or_skip().expect("second failure-injection load requires Metal");
        let mut fail_first = |_, _| Err(io::Error::from_raw_os_error(libc::ENOMEM));
        let second_weights = TextWeights::load_internal_with_mlock(
            &dir,
            &selection,
            context,
            4 * 1024 * 1024,
            DEFAULT_SEGMENT_BYTES,
            true,
            true,
            &mut fail_first,
        )
        .expect("first-call lock errors must remain warning-and-continue");
        let second_summary = second_weights.mlock_summary();
        assert_eq!(second_summary.locked_page_bytes, 0);
        assert_eq!(second_summary.failed_page_bytes, 2 * page as u64);
        assert!(
            second_summary
                .failure
                .as_deref()
                .unwrap()
                .contains(&format!("errno={}", libc::ENOMEM))
        );
        assert!(format_mlock_summary(&second_summary).contains(&second_summary.model));
        drop(second_weights);
        fs::remove_dir_all(dir).unwrap();
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
    fn dedicated_loader_keeps_locked_packed_shard_clean_and_zero_copy() {
        let context = context_or_skip().expect("the clean mmap regression requires a Metal device");
        let name = "model.language_model.layers.0.self_attn.q_proj.trellis";
        let payload_len = 2 * 1024 * 1024;
        let payload: Vec<u8> = (0..payload_len).map(|index| (index % 251) as u8).collect();
        let (dir, selection) = (1..65)
            .find_map(|pad| {
                let dir = temp_dir(&format!("clean-mmap-locking-{pad}"));
                let selection = selection_for(
                    &dir,
                    vec![
                        ("pad.bytes", DType::U8, vec![pad], vec![0_u8; pad]),
                        (name, DType::I16, vec![payload_len / 2], payload.clone()),
                    ],
                );
                let start = selection.absolute_range(name).unwrap().0;
                if start.is_multiple_of(EXL3_PACKED_ALIGNMENT as u64) {
                    Some((dir, selection))
                } else {
                    fs::remove_dir_all(dir).unwrap();
                    None
                }
            })
            .expect("fixture must produce an actually lockable aligned tensor");
        let shard_path = dir.join("model.safetensors");
        let weights = TextWeights::load_dedicated(
            &dir,
            &selection,
            context,
            (payload_len + 1024 * 1024) as u64,
        )
        .unwrap();

        let tensor = weights.tensor(name).unwrap();
        assert!(tensor.spans.is_empty(), "packed weights must stay no-copy");
        assert_eq!(
            weights.pinned_tensors().len(),
            1,
            "mlock must remain enabled"
        );
        assert!(
            weights.pinned_bytes() > 0,
            "the fixture must be actually locked"
        );

        let reader = weights.shards.get("model.safetensors").unwrap();
        let (buffer, offset) = weights.zero_copy_tensor_buffer(name).unwrap().unwrap();
        assert_eq!(buffer.contents() as usize, reader.pointer.as_ptr() as usize);
        let mapped = unsafe {
            std::slice::from_raw_parts(
                (buffer.contents() as *const u8).add(offset as usize),
                payload_len,
            )
        };
        assert_eq!(
            mapped,
            payload.as_slice(),
            "no-copy Metal view changed bytes"
        );
        assert_eq!(
            fs::read(shard_path).unwrap()[offset as usize..][..payload_len],
            payload
        );

        // `mlock` on a private mapping faults private COW pages on Darwin. The
        // mapped pages must remain clean after the real loader and Metal view
        // have touched every page.
        let page = page_size();
        let tensor_start = reader.pointer.as_ptr() as usize + offset as usize;
        let start = tensor_start & !(page - 1);
        let end = (tensor_start + payload_len).next_multiple_of(page);
        let mapped_len = end - start;
        assert_eq!(
            weights.pinned_unique_bytes(),
            mapped_len as u64,
            "one actual lock run must count each mapped page once"
        );
        let page_count = mapped_len / page;
        let mut page_state = vec![0_i8; page_count];
        let rc = unsafe {
            libc::mincore(
                start as *const std::ffi::c_void,
                mapped_len,
                page_state.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "mincore failed: {}", io::Error::last_os_error());
        let modified = page_state
            .iter()
            .filter(|state| **state as u8 & libc::MINCORE_MODIFIED as u8 != 0)
            .count();
        assert_eq!(
            modified, 0,
            "locked mmap pages became modified/private ({modified}/{page_count}); expected clean shared file backing"
        );

        drop(buffer);
        drop(weights);
        fs::remove_dir_all(dir).unwrap();

        // A shard smaller than one VM page still maps, locks, and binds. This
        // exercises the final partial file page independently of the multi-page
        // fixture above.
        let context = context_or_skip().expect("the clean mmap regression requires a Metal device");
        let tiny_payload = vec![0x34_u8, 0x12];
        let (dir, selection) = (1..65)
            .find_map(|pad| {
                let dir = temp_dir(&format!("clean-mmap-subpage-{pad}"));
                let selection = selection_for(
                    &dir,
                    vec![
                        ("pad.bytes", DType::U8, vec![pad], vec![0_u8; pad]),
                        (name, DType::I16, vec![1], tiny_payload.clone()),
                    ],
                );
                let start = selection.absolute_range(name).unwrap().0;
                if start.is_multiple_of(EXL3_PACKED_ALIGNMENT as u64) {
                    Some((dir, selection))
                } else {
                    fs::remove_dir_all(dir).unwrap();
                    None
                }
            })
            .expect("sub-page fixture must produce an aligned packed tensor");
        assert!(
            fs::metadata(dir.join("model.safetensors")).unwrap().len() < page_size() as u64,
            "fixture must end inside its first VM page"
        );
        let weights = TextWeights::load_dedicated(&dir, &selection, context, 1024 * 1024).unwrap();
        assert!(
            weights.pinned_bytes() > 0,
            "sub-page packed data must be locked"
        );
        assert_eq!(weights.pinned_unique_bytes(), page_size() as u64);
        let (buffer, offset) = weights.zero_copy_tensor_buffer(name).unwrap().unwrap();
        let actual = unsafe {
            std::slice::from_raw_parts(
                (buffer.contents() as *const u8).add(offset as usize),
                tiny_payload.len(),
            )
        };
        assert_eq!(actual, tiny_payload.as_slice());
        drop(buffer);
        drop(weights);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn shard_open_preserves_missing_and_empty_file_errors() {
        let dir = temp_dir("shard-open-errors");
        let missing = match ShardReader::open(&dir.join("missing.safetensors")) {
            Ok(_) => panic!("missing shard unexpectedly opened"),
            Err(error) => error,
        };
        assert!(matches!(missing, WeightError::Io { .. }));

        let empty_path = dir.join("empty.safetensors");
        fs::write(&empty_path, []).unwrap();
        let empty = match ShardReader::open(&empty_path) {
            Ok(_) => panic!("empty shard unexpectedly opened"),
            Err(error) => error,
        };
        assert!(
            empty.to_string().contains("cannot read an empty shard"),
            "unexpected empty-shard error: {empty}"
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

    /// Phase 2.5 releases exactly the *copied* ranges' page cache, and never the
    /// zero-copy ones -- the fix for "the model loads twice."
    ///
    /// `WeightArena::read_into` `pread`s the tensor straight into the Metal
    /// buffer, and `read_at` populates the unified page cache as a side effect,
    /// so every copied tensor is resident twice until this pass drops it (11.00
    /// GiB at 3 bpw, 14.66 GiB at 4 bpw -- `spec/memory-swap-cliff/
    /// implementation_notes.md` §5.1). That duplicate is what drove the
    /// Zero-copy tensor views retain page-aligned bases and lengths.
    #[test]
    fn tensor_metal_buffer_is_page_aligned() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("shard-align");
        let embed = "model.language_model.embed_tokens.weight";
        let selection = selection_for(
            &dir,
            vec![(embed, DType::BF16, vec![2048], vec![0_u8; 4096])],
        );
        let weights =
            TextWeights::load_with_segment_size(&dir, &selection, context, 1024 * 1024, 64)
                .unwrap();
        let (buf, offset) = weights
            .raw_zero_copy_tensor_buffer(embed)
            .unwrap()
            .expect("embed is zero copy");
        let page = page_size();
        assert_eq!(
            buf.length() as usize % page,
            0,
            "shard metal buffer length must be a multiple of page size ({page})"
        );
        let ptr = buf.contents() as usize;
        assert_eq!(
            ptr % page,
            0,
            "shard metal buffer base address must be page aligned ({page})"
        );
        assert!(offset < buf.length(), "offset must be within the buffer");
        fs::remove_dir_all(dir).unwrap();
    }

    /// Tensors occupying the same rounded page range reuse one Metal view.
    #[test]
    fn no_copy_views_share_identical_page_ranges() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("shard-share");
        let trellis1 = "model.language_model.layers.0.self_attn.q_proj.trellis";
        let trellis2 = "model.language_model.layers.0.self_attn.k_proj.trellis";
        let selection = selection_for(
            &dir,
            vec![
                (trellis1, DType::F16, vec![64], vec![1_u8; 128]),
                (trellis2, DType::F16, vec![64], vec![2_u8; 128]),
            ],
        );
        let weights =
            TextWeights::load_with_segment_size(&dir, &selection, context, 1024 * 1024, 64)
                .unwrap();
        let (buf1, off1) = weights
            .zero_copy_tensor_buffer(trellis1)
            .unwrap()
            .expect("trellis1 zero copy");
        let (buf2, off2) = weights
            .zero_copy_tensor_buffer(trellis2)
            .unwrap()
            .expect("trellis2 zero copy");

        assert_eq!(
            buf1.as_ptr(),
            buf2.as_ptr(),
            "both tensors occupy the same page range and must share that view"
        );
        assert_ne!(off1, off2, "tensors must have distinct offsets in the shard");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn no_copy_views_exclude_unrelated_pages_and_check_boundaries() {
        let device = metal::Device::system_default().expect("Metal device");
        let dir = temp_dir("bounded-views");
        let path = dir.join("mapped.bin");
        let page = page_size();
        let data: Vec<u8> = (0..page * 3 + 7).map(|i| (i / page + 1) as u8).collect();
        fs::write(&path, &data).unwrap();
        let reader = ShardReader::open(&path).unwrap();
        for (offset, len, expected_start, expected_length) in [
            (0, 1, 0, page),
            (page + 4, 128, page, page),
            (page - 1, 2, 0, page * 2),
            (page * 3, 7, page * 3, page),
        ] {
            let (buffer, local) = reader.no_copy_view(&device, offset as u64, len as u64).unwrap();
            assert_eq!(buffer.length() as usize, expected_length, "resource must exclude cold pages");
            assert_eq!(buffer.contents() as usize, reader.pointer.as_ptr() as usize + expected_start);
            assert_eq!(local as usize, offset - expected_start);
            let actual = unsafe { std::slice::from_raw_parts(
                (buffer.contents() as *const u8).add(local as usize), len,
            ) };
            assert_eq!(actual, &data[offset..offset + len]);
        }
        let (first, _) = reader.no_copy_view(&device, page as u64 + 4, 128).unwrap();
        let (same, _) = reader.no_copy_view(&device, page as u64 + 8, 64).unwrap();
        let (other, _) = reader.no_copy_view(&device, (page * 2) as u64, 64).unwrap();
        assert_eq!(first.as_ptr(), same.as_ptr(), "reuse identical rounded ranges");
        assert_ne!(first.as_ptr(), other.as_ptr(), "disjoint pages need independent views");
        assert!(matches!(reader.no_copy_view(&device, u64::MAX, 2), Err(WeightError::ArithmeticOverflow)));
        assert!(matches!(reader.no_copy_view(&device, data.len() as u64, 1), Err(WeightError::Range { .. })));
        assert!(matches!(reader.no_copy_view(&device, 0, 0), Err(WeightError::InvalidSegmentSize(0))));
        fs::remove_dir_all(dir).unwrap();
    }

    /// Which trellis tensors get their FILE range wired, and why.
    ///
    /// `align_packed_view` copies every tensor whose absolute shard offset is
    /// 4-misaligned; the GPU then reads the anonymous copy forever. Pinning the
    /// file range of such a tensor wires a dead second copy -- that is the
    /// "two instances, 20 instead of 10 GiB" bug (measured on `Gxma`:
    /// 311/411 tensors, 8.61 GiB wired-dead beside 8.61 GiB live-anonymous,
    /// 9040 MB of the live side in the compressor). So:
    ///
    ///   * offset % 4 == 0  -> zero-copy view is used -> FILE range must be wired;
    ///   * offset % 4 != 0  -> realigned copy is used -> FILE range must NOT be
    ///     wired (`pin_realigned_copy` wires the copy instead, exercised in
    ///     [`pin_realigned_copy_wires_and_records_the_live_copy`]).
    ///
    /// The fixture brute-forces the payload pad until the trellis' *measured*
    /// absolute offset has the wanted residue -- header length shifts with the
    /// pad, so predicting instead of measuring is how this would go vacuous.
    #[test]
    fn pinning_follows_the_realigned_copy_split() {
        let trellis = "model.language_model.layers.0.self_attn.q_proj.trellis";
        let embed = "model.language_model.embed_tokens.weight";
        for &target in &[0_u64, 1_u64] {
            let Some(context) = context_or_skip() else {
                return;
            };
            let (dir, selection, start) = fixture_with_trellis_residue(trellis, embed, target);
            let weights =
                TextWeights::load_with_segment_size(&dir, &selection, context, 1024 * 1024, 64)
                    .unwrap();
            let pinned: Vec<String> = weights
                .pinned_tensors()
                .into_iter()
                .map(|(name, _bytes)| name)
                .collect();
            let zero_copy_used = start % EXL3_PACKED_ALIGNMENT == 0;
            assert_eq!(
                pinned.iter().any(|name| name == trellis),
                zero_copy_used,
                "trellis at absolute offset {start} (residue {}): pin_trellis_pages {} -- \
                 a misaligned tensor is copied by align_packed_view, so wiring its FILE \
                 range wires a dead second copy (the 20-vs-10 GiB bug); an aligned \
                 tensor is read zero-copy and its file range IS the live instance",
                start % EXL3_PACKED_ALIGNMENT,
                if zero_copy_used {
                    "must pin the file range"
                } else {
                    "must NOT pin the file range"
                },
            );
            assert!(
                !pinned.iter().any(|name| name == embed),
                "embed_tokens must never be wired (R3)"
            );
            fs::remove_dir_all(&dir).unwrap();
        }
    }

    /// Write the fixture with a payload pad until the trellis tensor's absolute
    /// shard offset lands on `target mod EXL3_PACKED_ALIGNMENT`, *measured* from
    /// the written file. Returns `(dir, selection, absolute_offset)`.
    fn fixture_with_trellis_residue(
        trellis: &str,
        embed: &str,
        target: u64,
    ) -> (PathBuf, TextTensorSelection, u64) {
        for pad in 1..65_usize {
            let dir = temp_dir(&format!("pin-split-{target}-{pad}"));
            // `pad.bytes` shifts every later tensor's absolute offset by
            // `pad` bytes; it is loaded as an ordinary small raw tensor and
            // cannot be pinned (not `.trellis`), so it only moves the residue.
            let selection = selection_for(
                &dir,
                vec![
                    ("pad.bytes", DType::U8, vec![pad], vec![0_u8; pad]),
                    (trellis, DType::F16, vec![64], vec![7_u8; 128]),
                    (embed, DType::BF16, vec![64], vec![0_u8; 128]),
                ],
            );
            let bytes = fs::read(dir.join("model.safetensors")).unwrap();
            let header_len =
                u64::from_le_bytes(bytes[0..8].try_into().expect("8-byte length")) as usize;
            let header: serde_json::Value =
                serde_json::from_slice(&bytes[8..8 + header_len]).expect("header json");
            let rel = header[trellis]["data_offsets"][0]
                .as_u64()
                .expect("trellis data_offsets");
            let absolute = (8 + header_len) as u64 + rel;
            if absolute % EXL3_PACKED_ALIGNMENT == target {
                return (dir, selection, absolute);
            }
            fs::remove_dir_all(&dir).unwrap();
        }
        panic!(
            "no pad in 1..65 produced residue {target}; header growth is \
             pathological and the branch coverage claim would be void"
        );
    }

    #[test]
    fn prunes_unused_shards_and_only_retains_zero_copy_shards() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("prune-shards");
        let shard1 = dir.join("model-00001-of-00003.safetensors");
        let shard2 = dir.join("model-00002-of-00003.safetensors");
        let shard3 = dir.join("model-00003-of-00003.safetensors");
        write_shard(
            &shard1,
            vec![(
                "model.language_model.embed_tokens.weight",
                DType::BF16,
                vec![2],
                vec![0, 0, 0, 0],
            )],
        );
        // `.trellis` is zero-copy in *every* production configuration, including
        // the dedicated loader -- see `trellis_stays_zero_copy_even_in_the_
        // dedicated_loader` below. Before that, `dedicated == true` copied it
        // into the arena and this shard was pruned.
        write_shard(
            &shard2,
            vec![(
                "model.language_model.layers.0.self_attn.q_proj.trellis",
                DType::F16,
                vec![1, 1, 2],
                vec![0x34, 0x12, 0x78, 0x56],
            )],
        );
        // A sign side-tensor: an EXL3 component that is *not* offloaded, because
        // the CPU reads it via `read_f16`. Its arena span is non-empty, so this
        // shard is the one that must be pruned.
        write_shard(
            &shard3,
            vec![(
                "model.language_model.layers.0.self_attn.q_proj.suh",
                DType::F16,
                vec![1],
                vec![0x34, 0x12],
            )],
        );
        let index_path = dir.join("model.safetensors.index.json");
        let mut map = serde_json::Map::new();
        map.insert(
            "model.language_model.embed_tokens.weight".to_string(),
            json!("model-00001-of-00003.safetensors"),
        );
        map.insert(
            "model.language_model.layers.0.self_attn.q_proj.trellis".to_string(),
            json!("model-00002-of-00003.safetensors"),
        );
        map.insert(
            "model.language_model.layers.0.self_attn.q_proj.suh".to_string(),
            json!("model-00003-of-00003.safetensors"),
        );
        fs::write(
            &index_path,
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 10},
                "weight_map": map
            }))
            .unwrap(),
        )
        .unwrap();
        let index = read_index(&index_path).unwrap();
        let selection = crate::safetensors::load_text_selection(&dir, index).unwrap();
        let weights =
            TextWeights::load_dedicated(&dir, &selection, context, 1024 * 1024)
                .unwrap();

        if std::env::var_os("KAPLUMBA_SABOTAGE_SHARD_PRUNING").is_some() {
            // Sabotage: assert that all three shards are retained, which must fail
            assert_eq!(
                weights.retained_shards(),
                3,
                "sabotage expects all 3 shards retained"
            );
        } else {
            // Shards 1 and 2 each hold a zero-copy tensor (embed_tokens and
            // `.trellis`). Shard 3 holds only an arena-copied sign side-tensor
            // and must be pruned.
            assert_eq!(
                weights.retained_shards(),
                2,
                "only shards holding zero-copy tensors must be retained"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    /// The `e1b05c1` regression, pinned directly.
    ///
    /// That commit gated the trellis offload on `!dedicated`. Production loads
    /// through `load_dedicated`, which passes `dedicated = true`, so the 12.04
    /// GiB of `.trellis` was copied into dirty anonymous Metal shared buffers.
    /// macOS compressed them under long-context pressure and decode collapsed
    /// from 76 ms to 2,721 ms (13.1 -> 0.37 tok/s) at 16.4k context.
    ///
    /// `dedicated` is true here and the trellis must still be zero-copy.
    #[test]
    fn trellis_stays_zero_copy_even_in_the_dedicated_loader() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dir = temp_dir("trellis-offload");
        let shard = dir.join("model-00001-of-00001.safetensors");
        write_shard(
            &shard,
            vec![(
                "model.language_model.layers.0.self_attn.q_proj.trellis",
                DType::F16,
                vec![1, 1, 2],
                vec![0x34, 0x12, 0x78, 0x56],
            )],
        );
        let index_path = dir.join("model.safetensors.index.json");
        fs::write(
            &index_path,
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 4},
                "weight_map": {
                    "model.language_model.layers.0.self_attn.q_proj.trellis":
                        "model-00001-of-00001.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let index = read_index(&index_path).unwrap();
        let selection = crate::safetensors::load_text_selection(&dir, index).unwrap();
        // `load_dedicated` is what `model.rs` uses, so this is the production shape.
        let weights =
            TextWeights::load_dedicated(&dir, &selection, context, 1024 * 1024)
                .unwrap();

        if std::env::var_os("KAPLUMBA_SABOTAGE_TRELLIS_ZERO_COPY").is_some() {
            // Inverted on purpose: this asserts the *regressed* condition, so it
            // fails on a healthy build. That is the point -- it is the check that
            // the assertion above is not vacuously true.
            let copied = weights
                .tensors()
                .values()
                .any(|t| t.name.ends_with(".trellis") && !t.spans.is_empty());
            assert!(
                copied,
                "sabotage expects the trellis to have been copied into the arena; \
                 on a healthy build it is a zero-copy view, so this must fail"
            );
        } else {
            let trellis = weights
                .tensors()
                .values()
                .find(|t| t.name.ends_with(".trellis"))
                .expect("the fixture trellis must exist");
            assert!(
                trellis.spans.is_empty(),
                "{} has {} arena spans but must stay a zero-copy view so its pages \
                 remain clean and file-backed, which is what stops macOS from \
                 compressing 12.04 GiB of dirty anonymous memory and collapsing \
                 long-context decode",
                trellis.name,
                trellis.spans.len()
            );
            // And the shard that holds it must therefore still be mapped.
            assert_eq!(
                weights.retained_shards(),
                1,
                "the shard holding the zero-copy trellis must still be mapped"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
