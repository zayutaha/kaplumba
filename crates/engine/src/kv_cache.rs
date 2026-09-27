use std::mem::size_of;
use std::ops::Range;

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KvCacheError {
    #[error("cache dimensions must be non-zero")]
    InvalidDimensions,
    #[error("cache shape overflows address space")]
    ShapeOverflow,
    #[error("position {got} is not the next cache position {expected}")]
    InvalidPosition { expected: usize, got: usize },
    #[error("cache position {position} exceeds capacity {capacity}")]
    PositionOutOfRange { position: usize, capacity: usize },
    #[error("key/value length {actual} does not match stream length {expected}")]
    InvalidStreamLength { actual: usize, expected: usize },
    #[error("key and value chunk lengths differ")]
    ChunkLengthMismatch,
    #[error("cache chunk must contain at least one position")]
    EmptyChunk,
}

pub struct FullKvCache {
    max_sequence_len: usize,
    kv_heads: usize,
    head_dim: usize,
    stream_len: usize,
    next_position: usize,
    keys: Vec<u16>,
    values: Vec<u16>,
    positions: Vec<u32>,
}

impl FullKvCache {
    pub fn new(
        max_sequence_len: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KvCacheError> {
        let stream_len = checked_stream_len(kv_heads, head_dim)?;
        if max_sequence_len == 0 {
            return Err(KvCacheError::InvalidDimensions);
        }
        let elements = checked_product(max_sequence_len, stream_len, KvCacheError::ShapeOverflow)?;
        Ok(Self {
            max_sequence_len,
            kv_heads,
            head_dim,
            stream_len,
            next_position: 0,
            keys: vec![0; elements],
            values: vec![0; elements],
            positions: vec![u32::MAX; elements / stream_len],
        })
    }

    pub fn max_sequence_len(&self) -> usize {
        self.max_sequence_len
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn len(&self) -> usize {
        self.next_position
    }

    pub fn is_empty(&self) -> bool {
        self.next_position == 0
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn stream_len(&self) -> usize {
        self.stream_len
    }

    pub fn resident_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * size_of::<u16>()
    }

    pub fn visible_positions(&self) -> Range<usize> {
        0..self.next_position
    }

    pub fn contains(&self, position: usize) -> bool {
        position < self.next_position
    }

    pub fn position_to_slot(&self, position: usize) -> Option<usize> {
        self.contains(position).then_some(position)
    }

    pub fn key_value(&self, position: usize) -> Option<(&[u16], &[u16])> {
        if !self.contains(position) {
            return None;
        }
        let start = position * self.stream_len;
        let end = start + self.stream_len;
        Some((&self.keys[start..end], &self.values[start..end]))
    }

    pub fn key_storage(&self) -> &[u16] {
        &self.keys
    }

    pub fn value_storage(&self) -> &[u16] {
        &self.values
    }

    pub fn active_key_count(&self) -> usize {
        self.next_position
    }

    pub fn key_positions(&self) -> &[u32] {
        &self.positions[..self.next_position]
    }

    pub fn append(
        &mut self,
        position: usize,
        key: &[u16],
        value: &[u16],
    ) -> Result<usize, KvCacheError> {
        if position != self.next_position {
            return Err(KvCacheError::InvalidPosition {
                expected: self.next_position,
                got: position,
            });
        }
        if position >= self.max_sequence_len {
            return Err(KvCacheError::PositionOutOfRange {
                position,
                capacity: self.max_sequence_len,
            });
        }
        validate_stream(key, self.stream_len)?;
        validate_stream(value, self.stream_len)?;
        let start = position * self.stream_len;
        self.keys[start..start + self.stream_len].copy_from_slice(key);
        self.values[start..start + self.stream_len].copy_from_slice(value);
        self.positions[position] =
            u32::try_from(position).map_err(|_| KvCacheError::ShapeOverflow)?;
        self.next_position += 1;
        Ok(position)
    }

    pub fn append_chunk(
        &mut self,
        position: usize,
        keys: &[u16],
        values: &[u16],
    ) -> Result<Range<usize>, KvCacheError> {
        let count = chunk_count(keys, values, self.stream_len)?;
        let end = position
            .checked_add(count)
            .ok_or(KvCacheError::ShapeOverflow)?;
        if end > self.max_sequence_len {
            return Err(KvCacheError::PositionOutOfRange {
                position: end - 1,
                capacity: self.max_sequence_len,
            });
        }
        for index in 0..count {
            let start = index * self.stream_len;
            let key = &keys[start..start + self.stream_len];
            let value = &values[start..start + self.stream_len];
            self.append(position + index, key, value)?;
        }
        Ok(position..end)
    }
}

pub struct SlidingKvCache {
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
    stream_len: usize,
    next_position: usize,
    keys: Vec<u16>,
    values: Vec<u16>,
    positions: Vec<u32>,
}

impl SlidingKvCache {
    pub fn new(kv_heads: usize, head_dim: usize) -> Result<Self, KvCacheError> {
        Self::with_capacity(1024, kv_heads, head_dim)
    }

    pub fn with_capacity(
        capacity: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KvCacheError> {
        let stream_len = checked_stream_len(kv_heads, head_dim)?;
        if capacity == 0 {
            return Err(KvCacheError::InvalidDimensions);
        }
        let elements = checked_product(capacity, stream_len, KvCacheError::ShapeOverflow)?;
        Ok(Self {
            capacity,
            kv_heads,
            head_dim,
            stream_len,
            next_position: 0,
            keys: vec![0; elements],
            values: vec![0; elements],
            positions: vec![u32::MAX; elements / stream_len],
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn len(&self) -> usize {
        self.next_position
    }

    pub fn is_empty(&self) -> bool {
        self.next_position == 0
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn stream_len(&self) -> usize {
        self.stream_len
    }

    pub fn resident_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * size_of::<u16>()
    }

    pub fn visible_positions(&self) -> Range<usize> {
        self.oldest_position()..self.next_position
    }

    pub fn oldest_position(&self) -> usize {
        self.next_position.saturating_sub(self.capacity)
    }

    pub fn contains(&self, position: usize) -> bool {
        position < self.next_position && position >= self.oldest_position()
    }

    pub fn position_to_slot(&self, position: usize) -> Option<usize> {
        self.contains(position).then_some(position % self.capacity)
    }

    pub fn key_value(&self, position: usize) -> Option<(&[u16], &[u16])> {
        let slot = self.position_to_slot(position)?;
        let start = slot * self.stream_len;
        let end = start + self.stream_len;
        Some((&self.keys[start..end], &self.values[start..end]))
    }

    pub fn key_storage(&self) -> &[u16] {
        &self.keys
    }

    pub fn value_storage(&self) -> &[u16] {
        &self.values
    }

    pub fn key_positions(&self) -> &[u32] {
        &self.positions
    }

    pub fn active_key_count(&self) -> usize {
        self.capacity
    }

    pub fn append(
        &mut self,
        position: usize,
        key: &[u16],
        value: &[u16],
    ) -> Result<usize, KvCacheError> {
        if position != self.next_position {
            return Err(KvCacheError::InvalidPosition {
                expected: self.next_position,
                got: position,
            });
        }
        validate_stream(key, self.stream_len)?;
        validate_stream(value, self.stream_len)?;
        let slot = position % self.capacity;
        let start = slot * self.stream_len;
        self.keys[start..start + self.stream_len].copy_from_slice(key);
        self.values[start..start + self.stream_len].copy_from_slice(value);
        self.positions[slot] = u32::try_from(position).map_err(|_| KvCacheError::ShapeOverflow)?;
        self.next_position = self
            .next_position
            .checked_add(1)
            .ok_or(KvCacheError::ShapeOverflow)?;
        Ok(slot)
    }

    pub fn append_chunk(
        &mut self,
        position: usize,
        keys: &[u16],
        values: &[u16],
    ) -> Result<Range<usize>, KvCacheError> {
        let count = chunk_count(keys, values, self.stream_len)?;
        let end = position
            .checked_add(count)
            .ok_or(KvCacheError::ShapeOverflow)?;
        for index in 0..count {
            let start = index * self.stream_len;
            let key = &keys[start..start + self.stream_len];
            let value = &values[start..start + self.stream_len];
            self.append(position + index, key, value)?;
        }
        Ok(position..end)
    }
}

pub enum KvCache {
    Full(FullKvCache),
    Sliding(SlidingKvCache),
}

impl KvCache {
    pub fn full(
        max_sequence_len: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KvCacheError> {
        Ok(Self::Full(FullKvCache::new(
            max_sequence_len,
            kv_heads,
            head_dim,
        )?))
    }

    pub fn sliding(kv_heads: usize, head_dim: usize) -> Result<Self, KvCacheError> {
        Ok(Self::Sliding(SlidingKvCache::new(kv_heads, head_dim)?))
    }

    pub fn append(
        &mut self,
        position: usize,
        key: &[u16],
        value: &[u16],
    ) -> Result<usize, KvCacheError> {
        match self {
            Self::Full(cache) => cache.append(position, key, value),
            Self::Sliding(cache) => cache.append(position, key, value),
        }
    }

    pub fn append_chunk(
        &mut self,
        position: usize,
        keys: &[u16],
        values: &[u16],
    ) -> Result<Range<usize>, KvCacheError> {
        match self {
            Self::Full(cache) => cache.append_chunk(position, keys, values),
            Self::Sliding(cache) => cache.append_chunk(position, keys, values),
        }
    }

    pub fn key_value(&self, position: usize) -> Option<(&[u16], &[u16])> {
        match self {
            Self::Full(cache) => cache.key_value(position),
            Self::Sliding(cache) => cache.key_value(position),
        }
    }

    pub fn position_to_slot(&self, position: usize) -> Option<usize> {
        match self {
            Self::Full(cache) => cache.position_to_slot(position),
            Self::Sliding(cache) => cache.position_to_slot(position),
        }
    }

    pub fn visible_positions(&self) -> Range<usize> {
        match self {
            Self::Full(cache) => cache.visible_positions(),
            Self::Sliding(cache) => cache.visible_positions(),
        }
    }

    pub fn contains(&self, position: usize) -> bool {
        match self {
            Self::Full(cache) => cache.contains(position),
            Self::Sliding(cache) => cache.contains(position),
        }
    }

    pub fn next_position(&self) -> usize {
        match self {
            Self::Full(cache) => cache.next_position(),
            Self::Sliding(cache) => cache.next_position(),
        }
    }

    pub fn len(&self) -> usize {
        self.next_position()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Full(cache) => cache.resident_bytes(),
            Self::Sliding(cache) => cache.resident_bytes(),
        }
    }
}

fn checked_stream_len(kv_heads: usize, head_dim: usize) -> Result<usize, KvCacheError> {
    if kv_heads == 0 || head_dim == 0 {
        return Err(KvCacheError::InvalidDimensions);
    }
    checked_product(kv_heads, head_dim, KvCacheError::ShapeOverflow)
}

fn checked_product(left: usize, right: usize, error: KvCacheError) -> Result<usize, KvCacheError> {
    left.checked_mul(right).ok_or(error)
}

fn validate_stream(values: &[u16], expected: usize) -> Result<(), KvCacheError> {
    if values.len() != expected {
        return Err(KvCacheError::InvalidStreamLength {
            actual: values.len(),
            expected,
        });
    }
    Ok(())
}

fn chunk_count(keys: &[u16], values: &[u16], stream_len: usize) -> Result<usize, KvCacheError> {
    if keys.is_empty() {
        return Err(KvCacheError::EmptyChunk);
    }
    if keys.len() != values.len() {
        return Err(KvCacheError::ChunkLengthMismatch);
    }
    if !keys.len().is_multiple_of(stream_len) {
        return Err(KvCacheError::InvalidStreamLength {
            actual: keys.len(),
            expected: stream_len,
        });
    }
    Ok(keys.len() / stream_len)
}
