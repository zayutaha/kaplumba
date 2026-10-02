//! GPU-resident key/value cache.
//!
//! The host cache in [`crate::kv_cache`] keeps keys and values in `Vec<u16>` and
//! is read back for every attention call, which forces a host round trip per
//! layer. This cache keeps the same storage in Metal buffers so a decoder layer
//! can append and attend entirely on device.
//!
//! Layout matches the host cache exactly so the two can be compared directly:
//! keys and values are `[capacity][stream_len]` in FP16, positions are
//! `[capacity]` in `u32` (empty slots hold `u32::MAX`), and sliding layers ring
//! through the buffer.

use metal::{Buffer, Device, MTLResourceOptions};
use thiserror::Error;

use crate::generate::DEFAULT_CONTEXT_LIMIT;
use crate::kv_cache::KvCacheError;

pub const EMPTY_POSITION: u32 = u32::MAX;

/// Hard ceiling on the capacity of a cache grown by [`GpuKvCache::ensure_capacity`],
/// in positions.
///
/// **Why `DEFAULT_CONTEXT_LIMIT` and not `max_position_embeddings`.**
/// `DecoderLayer::max_sequence_len` is set to the checkpoint's
/// `max_position_embeddings` (`model.rs:3396`) and is handed straight to
/// `ensure_capacity` as `max_capacity` (`model.rs:1663` and `:1783`). That field
/// is **262144** for this checkpoint, while the policy limit the engine
/// actually enforces before every forward is 32768 (`generate.rs:35`,
/// `GenerationConfig::default` at `generate.rs:79`, and the CLI's own boundary
/// test at `crates/cli/src/args.rs:376-388`). `model.rs:3381-3395` states the
/// intent explicitly --
///
/// > The policy bound on how many positions that may be is
/// > `DEFAULT_CONTEXT_LIMIT`, checked before every forward by
/// > `fits_in_context` / `generate`, so the two cannot disagree about what is
/// > reachable.
///
/// They can and do disagree. Every caller that reaches
/// `TextModel::forward_buffer` directly -- `tests/real_forward_buffer.rs`,
/// `tests/memory_proof.rs`, most of `crates/engine/examples/` -- never runs
/// `fits_in_context`, so the bound is the only thing standing between a run and
/// the full-attention KV growing to
/// `10 layers x 262144 x (stream_len 2048 x 2 B x 2 + 4 B) = 22_324_387_840 B`
/// = **20.79 GiB** on a 24 GiB box. `resident_limit_bytes()`
/// (`metal.rs:4`, 18 GB) does not help: it bounds the *weight arena*, not the
/// KV, and 18 GB is 16.76 GiB of the box anyway.
///
/// So the policy limit is enforced where the allocation happens, not only in the
/// caller. That makes the two bounds structurally unable to disagree rather than
/// merely documented as agreeing, and it is the same constant, so raising the
/// policy raises the bound.
///
/// **Cost: nothing today, by construction.** No reachable path exceeds 32768
/// positions -- `GenerationConfig::default` uses `DEFAULT_CONTEXT_LIMIT` and
/// there is no CLI flag to raise it -- so this removes ~17.5 GiB of latent
/// allocation exposure and changes no existing run. Raising the policy limit
/// means raising this constant in the same commit.
pub const MAX_POLICY_CAPACITY: usize = DEFAULT_CONTEXT_LIMIT;

/// `PartialEq` so a test can assert the whole error, not just that one
/// variant matched: the *bound* inside `PositionOutOfRange` is the thing the
/// growth policy actually promises, and `matches!` cannot see it.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum GpuCacheError {
    #[error("GPU cache dimensions must be non-zero")]
    InvalidDimensions,
    #[error("GPU cache shape overflows address space")]
    ShapeOverflow,
    #[error("position {got} is not the next cache position {expected}")]
    #[allow(dead_code)]
    InvalidPosition { expected: usize, got: usize },
    #[error("position {position} exceeds capacity {capacity}")]
    PositionOutOfRange { position: usize, capacity: usize },
    #[error(transparent)]
    Cache(#[from] KvCacheError),
}

pub struct GpuKvCache {
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
    stream_len: usize,
    next_position: usize,
    keys: Buffer,
    values: Buffer,
    positions: Buffer,
}

impl GpuKvCache {
    pub fn new(
        device: &Device,
        capacity: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, GpuCacheError> {
        if capacity == 0 || kv_heads == 0 || head_dim == 0 {
            return Err(GpuCacheError::InvalidDimensions);
        }
        let stream_len = kv_heads
            .checked_mul(head_dim)
            .ok_or(GpuCacheError::ShapeOverflow)?;
        let elements = capacity
            .checked_mul(stream_len)
            .ok_or(GpuCacheError::ShapeOverflow)?;
        let key_bytes = (elements * 2) as u64;
        // Tracked (default hazard tracking): KV keys/values are written by the
        // qkv/kv-append dispatch and read by attention on the next dispatch in
        // the same command buffer. Untracked mode skips Metal's automatic
        // inter-dispatch barrier, and no explicit `memoryBarrier` replaces it,
        // so the two dispatches raced and attention read torn keys -> NaN.
        // Reverted from eed081e (untracked) for correctness.
        let options = MTLResourceOptions::StorageModeShared;
        let keys = device.new_buffer(key_bytes, options);
        let values = device.new_buffer(key_bytes, options);
        let positions =
            device.new_buffer((capacity * 4) as u64, options);
        // Empty slots must read as "no position" (all bits set) so the attention
        // kernel's window check discards them; Metal does not promise zeroed
        // contents for a freshly created buffer.
        unsafe {
            let pointer = positions.contents() as *mut u32;
            std::ptr::write_bytes(pointer, 0xFF, capacity);
        }
        Ok(Self {
            capacity,
            kv_heads,
            head_dim,
            stream_len,
            next_position: 0,
            keys,
            values,
            positions,
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

    pub fn stream_len(&self) -> usize {
        self.stream_len
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn len(&self) -> usize {
        self.next_position
    }

    pub fn is_empty(&self) -> bool {
        self.next_position == 0
    }

    pub fn resident_bytes(&self) -> usize {
        (self.capacity * self.stream_len * 2 * 2) + (self.capacity * 4)
    }

    /// Number of key slots the attention kernel should scan.
    ///
    /// Full caches only ever hold valid entries in the low slots, so the scan can
    /// stop at the write position.
    ///
    /// Sliding caches ring, so returning `capacity` unconditionally made every
    /// sliding layer scan (and accumulate) all 1024 slots at every step, even at
    /// position 10. `min` is enough because the two regimes collapse on their own:
    ///
    /// * Before the ring wraps (`next_position < capacity`) the only live slots
    ///   are `[0, next_position)`. Skipping the tail is safe because unwritten
    ///   slots are still `EMPTY_POSITION` (zeroed in [`Self::new`], refilled on
    ///   [`Self::ensure_capacity`], cleared on [`Self::reset`]), so the kernel's
    ///   window check would have masked them to weight 0 anyway.
    /// * Once the ring wraps (`next_position >= capacity`) `min` yields
    ///   `capacity` and every slot is live again, so no separate wrapped path is
    ///   needed -- the same expression is correct in both regimes.
    pub fn key_count(&self, full_attention: bool) -> usize {
        if full_attention {
            self.next_position
        } else {
            self.next_position.min(self.capacity)
        }
    }

    /// Slot that absolute `position` maps to.
    pub fn slot_for(&self, position: usize, full_attention: bool) -> usize {
        if full_attention {
            position
        } else {
            position % self.capacity
        }
    }

    /// Grow the cache so `needed` positions fit, preserving existing contents.
    ///
    /// Full-attention caches are the memory bulk of the KV state (10 layers at
    /// 8192 positions versus 50 sliding layers at 1024), but a short conversation
    /// only ever touches the first few hundred. Allocating for the worst case up
    /// front cost ~1.4 GiB of resident memory for nothing, so these start small
    /// and double here. Sliding caches are never grown: their capacity is fixed
    /// at the window size and they ring.
    ///
    /// `max_capacity` is the caller's *stated* bound and is clamped to
    /// [`MAX_POLICY_CAPACITY`] before it is trusted, so no caller -- however it
    /// obtained the number -- can grow the cache past the engine's policy
    /// context limit. The clamp is what makes the bound load-bearing: the
    /// caller-supplied value is `max_position_embeddings` = 262144, which at
    /// the model's own full-attention shape is a 20.79 GiB allocation, and a
    /// `forward_buffer` caller never passes through `fits_in_context`, the only
    /// place that limit was previously enforced. See [`MAX_POLICY_CAPACITY`].
    pub fn ensure_capacity(
        &mut self,
        device: &Device,
        needed: usize,
        max_capacity: usize,
    ) -> Result<(), GpuCacheError> {
        // Clamped before the bounds check, so the error names the bound that
        // actually applies rather than the one the caller asked for.
        let max_capacity = max_capacity.min(MAX_POLICY_CAPACITY);
        if needed <= self.capacity {
            return Ok(());
        }
        if needed > max_capacity {
            return Err(GpuCacheError::PositionOutOfRange {
                position: needed - 1,
                capacity: max_capacity,
            });
        }
        // Double until it fits, but never past the model's context limit.
        let mut target = self.capacity.max(1);
        while target < needed {
            target = target.saturating_mul(2);
        }
        let target = target.min(max_capacity);

        let new_bytes = (target * self.stream_len * 2 * 2) as u64;
        // Tracked: same reason as `new` above — attention and kv-append race
        // without Metal's hazard barrier. Reverted from eed081e.
        let options = MTLResourceOptions::StorageModeShared;
        let new_keys = device.new_buffer(new_bytes, options);
        let new_values = device.new_buffer(new_bytes, options);
        let new_positions =
            device.new_buffer((target * 4) as u64, options);

        // Carry the live prefix over. Full caches only hold valid entries in the
        // low slots, so the copy is a straight prefix copy; sliding caches never
        // reach this path.
        let live = self.next_position.min(self.capacity);
        if live > 0 {
            let element_bytes = self.stream_len * 2;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.keys.contents() as *const u8,
                    new_keys.contents() as *mut u8,
                    live * element_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    self.values.contents() as *const u8,
                    new_values.contents() as *mut u8,
                    live * element_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    self.positions.contents() as *const u8,
                    new_positions.contents() as *mut u8,
                    live * 4,
                );
            }
        }
        // Slots beyond the copied prefix must read as empty.
        unsafe {
            let pointer = (new_positions.contents() as *mut u8).add(live * 4);
            std::ptr::write_bytes(pointer, 0xFF, target - live);
        }

        self.keys = new_keys;
        self.values = new_values;
        self.positions = new_positions;
        self.capacity = target;
        Ok(())
    }

    /// Validate that a chunk may be appended at `position` and return its slot.
    pub fn reserve(
        &self,
        position: usize,
        tokens: usize,
        full_attention: bool,
    ) -> Result<usize, GpuCacheError> {
        if position != self.next_position {
            return Err(GpuCacheError::InvalidPosition {
                expected: self.next_position,
                got: position,
            });
        }
        if full_attention && position + tokens > self.capacity {
            return Err(GpuCacheError::PositionOutOfRange {
                position,
                capacity: self.capacity,
            });
        }
        Ok(self.slot_for(position, full_attention))
    }

    pub fn advance(&mut self, tokens: usize) {
        self.next_position += tokens;
    }

    pub fn retract(&mut self, tokens: usize) {
        self.next_position = self.next_position.saturating_sub(tokens);
    }

    pub fn reset(&mut self) {
        self.next_position = 0;
        unsafe {
            let pointer = self.positions.contents() as *mut u32;
            std::ptr::write_bytes(pointer, 0xFF, self.capacity);
        }
    }

    pub fn keys(&self) -> &Buffer {
        &self.keys
    }

    pub fn values(&self) -> &Buffer {
        &self.values
    }

    pub fn positions(&self) -> &Buffer {
        &self.positions
    }

    /// Copy the device cache back to the host for verification and debugging.
    pub fn read_to_host(&self) -> (Vec<u16>, Vec<u16>, Vec<u32>) {
        let elements = self.capacity * self.stream_len;
        let mut keys = vec![0_u16; elements];
        let mut values = vec![0_u16; elements];
        let mut positions = vec![EMPTY_POSITION; self.capacity];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.keys.contents() as *const u16,
                keys.as_mut_ptr(),
                elements,
            );
            std::ptr::copy_nonoverlapping(
                self.values.contents() as *const u16,
                values.as_mut_ptr(),
                elements,
            );
            std::ptr::copy_nonoverlapping(
                self.positions.contents() as *const u32,
                positions.as_mut_ptr(),
                self.capacity,
            );
        }
        (keys, values, positions)
    }
}
