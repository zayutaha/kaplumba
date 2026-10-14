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

use kaplumba_kernels::encoder::MetalEncoder;

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
    #[error("sliding cache slot {slot} contains position {got}, expected {expected}")]
    SlidingPositionMismatch {
        slot: usize,
        expected: u32,
        got: u32,
    },
    #[error(transparent)]
    Cache(#[from] KvCacheError),
    #[error("Metal queue failed before sliding-cache growth: {0}")]
    Queue(String),
}

/// Finish queued work before host migration reads cache buffers or replaces
/// buffers that pending Metal commands may still reference.
pub fn wait_for_sliding_growth(encoder: &mut MetalEncoder<'_>) -> Result<(), GpuCacheError> {
    encoder
        .flush_and_wait_queue()
        .map_err(|error| GpuCacheError::Queue(error.to_string()))?;
    Ok(())
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
    /// front cost ~1.4 GiB of resident memory for nothing, so full caches start
    /// small and step up here. Sliding caches grow through
    /// [`Self::ensure_sliding_capacity`].
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
        let target = growth_target(self.capacity, needed, max_capacity)?;
        if target == self.capacity {
            return Ok(());
        }

        // Exact 1x sizing: one FP16 element is 2 bytes, so a buffer holds
        // `target * stream_len * 2` bytes. This was `* 2 * 2`, which allocated
        // **twice** what `new` allocates for the same shape (and twice what the
        // prefix copy below writes): at `stream_len = 2048` a 16384-slot ladder
        // step asked for 134,217,728 B per buffer where 67,108,864 B is the
        // exact need, over-allocating the full-attention KV by 2x at every rung
        // (`spec/llama-mlock-zero-swap` R5).
        let new_bytes = (target * self.stream_len * 2) as u64;
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
        debug_assert!(
            target > self.capacity,
            "the ladder must grow: target {target} vs capacity {}",
            self.capacity,
        );
        let live = self.next_position.min(self.capacity);
        // `target > self.capacity >= live` holds here: `growth_target` only
        // returns something different when `needed > self.capacity`, and it
        // returns `>= needed` (see its property 1). Both the copy length and the
        // `target - live` length below depend on it -- in release a `target <
        // live` would wrap `target - live` into a ~2^64 memset past the end of
        // the buffer rather than panicking, so it is asserted rather than left
        // to inspection.
        debug_assert!(live <= target, "live {live} exceeds target {target}");
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
        // Slots beyond the copied prefix must read as empty. `positions` is
        // `u32` per slot and the pointer is `*mut u8`, so the byte count is
        // `(target - live) * 4`; writing `target - live` bytes covered only a
        // quarter of the tail, leaving three of every four bytes of each new
        // slot undefined (`spec/llama-mlock-zero-swap` R6).
        unsafe {
            let pointer = (new_positions.contents() as *mut u8).add(live * 4);
            std::ptr::write_bytes(pointer, 0xFF, (target - live) * 4);
        }

        self.keys = new_keys;
        self.values = new_values;
        self.positions = new_positions;
        self.capacity = target;
        Ok(())
    }

    /// Ensure a sliding cache can retain its live window while appending a
    /// whole chunk. Sliding caches start small and grow only when a prompt or
    /// decode step needs more slots than have been allocated so far.
    ///
    /// The caller must finish all GPU work that can reference this cache before
    /// calling this method. Growth migrates the last `window` absolute positions
    /// into the new ring layout, rather than copying a prefix as full caches do.
    pub fn ensure_sliding_capacity(
        &mut self,
        device: &Device,
        window: usize,
        tokens: usize,
        max_capacity: usize,
    ) -> Result<(), GpuCacheError> {
        let needed = sliding_capacity_needed(self.next_position, tokens, window)?;
        let max_capacity = max_capacity.min(MAX_POLICY_CAPACITY);
        let target = growth_target(self.capacity, needed, max_capacity)?;
        if target == self.capacity {
            return Ok(());
        }

        let key_bytes = target
            .checked_mul(self.stream_len)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(GpuCacheError::ShapeOverflow)?;
        let position_bytes = target
            .checked_mul(4)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(GpuCacheError::ShapeOverflow)?;
        let options = MTLResourceOptions::StorageModeShared;
        let new_keys = device.new_buffer(key_bytes, options);
        let new_values = device.new_buffer(key_bytes, options);
        let new_positions = device.new_buffer(position_bytes, options);
        unsafe {
            std::ptr::write_bytes(new_positions.contents() as *mut u8, 0xFF, target * 4);
        }

        let live_start = self.next_position.saturating_sub(window);
        let element_bytes = self
            .stream_len
            .checked_mul(2)
            .ok_or(GpuCacheError::ShapeOverflow)?;
        for absolute in live_start..self.next_position {
            let expected = u32::try_from(absolute).map_err(|_| GpuCacheError::ShapeOverflow)?;
            let old_slot = absolute % self.capacity;
            let new_slot = absolute % target;
            let old_position =
                unsafe { *((self.positions.contents() as *const u32).add(old_slot)) };
            if old_position != expected {
                return Err(GpuCacheError::SlidingPositionMismatch {
                    slot: old_slot,
                    expected,
                    got: old_position,
                });
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (self.keys.contents() as *const u8).add(old_slot * element_bytes),
                    (new_keys.contents() as *mut u8).add(new_slot * element_bytes),
                    element_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    (self.values.contents() as *const u8).add(old_slot * element_bytes),
                    (new_values.contents() as *mut u8).add(new_slot * element_bytes),
                    element_bytes,
                );
                *((new_positions.contents() as *mut u32).add(new_slot)) = expected;
            }
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

/// Minimum ring capacity that lets a chunk append without overwriting any key
/// still visible to the earliest query in that chunk.
pub fn sliding_capacity_needed(
    position: usize,
    tokens: usize,
    window: usize,
) -> Result<usize, GpuCacheError> {
    if tokens == 0 || window == 0 {
        return Err(GpuCacheError::InvalidDimensions);
    }
    position
        .min(window)
        .checked_add(tokens)
        .ok_or(GpuCacheError::ShapeOverflow)
}

/// Capacity growth factor: 1.25x, as `GROWTH_NUM / GROWTH_DEN`.
///
/// 1.25 and not 2 because the KV ladder's over-allocation *is* memory: at
/// `stream_len = 2048` (4 global KV heads x 512) a full-attention layer holds
/// `8192 + 4 = 8196` B per position, so doubling from 513 positions allocated
/// 1024 slots to hold 513 -- **99.6 % waste**, and
/// `needed 16385 -> capacity 32768` wasted 1.25 GiB of a 2.50 GiB ladder across
/// the 10 full layers. At exactly 16384 the ladder lands on the power of two
/// and wastes nothing, which is why the doubling never showed up in a 16k
/// measurement; it appears the first time a run reaches 16385.
///
/// The factor is written as a ratio rather than a literal because the waste
/// bound, the step count and the loop's termination all fall out of it:
/// `GROWTH_NUM / GROWTH_DEN > 1` guarantees every step makes progress.
const GROWTH_NUM: usize = 5;
const GROWTH_DEN: usize = 4;

/// Capacity the ladder should allocate next, or `current` if none is needed.
///
/// Pure: no device, no allocation, so the growth policy is testable on its own
/// (see the `#[cfg(test)]` tests below).
///
/// Three properties, and the tests pin all three:
///
/// 1. **It fits.** `Ok(target) >= needed` whenever `Ok`. The final
///    `.min(max_capacity)` can only ever *lower* the target onto
///    `max_capacity`, and `needed > max_capacity` was already rejected, so the
///    clamp cannot produce a capacity smaller than `needed` -- which would
///    silently shrink the cache below the live data and make `reserve` reject a
///    legal position afterwards.
/// 2. **Waste is bounded by 1/4.** The loop exits on the first `target >=
///    needed`, and the rung that got it there started from `target <= needed - 1`
///    and is `ceil(5t/4) <= 5t/4 + 3/4`, so
///    `target <= 5(needed - 1)/4 + 3/4 = 1.25*needed - 0.5`, i.e.
///    `(target - needed) * GROWTH_DEN < needed` -- strictly under a quarter.
///    Clamping onto the bound only lowers the target, so the bound survives it.
/// 3. **The step count is `O(log needed)`**, because the ladder is geometric:
///    `ceil(ln(32768 / 512) / ln 1.25) = 19` rungs against doubling's
///    `ceil(ln(32768 / 512) / ln 2) = 6`. That is the price of property 2, and
///    it is paid in `ensure_capacity` calls and in copied bytes: each call
///    memcpys the live prefix once, so total copied bytes is
///    `sum(1.25^k) ~= 4.3x` the final size against `2x` for doubling.
///    **Measured** with `cargo run --example kv_growth` at the model's real
///    full-attention shape (`stream_len` 2048, 8196 B per position per layer):
///    13 rungs to 8192 positions, 2752.6 MiB copied across the 10 full layers
///    over the whole run, worst single rung 58.3 MiB/layer (6.2 ms wall, 0.06 %
///    of an 80 ms step), and a median no-op `ensure_capacity` of 0.00 us
///    (max 0.17 us) so the extra rungs cost nothing when they are not taken. So
///    the peak rung is *larger* than doubling's last one (7459 slots copied
///    against 4096 -- 1.82x) and the total is 4.3x doubling's; at 344 KiB per
///    token amortised across the 10 layers neither is visible against a ~70 ms
///    decode step. That is the whole trade, measured rather than asserted.
///
/// Deliberately does **not** clamp to [`MAX_POLICY_CAPACITY`]: that is
/// [`GpuKvCache::ensure_capacity`]'s job, at the point where the caller's bound
/// is interpreted. This function honours whatever bound it is given, which is
/// also what lets the ladder be tested far above the policy limit.
fn growth_target(
    current: usize,
    needed: usize,
    max_capacity: usize,
) -> Result<usize, GpuCacheError> {
    if needed <= current {
        return Ok(current);
    }
    if needed > max_capacity {
        return Err(GpuCacheError::PositionOutOfRange {
            position: needed - 1,
            capacity: max_capacity,
        });
    }
    let mut target = current.max(1);
    while target < needed {
        let next = grow_once(target);
        if next <= target {
            // Unreachable while GROWTH_NUM > GROWTH_DEN; a hang in here would be
            // the worst failure this function could have, so it is checked rather
            // than assumed.
            return Err(GpuCacheError::ShapeOverflow);
        }
        target = next;
    }
    // The step can overshoot `max_capacity` by at most GROWTH_NUM/GROWTH_DEN - 1,
    // and clamping down onto the bound is safe: property 1 above.
    let target = target.min(max_capacity);
    debug_assert!(target >= needed, "clamped ladder target {target} < needed {needed}");
    Ok(target)
}

/// One rung of the ladder: `ceil(current * GROWTH_NUM / GROWTH_DEN)`, saturating.
///
/// Integer arithmetic only, and it never rounds *down* into the current value --
/// the `+ (DEN - 1)` ceiling idiom so 1 -> 2, 4 -> 5 and 8 -> 10, and saturating
/// so a `usize::MAX` capacity cannot wrap into a smaller number and stall
/// [`growth_target`] in its loop.
///
/// **Not `usize::div_ceil`.** It is defined as `(self + rhs - 1) / rhs`, which
/// *panics* on overflow rather than saturating, so `div_ceil(usize::MAX)` traps
/// in a debug build -- caught by `grow_once_rounds_up_and_never_stalls` while
/// this was being written. The `saturating_add` below is what makes the
/// `usize::MAX` rung safe; the exact value of a saturated `step` is irrelevant
/// because the final `saturating_add` caps the result at `usize::MAX` either
/// way, and only `step >= 1` matters for the ladder making progress.
///
/// Written as `current + ceil(current / GROWTH_DEN)` rather than
/// `ceil(current * GROWTH_NUM / GROWTH_DEN)` because the product overflows long
/// before the sum does, and because the identity only holds while
/// `GROWTH_NUM == GROWTH_DEN + 1` -- which is asserted, at compile time here and
/// against the ratio itself in the `debug_assert` below, so the two constants
/// cannot drift apart silently.
fn grow_once(current: usize) -> usize {
    const _: () = assert!(
        GROWTH_NUM == GROWTH_DEN + 1,
        "grow_once implements GROWTH_NUM == GROWTH_DEN + 1"
    );
    let step = current
        .saturating_add(GROWTH_DEN - 1)
        .checked_div(GROWTH_DEN)
        .unwrap_or(0)
        .max(1);
    // Cross-check the closed form against the ratio it claims to implement, for
    // every capacity the ladder can hold in practice. The `1 << 20` cutoff skips
    // the near-`usize::MAX` capacities, where `current * GROWTH_NUM` overflows
    // and where a saturated step is deliberately not the exact ceiling.
    if let Some(ceiling) = (current < (1 << 20)).then(|| {
        let product = current * GROWTH_NUM;
        product / GROWTH_DEN + usize::from(!product.is_multiple_of(GROWTH_DEN))
    }) {
        debug_assert!(
            current + step == ceiling,
            "grow_once({current}) = {} is not ceil({current} * {GROWTH_NUM} / {GROWTH_DEN}) = {ceiling}",
            current + step,
        );
    }
    current.saturating_add(step)
}

#[cfg(test)]
mod tests {
    use super::*;

    const INITIAL_FULL_CACHE: usize = 512;

    /// The whole point of F3: the ladder over-allocates by strictly less than a
    /// quarter, for every position a run can reach, and still lands exactly on
    /// the bound rather than short of it.
    #[test]
    fn the_ladder_fits_and_wastes_under_a_quarter() {
        let mut capacity = INITIAL_FULL_CACHE;
        let mut steps = 0;
        let mut first_step_at = None;

        for needed in INITIAL_FULL_CACHE + 1..=MAX_POLICY_CAPACITY {
            let target = growth_target(capacity, needed, MAX_POLICY_CAPACITY)
                .unwrap_or_else(|error| panic!("growth_target({capacity}, {needed}): {error}"));
            assert!(target >= needed, "capacity {target} does not fit {needed}");
            assert!(
                (target - needed) * GROWTH_DEN < needed,
                "capacity {target} over-allocates for {needed} by {} B, \
                 which is not under a quarter of {needed} B",
                target - needed,
            );
            if target != capacity {
                capacity = target;
                steps += 1;
                first_step_at.get_or_insert(needed);
            }
        }
        assert_eq!(
            capacity,
            MAX_POLICY_CAPACITY,
            "the ladder must land exactly on the bound"
        );
        assert_eq!(
            first_step_at,
            Some(INITIAL_FULL_CACHE + 1),
            "the first rung must be taken at the first position that does not fit"
        );
        // O(log n), with the constant the 1.25 factor implies. ln(32768/512) is
        // 4.159 and /ln(1.25) is 18.64, so 19 rungs; the band leaves room for a
        // rung that lands on a round number without letting a doubling's 6 --
        // or a linear walk's 32255 -- through.
        assert!(
            (10..=20).contains(&steps),
            "{steps} rungs to {MAX_POLICY_CAPACITY}; expected 19 for the 1.25 factor"
        );
    }

    /// The same two properties far above the policy limit, so the ladder's
    /// asymptotic behaviour is pinned and not just its behaviour on the one
    /// range the engine can reach. Doubling fails this at the first step.
    #[test]
    fn the_ladder_stays_within_a_quarter_over_a_wide_range() {
        let mut capacity = INITIAL_FULL_CACHE;
        for needed in (INITIAL_FULL_CACHE + 1..=1_048_576).step_by(7) {
            let target = growth_target(capacity, needed, 1_048_576).unwrap();
            assert!(target >= needed, "capacity {target} does not fit {needed}");
            assert!(
                (target - needed) * GROWTH_DEN < needed,
                "capacity {target} over-allocates for {needed} by {} B",
                target - needed,
            );
            capacity = target;
        }
    }

    /// `grow_once` is the arithmetic the waste bound rests on.
    #[test]
    fn grow_once_rounds_up_and_never_stalls() {
        assert_eq!(grow_once(1), 2);
        assert_eq!(grow_once(2), 3);
        assert_eq!(grow_once(3), 4);
        assert_eq!(grow_once(4), 5);
        assert_eq!(grow_once(5), 7);
        assert_eq!(grow_once(8), 10);
        assert_eq!(grow_once(512), 640, "model.rs:1077 INITIAL_FULL_CACHE");
        assert_eq!(grow_once(16_384), 20_480);
        assert_eq!(grow_once(usize::MAX), usize::MAX, "must not wrap or stall");
        for current in 1..4096usize {
            assert!(grow_once(current) > current, "grow_once({current}) did not grow");
        }
    }

    /// The inputs the reachable-range tests never produce: a zero capacity, a
    /// bound of zero, and a `needed` far past the bound. Each must return rather
    /// than loop.
    #[test]
    fn growth_target_handles_degenerate_inputs() {
        assert_eq!(growth_target(0, 1, MAX_POLICY_CAPACITY), Ok(1));
        assert_eq!(
            growth_target(512, 513, 512),
            Err(GpuCacheError::PositionOutOfRange {
                position: 512,
                capacity: 512,
            }),
        );
        assert_eq!(
            growth_target(1, 2, 0),
            Err(GpuCacheError::PositionOutOfRange {
                position: 1,
                capacity: 0,
            }),
        );
        assert_eq!(growth_target(4096, 4096, MAX_POLICY_CAPACITY), Ok(4096));
        assert_eq!(growth_target(4096, 1, MAX_POLICY_CAPACITY), Ok(4096));
        assert_eq!(growth_target(0, 0, 0), Ok(0));
    }

    /// A rung that overshoots the bound clamps onto it, and clamping must never
    /// leave the cache too small for `needed`.
    #[test]
    fn an_overshooting_rung_lands_exactly_on_the_bound() {
        assert_eq!(growth_target(4096, 4097, 4097), Ok(4097));
        assert_eq!(growth_target(4096, 4097, 4098), Ok(4098));
        for needed in 2049..=4096usize {
            let target = growth_target(2048, needed, needed).unwrap();
            assert_eq!(target, needed, "clamped target {target} != bound {needed}");
        }
    }
}
