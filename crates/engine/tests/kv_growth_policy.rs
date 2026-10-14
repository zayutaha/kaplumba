//! The KV growth policy: what a cache is allowed to grow to, and how fast.
//!
//! `GpuKvCache::ensure_capacity` is the only place the engine allocates KV, and
//! its `max_capacity` argument is the checkpoint's `max_position_embeddings`
//! (262144 for this model, `model.rs:3396` -> `model.rs:1663` / `:1783`) -- an
//! order of magnitude above the policy limit `DEFAULT_CONTEXT_LIMIT` = 32768
//! that `generate.rs:35` enforces before every forward. A caller that reaches
//! `TextModel::forward_buffer` directly never runs that check, and at 262144
//! positions the full-attention KV alone is 20.79 GiB on a 24 GiB box.
//!
//! No model and no checkpoint: `GpuKvCache` allocates its own Metal buffers, and
//! the shapes used here are deliberately tiny (`kv_heads * head_dim == 1`) so the
//! whole 32768-position walk costs ~192 KiB instead of the 0.54 GiB the model's
//! real shape would.

use kaplumba_engine::generate::DEFAULT_CONTEXT_LIMIT;
use kaplumba_engine::gpu_cache::{
    EMPTY_POSITION, GpuCacheError, GpuKvCache, MAX_POLICY_CAPACITY, sliding_capacity_needed,
    wait_for_sliding_growth,
};
use kaplumba_kernels::elements::ElementKernels;
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Device, MTLResourceOptions};

/// What `model.rs:3396` puts in `LayerConfig::max_sequence_len`.
const MAX_POSITION_EMBEDDINGS: usize = 262_144;
/// `model.rs:1077`, `INITIAL_FULL_CACHE`.
const INITIAL_FULL_CACHE: usize = 512;

fn device() -> Device {
    Device::system_default().expect("a Metal device")
}

#[test]
fn capacity_is_clamped_to_the_policy_limit() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, INITIAL_FULL_CACHE, 1, 1).unwrap();

    // Walk every position the policy limit permits, asking for the checkpoint's
    // own bound each time -- which is exactly what model.rs:1663 / :1783 do.
    for needed in 1..=DEFAULT_CONTEXT_LIMIT {
        cache
            .ensure_capacity(&device, needed, MAX_POSITION_EMBEDDINGS)
            .unwrap_or_else(|error| panic!("ensure_capacity({needed}) failed: {error}"));
        assert!(
            cache.capacity() <= DEFAULT_CONTEXT_LIMIT,
            "capacity {} exceeds the policy limit {DEFAULT_CONTEXT_LIMIT} after asking for {needed}",
            cache.capacity(),
        );
    }
    assert_eq!(
        cache.capacity(),
        DEFAULT_CONTEXT_LIMIT,
        "the ladder should land exactly on the policy limit"
    );
    assert_eq!(MAX_POLICY_CAPACITY, DEFAULT_CONTEXT_LIMIT);

    // One past the policy limit is refused, and the error names the *clamped*
    // bound rather than the 262144 the caller passed.
    let before = cache.capacity();
    assert_eq!(
        cache.ensure_capacity(&device, DEFAULT_CONTEXT_LIMIT + 1, MAX_POSITION_EMBEDDINGS),
        Err(GpuCacheError::PositionOutOfRange {
            position: DEFAULT_CONTEXT_LIMIT,
            capacity: DEFAULT_CONTEXT_LIMIT,
        }),
        "position {} must be refused with the policy bound in the error",
        DEFAULT_CONTEXT_LIMIT,
    );
    assert_eq!(
        cache.capacity(),
        before,
        "a refused growth must not have reallocated"
    );
}

#[test]
fn the_clamp_does_not_weaken_a_smaller_caller_bound() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, INITIAL_FULL_CACHE, 1, 1).unwrap();

    // A caller asking for less than the policy still gets its own bound.
    // `position` is the first position that does not fit, so `needed - 1`.
    assert_eq!(
        cache.ensure_capacity(&device, 1001, 1000),
        Err(GpuCacheError::PositionOutOfRange {
            position: 1000,
            capacity: 1000,
        }),
    );
    assert_eq!(cache.capacity(), INITIAL_FULL_CACHE);

    cache.ensure_capacity(&device, 1000, 1000).unwrap();
    assert_eq!(cache.capacity(), 1000, "the target is clamped up to the bound");

    // And a bound below the *current* capacity cannot shrink the cache.
    cache.ensure_capacity(&device, 1000, 10).unwrap();
    assert_eq!(cache.capacity(), 1000);
}

/// `resident_bytes` must account for all three buffers, and it is the number the
/// memory budget is derived from. (The host cache in `kv_cache.rs` used to omit
/// `positions`; see `tests/kv_cache.rs`.)
#[test]
fn resident_bytes_counts_keys_values_and_positions() {
    let device = device();
    let cache = GpuKvCache::new(&device, 1000, 4, 8).unwrap();
    let stream_len = 4 * 8;
    assert_eq!(cache.stream_len(), stream_len);
    assert_eq!(
        cache.resident_bytes(),
        1000 * stream_len * 2 * 2 + 1000 * 4,
    );
    assert_eq!(
        cache.keys().length()
            + cache.values().length()
            + cache.positions().length(),
        cache.resident_bytes() as u64,
        "resident_bytes must equal the sum of the three buffers it owns"
    );
}

#[test]
fn sliding_capacity_covers_live_window_and_whole_append_chunk() {
    assert_eq!(sliding_capacity_needed(0, 2, 1024), Ok(2));
    assert_eq!(sliding_capacity_needed(1024, 1, 1024), Ok(1025));
    assert_eq!(sliding_capacity_needed(4096, 1024, 1024), Ok(2048));
    assert_eq!(
        sliding_capacity_needed(usize::MAX, 2, usize::MAX),
        Err(GpuCacheError::ShapeOverflow)
    );
    assert_eq!(
        sliding_capacity_needed(0, 0, 1024),
        Err(GpuCacheError::InvalidDimensions)
    );
}

#[test]
fn sliding_growth_remaps_wrapped_positions_and_preserves_kv_bytes() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, 3, 1, 2).unwrap();
    for absolute in 0..7usize {
        let slot = absolute % cache.capacity();
        unsafe {
            let keys = cache.keys().contents() as *mut u16;
            let values = cache.values().contents() as *mut u16;
            *keys.add(slot * 2) = absolute as u16;
            *keys.add(slot * 2 + 1) = (absolute + 100) as u16;
            *values.add(slot * 2) = (absolute + 200) as u16;
            *values.add(slot * 2 + 1) = (absolute + 300) as u16;
            *((cache.positions().contents() as *mut u32).add(slot)) = absolute as u32;
        }
        cache.advance(1);
    }

    cache.ensure_sliding_capacity(&device, 3, 1, 8).unwrap();
    assert_eq!(cache.capacity(), 4);
    let (keys, values, positions) = cache.read_to_host();
    for absolute in 4..7usize {
        let slot = absolute % cache.capacity();
        assert_eq!(positions[slot], absolute as u32);
        assert_eq!(
            &keys[slot * 2..slot * 2 + 2],
            &[absolute as u16, (absolute + 100) as u16]
        );
        assert_eq!(
            &values[slot * 2..slot * 2 + 2],
            &[(absolute + 200) as u16, (absolute + 300) as u16]
        );
    }
    assert_eq!(positions[3], EMPTY_POSITION);
    assert_eq!(cache.resident_bytes(), 4 * 2 * 2 * 2 + 4 * 4);

    // Growth is a no-op when the current capacity already fits.
    cache.ensure_sliding_capacity(&device, 3, 1, 8).unwrap();
    assert_eq!(cache.capacity(), 4);
}

#[test]
fn sliding_growth_rejects_missing_live_slots_and_insufficient_bound() {
    let device = device();
    let mut missing = GpuKvCache::new(&device, 3, 1, 2).unwrap();
    missing.advance(1);
    assert_eq!(
        missing.ensure_sliding_capacity(&device, 3, 3, 8),
        Err(GpuCacheError::SlidingPositionMismatch {
            slot: 0,
            expected: 0,
            got: EMPTY_POSITION,
        })
    );
    assert_eq!(missing.capacity(), 3);

    let mut bounded = GpuKvCache::new(&device, 3, 1, 2).unwrap();
    assert_eq!(
        bounded.ensure_sliding_capacity(&device, 3, 4, 3),
        Err(GpuCacheError::PositionOutOfRange {
            position: 3,
            capacity: 3,
        })
    );
    assert_eq!(bounded.capacity(), 3);
}

#[test]
fn sliding_growth_keeps_window_disjoint_from_a_multi_token_append() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, 3, 1, 1).unwrap();
    for absolute in 0..8usize {
        let slot = absolute % cache.capacity();
        unsafe {
            *((cache.positions().contents() as *mut u32).add(slot)) = absolute as u32;
            *((cache.keys().contents() as *mut u16).add(slot)) = absolute as u16;
            *((cache.values().contents() as *mut u16).add(slot)) = (absolute + 20) as u16;
        }
        cache.advance(1);
    }
    cache.ensure_sliding_capacity(&device, 3, 3, 8).unwrap();
    assert!(cache.capacity() >= 6);

    let mut live_slots = Vec::new();
    for absolute in 5..11 {
        live_slots.push(cache.slot_for(absolute, false));
    }
    live_slots.sort_unstable();
    live_slots.dedup();
    assert_eq!(
        live_slots.len(),
        6,
        "the append must not overwrite its live window"
    );
    let (keys, values, positions) = cache.read_to_host();
    for absolute in 5..8usize {
        let slot = cache.slot_for(absolute, false);
        assert_eq!(positions[slot], absolute as u32);
        assert_eq!(keys[slot], absolute as u16);
        assert_eq!(values[slot], (absolute + 20) as u16);
    }
}

#[test]
fn sliding_growth_after_retract_copies_only_the_rewound_live_prefix() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, 3, 1, 1).unwrap();
    for absolute in 0..5usize {
        let slot = absolute % cache.capacity();
        unsafe {
            *((cache.positions().contents() as *mut u32).add(slot)) = absolute as u32;
            *((cache.keys().contents() as *mut u16).add(slot)) = absolute as u16;
            *((cache.values().contents() as *mut u16).add(slot)) = (absolute + 10) as u16;
        }
        cache.advance(1);
    }
    cache.retract(1);
    cache.ensure_sliding_capacity(&device, 2, 2, 8).unwrap();
    let (keys, values, positions) = cache.read_to_host();
    for absolute in 2..4usize {
        let slot = cache.slot_for(absolute, false);
        assert_eq!(positions[slot], absolute as u32);
        assert_eq!(keys[slot], absolute as u16);
        assert_eq!(values[slot], (absolute + 10) as u16);
    }
    assert_eq!(positions[cache.slot_for(4, false)], EMPTY_POSITION);

    cache.reset();
    assert!(cache.is_empty());
    assert!(cache.read_to_host().2.iter().all(|p| *p == EMPTY_POSITION));
}

#[test]
fn sliding_growth_waits_for_queued_gpu_writes_before_migration() {
    let device = device();
    let queue = device.new_command_queue();
    let mut encoder = MetalEncoder::new(&queue).unwrap();
    let kernels = ElementKernels::shared().unwrap();
    let mut cache = GpuKvCache::new(&device, 2, 1, 1).unwrap();
    unsafe {
        *(cache.positions().contents() as *mut u32) = 0;
    }
    cache.advance(1);

    let source = device.new_buffer(2, metal::MTLResourceOptions::StorageModeShared);
    unsafe {
        *(source.contents() as *mut u16) = 0x1234;
    }
    kernels
        .encode_copy(&mut encoder, &source, cache.keys(), 1)
        .unwrap();
    let before = encoder.command_buffers_created();
    wait_for_sliding_growth(&mut encoder).unwrap();
    assert_eq!(
        encoder.command_buffers_created(),
        before + 1,
        "growth barrier must commit, wait for, and replace the pending command buffer"
    );

    cache.ensure_sliding_capacity(&device, 2, 2, 8).unwrap();
    assert_eq!(unsafe { *(cache.keys().contents() as *const u16) }, 0x1234);
}

#[test]
fn sliding_growth_waits_for_a_separate_encoder_before_migration() {
    let device = device();
    let queue = device.new_command_queue();
    let kernels = ElementKernels::shared().unwrap();
    let mut cache = GpuKvCache::new(&device, 15, 1, 1).unwrap();
    let keys = device.new_buffer(15 * 2, MTLResourceOptions::StorageModeShared);
    let values = device.new_buffer(15 * 2, MTLResourceOptions::StorageModeShared);
    unsafe {
        let key_ptr = keys.contents() as *mut u16;
        let value_ptr = values.contents() as *mut u16;
        for position in 0..15usize {
            *key_ptr.add(position) = (position + 10) as u16;
            *value_ptr.add(position) = (position + 30) as u16;
        }
    }

    // Simulate the prior pipelined decode encoder: it writes all current tags
    // and commits, while the next forward uses a distinct, empty encoder.
    let mut producer = MetalEncoder::new(&queue).unwrap();
    kernels
        .encode_cache_append(
            &mut producer,
            &keys,
            &values,
            cache.keys(),
            cache.values(),
            cache.positions(),
            15,
            1,
            cache.capacity(),
            0,
            0,
        )
        .unwrap();
    cache.advance(15);
    producer.commit_terminal().unwrap();

    let mut next_forward = MetalEncoder::new(&queue).unwrap();
    wait_for_sliding_growth(&mut next_forward).unwrap();
    cache
        .ensure_sliding_capacity(&device, 1024, 1, 2048)
        .unwrap();

    let (actual_keys, actual_values, positions) = cache.read_to_host();
    for position in 0..15usize {
        let slot = position % cache.capacity();
        assert_eq!(positions[slot], position as u32, "position {position}");
        assert_eq!(actual_keys[slot], (position + 10) as u16);
        assert_eq!(actual_values[slot], (position + 30) as u16);
    }
    next_forward.abandon();
}
