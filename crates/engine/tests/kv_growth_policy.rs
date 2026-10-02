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
use kaplumba_engine::gpu_cache::{GpuCacheError, GpuKvCache, MAX_POLICY_CAPACITY};
use metal::Device;

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