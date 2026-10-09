//! Probe: `GpuKvCache` must allocate **exactly 1x** bytes at every ladder rung
//! and wipe every byte of every new position slot (`spec/llama-mlock-zero-swap`
//! R5/R6).
//!
//! Why this is a probe and not a unit test in `gpu_cache.rs`: the bug it pins
//! was invisible to `resident_bytes()` -- `resident_bytes` was always the
//! correct 1x figure, while `ensure_capacity` silently allocated `* 2 * 2` (2x)
//! bytes per buffer after the first growth. A fresh cache and a grown cache
//! therefore had *different* per-position costs, and only measuring the
//! `Buffer::length()`s after a growth step catches it.
//!
//! Non-vacuity & sabotage (AGENTS.md §3, testing.md §2): reverting
//! `crates/engine/src/gpu_cache.rs`'s `new_bytes` back to `* 2 * 2` must fail
//! [`kv_buffers_are_exactly_1x_at_every_ladder_rung`] with both byte figures in
//! the message, and restoring the `target - live` byte count must fail
//! [`the_position_tail_is_wiped_4_bytes_per_slot`]. Both sabotage runs are
//! recorded in `spec/llama-mlock-zero-swap/implementation_notes.md`.

use kaplumba_engine::gpu_cache::GpuKvCache;
use metal::Device;

/// Model's real full-attention shape: 4 KV heads x 512 head_dim -> stream_len
/// 2048, so one exact-1x buffer at 16384 slots is
/// `16384 * 2048 * 2 = 67,108,864` B (the figure testing.md pins).
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 512;
const INITIAL_FULL_CACHE: usize = 512;
/// `model.rs:3396` `max_sequence_len` handed to `ensure_capacity`.
const MAX_POSITION_EMBEDDINGS: usize = 262_144;

fn device() -> Device {
    Device::system_default().expect("a Metal device")
}

/// Exact 1x sizing of all three buffers, checked against the cache's own shape.
///
/// `expected` is derived from the cache (`capacity * stream_len * 2`), never
/// hard-coded, so the assertion states the invariant rather than one snapshot:
/// `new` allocates `capacity * stream_len * 2` per buffer, and after a ladder
/// growth `ensure_capacity` must allocate the same formula at the new capacity.
/// A 2x allocation therefore fails with both numbers visible.
fn assert_exact_1x(cache: &GpuKvCache, stage: &str) {
    let stream_len = cache.stream_len();
    let capacity = cache.capacity();
    let expected_keys_values = (capacity * stream_len * 2) as u64;
    let expected_positions = (capacity * 4) as u64;

    let keys = cache.keys().length();
    let values = cache.values().length();
    let positions = cache.positions().length();

    assert_eq!(
        keys,
        expected_keys_values,
        "{stage}: keys buffer is {keys} B but exact 1x is {expected_keys_values} B \
         for capacity {capacity} x stream_len {stream_len} x 2 B ({}x over-allocated)",
        keys / expected_keys_values.max(1),
    );
    assert_eq!(
        values,
        expected_keys_values,
        "{stage}: values buffer is {values} B but exact 1x is {expected_keys_values} B \
         for capacity {capacity} x stream_len {stream_len} x 2 B",
    );
    assert_eq!(
        positions,
        expected_positions,
        "{stage}: positions buffer is {positions} B but exact 1x is \
         {expected_positions} B (4 B per slot x {capacity} slots)",
    );
    assert_eq!(
        keys + values + positions,
        cache.resident_bytes() as u64,
        "{stage}: the three buffer lengths must sum to resident_bytes() = {}; \
         otherwise the memory budget is derived from a number nothing owns",
        cache.resident_bytes(),
    );
}

/// Walk the real growth ladder from `INITIAL_FULL_CACHE` to the policy limit at
/// the model's real shape, asserting exact 1x sizing after **every** growth
/// step -- and the explicit model-shape figure testing.md pins.
#[test]
fn kv_buffers_are_exactly_1x_at_every_ladder_rung() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, INITIAL_FULL_CACHE, KV_HEADS, HEAD_DIM).unwrap();
    assert_exact_1x(&cache, "GpuKvCache::new");

    let max = kaplumba_engine::generate::DEFAULT_CONTEXT_LIMIT;
    let mut grew_at = Vec::new();
    let mut previous_capacity = cache.capacity();
    for needed in (INITIAL_FULL_CACHE + 1)..=max {
        cache
            .ensure_capacity(&device, needed, MAX_POSITION_EMBEDDINGS)
            .unwrap_or_else(|error| panic!("ensure_capacity({needed}) failed: {error}"));
        if cache.capacity() != previous_capacity {
            previous_capacity = cache.capacity();
            grew_at.push((needed, previous_capacity));
            assert_exact_1x(&cache, &format!("after growth to {previous_capacity}"));
        }
    }
    assert_eq!(
        cache.capacity(),
        max,
        "the ladder must land exactly on the policy limit {max}",
    );
    assert!(
        (10..=25).contains(&grew_at.len()),
        "expected ~19 rungs from {INITIAL_FULL_CACHE} to {max}, saw {}: the walk \
         never exercised the growth path, so nothing was asserted about it",
        grew_at.len(),
    );
    println!(
        "KV ladder rungs: {grew_at:?} -- exact 1x at all {} growth steps",
        grew_at.len()
    );

    // The exact model-shape number from testing.md: at 16384 slots one buffer
    // must be 67,108,864 B. A fresh cache grown with a 16384 bound lands on
    // exactly 16384 (the ladder clamps onto the bound), so the failure message
    // can name both figures if the `* 2 * 2` ever comes back.
    let mut sixteen_k = GpuKvCache::new(&device, INITIAL_FULL_CACHE, KV_HEADS, HEAD_DIM).unwrap();
    sixteen_k.ensure_capacity(&device, 16_384, 16_384).unwrap();
    assert_eq!(sixteen_k.capacity(), 16_384);
    assert_eq!(
        sixteen_k.keys().length(),
        67_108_864,
        "at capacity 16384 x stream_len 2048 x 2 B the keys buffer must be \
         67,108,864 B (1x); a `* 2 * 2` allocation reports 134,217,728 B \
         (200% over-allocated!)",
    );
    assert_exact_1x(&sixteen_k, "capacity 16384");
}

/// R6: every one of the 4 bytes of every slot in `[live, target)` must be
/// `0xFF` after a growth, and the live prefix must survive untouched.
///
/// The old code wrote `target - live` **bytes** starting at `live * 4` -- a
/// quarter of the tail -- leaving the remaining slots to whatever a fresh Metal
/// buffer contains (zeroed pages in practice, which read back as slot 0, not
/// `EMPTY_POSITION`).
#[test]
fn the_position_tail_is_wiped_4_bytes_per_slot() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, 512, 1, 1).unwrap();

    // 100 live slots, written to a recognisable non-0xFF pattern so "preserved"
    // and "wiped" are distinguishable in both directions.
    let live = 100;
    cache.advance(live);
    unsafe {
        let positions = cache.positions().contents() as *mut u32;
        for slot in 0..live {
            positions.add(slot).write(0x1122_3344 + slot as u32);
        }
    }

    cache
        .ensure_capacity(&device, 2048, MAX_POSITION_EMBEDDINGS)
        .unwrap();
    let target = cache.capacity();
    assert!(target > 2048 && target > live, "target {target}");

    unsafe {
        let positions = cache.positions().contents() as *const u32;
        // Live prefix preserved.
        for slot in 0..live {
            let got = positions.add(slot).read();
            assert_eq!(
                got,
                0x1122_3344 + slot as u32,
                "slot {slot} of the live prefix was clobbered by the growth: \
                 got {got:#010x}",
            );
        }
        // Tail: all 4 bytes of every slot, including the last one -- checking
        // only the first tail slot would pass the old quarter-wipe, because the
        // old code did fill the first `(target - live)` bytes.
        for slot in live..target {
            let got = positions.add(slot).read();
            assert_eq!(
                got,
                u32::MAX,
                "R6: slot {slot} of [{live}, {target}) reads {got:#010x} instead of \
                 EMPTY_POSITION (0xFFFFFFFF) -- the tail wipe covered only {} of {} \
                 bytes (target - live was written as a byte count, not a slot count)",
                target - live,
                (target - live) * 4,
            );
        }
    }

    // A second growth: the pattern must hold across repeated rungs, not just
    // the first one.
    let live_before = cache.next_position();
    cache
        .ensure_capacity(&device, 8192, MAX_POSITION_EMBEDDINGS)
        .unwrap();
    assert!(cache.capacity() > 2048);
    unsafe {
        let positions = cache.positions().contents() as *const u32;
        for slot in 0..live_before {
            assert_eq!(
                positions.add(slot).read(),
                0x1122_3344 + slot as u32,
                "second growth clobbered live slot {slot}",
            );
        }
        for slot in live_before..cache.capacity() {
            assert_eq!(
                positions.add(slot).read(),
                u32::MAX,
                "second growth left slot {slot} of the tail unwiped",
            );
        }
    }
}

/// The live **key/value** prefix must survive a growth byte-for-byte: the copy
/// takes `live * stream_len * 2` bytes per buffer, which is exactly the 1x
/// size. A regression to either a 2x copy length (over-read) or a slot-count
/// mix-up would corrupt attention state that no short-context output check
/// would notice.
#[test]
fn the_live_key_value_prefix_survives_growth() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, 512, 2, 8).unwrap();
    let stream_len = cache.stream_len();
    let live = 64;
    cache.advance(live);

    let pattern: Vec<u16> = (0..live * stream_len)
        .map(|i| (i % 65_536) as u16)
        .collect();
    unsafe {
        let keys = cache.keys().contents() as *mut u16;
        let values = cache.values().contents() as *mut u16;
        for (i, byte) in pattern.iter().enumerate() {
            keys.add(i).write(*byte);
            values.add(i).write(byte.wrapping_add(1));
        }
    }

    cache
        .ensure_capacity(&device, 4096, MAX_POSITION_EMBEDDINGS)
        .unwrap();
    assert!(cache.capacity() > 512);

    unsafe {
        let keys = cache.keys().contents() as *const u16;
        let values = cache.values().contents() as *const u16;
        for (i, expected) in pattern.iter().enumerate() {
            let k = keys.add(i).read();
            let v = values.add(i).read();
            assert_eq!(
                k,
                *expected,
                "key element {i} of the live prefix changed across growth",
            );
            assert_eq!(
                v,
                expected.wrapping_add(1),
                "value element {i} of the live prefix changed across growth",
            );
        }
    }
    assert_exact_1x(&cache, "after prefix-copy growth");
}
