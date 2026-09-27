use std::mem::size_of;

use kaplumba_engine::kv_cache::{FullKvCache, KvCache, KvCacheError, SlidingKvCache};

fn stream(position: usize, length: usize) -> (Vec<u16>, Vec<u16>) {
    let key = (0..length)
        .map(|index| (position * 100 + index) as u16)
        .collect();
    let value = (0..length)
        .map(|index| (position * 200 + index) as u16)
        .collect();
    (key, value)
}

#[test]
fn full_cache_appends_chunks_and_stores_distinct_k_v() {
    let mut cache = FullKvCache::new(4, 2, 2).unwrap();
    let (keys, values) = stream(0, 8);
    assert_eq!(cache.append_chunk(0, &keys, &values).unwrap(), 0..2);
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.visible_positions(), 0..2);
    assert_eq!(cache.resident_bytes(), 4 * 4 * 2 * size_of::<u16>());

    let (key, value) = stream(2, 4);
    assert_eq!(cache.append(2, &key, &value).unwrap(), 2);
    let (stored_key, stored_value) = cache.key_value(2).unwrap();
    assert_eq!(stored_key, key);
    assert_eq!(stored_value, value);
    assert_eq!(cache.position_to_slot(2), Some(2));
    assert!(!cache.contains(4));
}

#[test]
fn full_cache_rejects_position_and_capacity_errors_without_mutation() {
    let mut cache = FullKvCache::new(2, 1, 2).unwrap();
    let (key, value) = stream(0, 2);
    assert_eq!(
        cache.append(1, &key, &value),
        Err(KvCacheError::InvalidPosition {
            expected: 0,
            got: 1
        })
    );
    cache.append(0, &key, &value).unwrap();
    let (key_one, value_one) = stream(1, 2);
    cache.append(1, &key_one, &value_one).unwrap();
    assert_eq!(
        cache.append(2, &key_one, &value_one),
        Err(KvCacheError::PositionOutOfRange {
            position: 2,
            capacity: 2
        })
    );
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.append_chunk(2, &[0; 4], &[0; 4]),
        Err(KvCacheError::PositionOutOfRange {
            position: 3,
            capacity: 2
        })
    );
    assert_eq!(cache.len(), 2);
    let error = FullKvCache::new(0, 1, 2).err().unwrap();
    assert_eq!(error, KvCacheError::InvalidDimensions);
}

#[test]
fn sliding_cache_uses_inclusive_window_and_rollover() {
    let mut cache = SlidingKvCache::with_capacity(4, 2, 2).unwrap();
    let keys: Vec<u16> = (0..16).map(|index| index as u16).collect();
    let values: Vec<u16> = (0..16)
        .map(|index| (index as u16).wrapping_add(1000))
        .collect();
    cache.append_chunk(0, &keys, &values).unwrap();
    assert_eq!(cache.visible_positions(), 0..4);
    assert!(cache.contains(0));
    assert!(cache.contains(3));
    assert!(!cache.contains(4));
    assert_eq!(cache.position_to_slot(0), Some(0));
    assert_eq!(cache.position_to_slot(3), Some(3));

    let (key, value) = stream(4, 4);
    assert_eq!(cache.append(4, &key, &value).unwrap(), 0);
    assert_eq!(cache.visible_positions(), 1..5);
    assert!(!cache.contains(0));
    assert!(cache.contains(1));
    assert_eq!(cache.position_to_slot(4), Some(0));
    let (stored_key, stored_value) = cache.key_value(4).unwrap();
    assert_eq!(stored_key, key);
    assert_eq!(stored_value, value);
    assert!(cache.key_value(0).is_none());
}

#[test]
fn sliding_cache_handles_multi_chunk_8k_positions() {
    let stream_len = 16 * 256;
    let mut cache = SlidingKvCache::with_capacity(1024, 16, 256).unwrap();
    let chunk_elements = 1024 * stream_len;
    for chunk in 0..8 {
        let keys = vec![(chunk * 1000) as u16; chunk_elements];
        let values = vec![(chunk * 2000) as u16; chunk_elements];
        let start = chunk * 1024;
        assert_eq!(
            cache.append_chunk(start, &keys, &values).unwrap(),
            start..start + 1024
        );
    }
    assert_eq!(cache.len(), 8192);
    assert_eq!(cache.visible_positions(), 7168..8192);
    assert!(cache.contains(7168));
    assert!(!cache.contains(7167));
    assert!(cache.contains(8191));
    assert_eq!(cache.position_to_slot(7168), Some(0));
    assert_eq!(cache.position_to_slot(8191), Some(1023));
    assert_eq!(
        cache.resident_bytes(),
        1024 * stream_len * 2 * size_of::<u16>()
    );
}

#[test]
fn cache_enum_preserves_layout_specific_behavior() {
    let mut full = KvCache::full(2, 1, 2).unwrap();
    let mut sliding = KvCache::sliding(1, 2).unwrap();
    let (key, value) = stream(0, 2);
    full.append(0, &key, &value).unwrap();
    sliding.append(0, &key, &value).unwrap();
    assert_eq!(full.visible_positions(), 0..1);
    assert_eq!(sliding.visible_positions(), 0..1);
    assert_eq!(full.key_value(0).unwrap().0, key);
    assert_eq!(sliding.key_value(0).unwrap().1, value);
    assert_eq!(full.resident_bytes(), 16);
    assert_eq!(sliding.resident_bytes(), 8192);
}

#[test]
fn cache_rejects_invalid_stream_and_empty_chunks() {
    let mut full = FullKvCache::new(4, 2, 2).unwrap();
    assert_eq!(
        full.append(0, &[0; 3], &[0; 4]),
        Err(KvCacheError::InvalidStreamLength {
            actual: 3,
            expected: 4
        })
    );
    assert_eq!(
        full.append_chunk(0, &[], &[]),
        Err(KvCacheError::EmptyChunk)
    );
    assert_eq!(
        full.append_chunk(0, &[0; 5], &[0; 5]),
        Err(KvCacheError::InvalidStreamLength {
            actual: 5,
            expected: 4
        })
    );
    assert!(matches!(
        SlidingKvCache::new(0, 2),
        Err(KvCacheError::InvalidDimensions)
    ));
}
