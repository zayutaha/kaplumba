use half::f16;
use kaplumba_kernels::attention::{AttentionKernels, AttentionRequest};

fn values(count: usize, offset: f32) -> Vec<u16> {
    (0..count)
        .map(|index| f16::from_f32(((index as f32 + offset) * 0.13).sin()).to_bits())
        .collect()
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn attention_expected(
    queries: &[u16],
    keys: &[u16],
    values: &[u16],
    query_positions: &[u32],
    key_positions: &[u32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    window: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; queries.len()];
    for query in 0..query_positions.len() {
        for q_head in 0..q_heads {
            let key_head = q_head / (q_heads / kv_heads);
            let query_base = (query * q_heads + q_head) * head_dim;
            let mut maximum = f32::NEG_INFINITY;
            let mut total = 0.0_f32;
            let mut accum = vec![0.0_f32; head_dim];
            for key_index in 0..key_positions.len() {
                let key_position = key_positions[key_index];
                let query_position = query_positions[query];
                if key_position > query_position
                    || (window != 0 && query_position - key_position >= window as u32)
                {
                    continue;
                }
                let key_base = (key_index * kv_heads + key_head) * head_dim;
                let mut score = 0.0_f32;
                for dimension in 0..head_dim {
                    score += f16::from_bits(queries[query_base + dimension]).to_f32()
                        * f16::from_bits(keys[key_base + dimension]).to_f32();
                }
                score *= scale;
                if maximum == f32::NEG_INFINITY {
                    maximum = score;
                    total = 1.0;
                    for dimension in 0..head_dim {
                        accum[dimension] = f16::from_bits(values[key_base + dimension]).to_f32();
                    }
                } else if score > maximum {
                    let factor = (maximum - score).exp();
                    total = total * factor + 1.0;
                    for dimension in 0..head_dim {
                        accum[dimension] = accum[dimension] * factor
                            + f16::from_bits(values[key_base + dimension]).to_f32();
                    }
                    maximum = score;
                } else {
                    let factor = (score - maximum).exp();
                    total += factor;
                    for dimension in 0..head_dim {
                        accum[dimension] +=
                            factor * f16::from_bits(values[key_base + dimension]).to_f32();
                    }
                }
            }
            if total != 0.0 {
                for dimension in 0..head_dim {
                    output[query_base + dimension] = accum[dimension] / total;
                }
            }
        }
    }
    output
}

fn assert_close(actual: &[u16], expected: &[f32], tolerance: f32, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, value) in actual.iter().enumerate() {
        let value = f16::from_bits(*value).to_f32();
        assert!(
            (value - expected[index]).abs() <= tolerance,
            "{name}[{index}] expected {} actual {value}",
            expected[index]
        );
    }
}

#[test]
fn online_attention_matches_cpu_for_decode_and_gqa() {
    let kernels = AttentionKernels::new().expect("Metal attention kernel should compile");
    let q_heads = 4;
    let kv_heads = 2;
    let head_dim = 16;
    let key_count = 5;
    let queries = values(q_heads * head_dim, 0.2);
    let keys = values(key_count * kv_heads * head_dim, 1.1);
    let values_input = values(key_count * kv_heads * head_dim, 2.2);
    let query_positions = [4];
    let key_positions = [0, 1, 2, 3, 4];
    let request = AttentionRequest {
        queries: &queries,
        keys: &keys,
        values: &values_input,
        query_positions: &query_positions,
        key_positions: &key_positions,
        q_heads,
        kv_heads,
        head_dim,
        scale: 1.0,
        window: 0,
    };
    let expected = attention_expected(
        &queries,
        &keys,
        &values_input,
        &query_positions,
        &key_positions,
        q_heads,
        kv_heads,
        head_dim,
        1.0,
        0,
    );
    let actual = kernels.online_attention(request).unwrap();
    assert_close(&actual, &expected, 5e-2, "decode attention");
    assert_eq!(kernels.working_set_bytes(&request), q_heads * head_dim * 4);
}

#[test]
fn online_attention_matches_cpu_for_prefill_and_sliding_window() {
    let kernels = AttentionKernels::new().expect("Metal attention kernel should compile");
    let q_heads = 4;
    let kv_heads = 2;
    let head_dim = 32;
    let key_count = 6;
    let queries = values(4 * q_heads * head_dim, 0.4);
    let keys = values(key_count * kv_heads * head_dim, 1.4);
    let values_input = values(key_count * kv_heads * head_dim, 2.4);
    let query_positions = [3, 4, 5, 6];
    let key_positions = [0, 1, 2, 3, 4, 5];
    let request = AttentionRequest {
        queries: &queries,
        keys: &keys,
        values: &values_input,
        query_positions: &query_positions,
        key_positions: &key_positions,
        q_heads,
        kv_heads,
        head_dim,
        scale: 1.0,
        window: 4,
    };
    let expected = attention_expected(
        &queries,
        &keys,
        &values_input,
        &query_positions,
        &key_positions,
        q_heads,
        kv_heads,
        head_dim,
        1.0,
        4,
    );
    let actual = kernels.online_attention(request).unwrap();
    assert_close(&actual, &expected, 3e-3, "prefill sliding attention");

    let full_values = keys;
    let full_request = AttentionRequest {
        queries: &queries,
        keys: &full_values,
        values: &full_values,
        query_positions: &query_positions,
        key_positions: &key_positions,
        q_heads,
        kv_heads,
        head_dim,
        scale: 1.0,
        window: 0,
    };
    let full_expected = attention_expected(
        &queries,
        &full_values,
        &full_values,
        &query_positions,
        &key_positions,
        q_heads,
        kv_heads,
        head_dim,
        1.0,
        0,
    );
    let full_actual = kernels.online_attention(full_request).unwrap();
    assert_close(&full_actual, &full_expected, 3e-3, "full K equals V");
}

#[test]
fn online_attention_supports_512_head_dimension_and_large_positions() {
    let kernels = AttentionKernels::new().expect("Metal attention kernel should compile");
    let q_heads = 2;
    let kv_heads = 1;
    let head_dim = 512;
    let queries = values(2 * q_heads * head_dim, 0.7);
    let keys = values(2 * kv_heads * head_dim, 1.7);
    let query_positions = [100_000, 100_001];
    let key_positions = [99_999, 100_000];
    let request = AttentionRequest {
        queries: &queries,
        keys: &keys,
        values: &keys,
        query_positions: &query_positions,
        key_positions: &key_positions,
        q_heads,
        kv_heads,
        head_dim,
        scale: 1.0,
        window: 0,
    };
    let expected = attention_expected(
        &queries,
        &keys,
        &keys,
        &query_positions,
        &key_positions,
        q_heads,
        kv_heads,
        head_dim,
        1.0,
        0,
    );
    let actual = kernels.online_attention(request).unwrap();
    assert_close(&actual, &expected, 3e-3, "large attention");
}

#[test]
fn online_attention_rejects_invalid_shapes() {
    let kernels = AttentionKernels::new().expect("Metal attention kernel should compile");
    let error = kernels
        .online_attention(AttentionRequest {
            queries: &[0; 4],
            keys: &[0; 4],
            values: &[0; 4],
            query_positions: &[0],
            key_positions: &[0],
            q_heads: 2,
            kv_heads: 3,
            head_dim: 2,
            scale: 1.0,
            window: 0,
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("dimensions"));

    let error = kernels
        .online_attention(AttentionRequest {
            queries: &[0; 8],
            keys: &[0; 4],
            values: &[0; 4],
            query_positions: &[1, 0],
            key_positions: &[0],
            q_heads: 1,
            kv_heads: 1,
            head_dim: 4,
            scale: 1.0,
            window: 0,
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("strictly increasing"));
}
