use half::f16;
use kaplumba_engine::exl3::{
    BitRate, Codebook, SignVector, pack_integer_states, reconstruct_matrix,
};
use kaplumba_engine::model::{
    DecoderLayer, DecoderLayerWeights, LayerConfig, QuantizedLinear, RopeMode, TextModel,
};
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};

fn half(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

fn value_at(index: usize, offset: f32) -> u16 {
    half(((index as f32 + offset) * 0.071).sin())
}

fn make_linear_with_rate(
    in_features: usize,
    out_features: usize,
    bits: u8,
    codebook: Exl3Codebook,
    seed: u32,
) -> QuantizedLinear {
    let mut packed = Vec::new();
    let tile_count = (in_features / 16) * (out_features / 16);
    let mask = (1_u32 << bits) - 1;
    for tile in 0..tile_count {
        let mut states = vec![0_u16; 256];
        let mut value = seed.wrapping_add(tile as u32 * 7919);
        for state in &mut states {
            value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *state = (value & mask) as u16;
        }
        let (words, _) = pack_integer_states(&states, bits).unwrap();
        packed.extend(words);
    }
    QuantizedLinear::new(
        in_features,
        out_features,
        Exl3Rate::Integer(bits),
        codebook,
        packed,
        vec![half(0.01); in_features],
        vec![half(0.01); out_features],
    )
    .unwrap()
}

fn make_linear(in_features: usize, out_features: usize, seed: u32) -> QuantizedLinear {
    make_linear_with_rate(in_features, out_features, 3, Exl3Codebook::Mcg, seed)
}

fn config(full: bool) -> LayerConfig {
    LayerConfig {
        hidden_size: 128,
        intermediate_size: 256,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 128,
        max_sequence_len: 8,
        full_attention: full,
        sliding_window: if full { 0 } else { 3 },
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_mode: RopeMode::Default,
        rope_partial_factor: 1.0,
    }
}

fn weights(config: &LayerConfig, seed: u32) -> DecoderLayerWeights {
    DecoderLayerWeights {
        q_proj: make_linear(config.hidden_size, config.q_heads * config.head_dim, seed),
        k_proj: make_linear(
            config.hidden_size,
            config.kv_heads * config.head_dim,
            seed + 1,
        ),
        v_proj: if config.full_attention {
            None
        } else {
            Some(make_linear(
                config.hidden_size,
                config.kv_heads * config.head_dim,
                seed + 6,
            ))
        },
        o_proj: make_linear(
            config.q_heads * config.head_dim,
            config.hidden_size,
            seed + 2,
        ),
        gate_proj: make_linear(config.hidden_size, config.intermediate_size, seed + 3),
        up_proj: make_linear(config.hidden_size, config.intermediate_size, seed + 4),
        down_proj: make_linear(config.intermediate_size, config.hidden_size, seed + 5),
        q_norm: vec![half(1.0); config.head_dim],
        k_norm: vec![half(1.0); config.head_dim],
        input_norm: vec![half(1.0); config.hidden_size],
        post_attention_norm: vec![half(1.0); config.hidden_size],
        pre_ffn_norm: vec![half(1.0); config.hidden_size],
        post_ffn_norm: vec![half(1.0); config.hidden_size],
        layer_scalar: 0.9,
    }
}

fn linear_cpu(linear: &QuantizedLinear, input: &[u16], tokens: usize) -> Vec<u16> {
    let rate = match linear.rate {
        Exl3Rate::Integer(bits) => BitRate::Integer(bits),
        Exl3Rate::Fractional(bits) => BitRate::Fractional(bits),
    };
    let codebook = match linear.codebook {
        Exl3Codebook::Plain => Codebook::Plain,
        Exl3Codebook::Mcg => Codebook::Mcg,
        Exl3Codebook::Mul1 => Codebook::Mul1,
    };
    let matrix = reconstruct_matrix(
        &linear.packed,
        linear.in_features / 16,
        linear.out_features / 16,
        rate,
        codebook,
        &SignVector::from_half_bits(linear.input_signs.clone()),
        &SignVector::from_half_bits(linear.output_signs.clone()),
    )
    .unwrap();
    let mut output = vec![0_u16; tokens * linear.out_features];
    for token in 0..tokens {
        for out in 0..linear.out_features {
            let mut sum = 0.0_f32;
            for value in 0..linear.in_features {
                sum += f16::from_bits(input[token * linear.in_features + value]).to_f32()
                    * matrix[value * linear.out_features + out];
            }
            output[token * linear.out_features + out] = half(sum);
        }
    }
    output
}

fn norm_cpu(
    input: &[u16],
    weight: Option<&[u16]>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Vec<u16> {
    let mut output = vec![0_u16; rows * width];
    for row in 0..rows {
        let base = row * width;
        let sum = (0..width)
            .map(|index| {
                let value = f16::from_bits(input[base + index]).to_f32();
                value * value
            })
            .sum::<f32>();
        let scale = 1.0 / (sum / width as f32 + eps).sqrt();
        for index in 0..width {
            let value = f16::from_bits(input[base + index]).to_f32() * scale;
            output[base + index] = half(weight.map_or(value, |weights| {
                value * f16::from_bits(weights[index]).to_f32()
            }));
        }
    }
    output
}

fn rope_cpu(
    input: &[u16],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
) -> Vec<u16> {
    let half_dim = head_dim / 2;
    let mut output = input.to_vec();
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for pair in 0..half_dim {
                let angle =
                    (offset + token) as f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let cosine = angle.cos();
                let sine = angle.sin();
                let left = f16::from_bits(input[base + pair]).to_f32();
                let right = f16::from_bits(input[base + half_dim + pair]).to_f32();
                output[base + pair] = half(left * cosine - right * sine);
                output[base + half_dim + pair] = half(right * cosine + left * sine);
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn attention_cpu(
    queries: &[u16],
    keys: &[u16],
    values: &[u16],
    query_positions: &[u32],
    key_positions: &[u32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
) -> Vec<u16> {
    let mut output = vec![0_u16; queries.len()];
    for query in 0..query_positions.len() {
        for q_head in 0..q_heads {
            let key_head = q_head / (q_heads / kv_heads);
            let query_base = (query * q_heads + q_head) * head_dim;
            let mut maximum = f32::NEG_INFINITY;
            let mut total = 0.0_f32;
            let mut accum = vec![0.0_f32; head_dim];
            for key_index in 0..key_positions.len() {
                if key_positions[key_index] > query_positions[query]
                    || (window != 0
                        && query_positions[query] - key_positions[key_index] >= window as u32)
                {
                    continue;
                }
                let key_base = (key_index * kv_heads + key_head) * head_dim;
                let mut score = 0.0_f32;
                for dimension in 0..head_dim {
                    score += f16::from_bits(queries[query_base + dimension]).to_f32()
                        * f16::from_bits(keys[key_base + dimension]).to_f32();
                }
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
            for dimension in 0..head_dim {
                output[query_base + dimension] = half(accum[dimension] / total);
            }
        }
    }
    output
}

fn layer_cpu(
    input: &[u16],
    all_positions: &[u32],
    layer_weights: &DecoderLayerWeights,
    layer_config: &LayerConfig,
) -> Vec<u16> {
    let tokens = all_positions.len();
    let normalized = norm_cpu(
        input,
        Some(&layer_weights.input_norm),
        tokens,
        layer_config.hidden_size,
        layer_config.rms_norm_eps,
    );
    let q = linear_cpu(&layer_weights.q_proj, &normalized, tokens);
    let k = linear_cpu(&layer_weights.k_proj, &normalized, tokens);
    let raw_v = layer_weights
        .v_proj
        .as_ref()
        .map(|v_proj| linear_cpu(v_proj, &normalized, tokens))
        .unwrap_or_else(|| k.clone());
    let q = norm_cpu(
        &q,
        Some(&layer_weights.q_norm),
        tokens * layer_config.q_heads,
        layer_config.head_dim,
        layer_config.rms_norm_eps,
    );
    let k = norm_cpu(
        &k,
        Some(&layer_weights.k_norm),
        tokens * layer_config.kv_heads,
        layer_config.head_dim,
        layer_config.rms_norm_eps,
    );
    let v = norm_cpu(
        &raw_v,
        None,
        tokens * layer_config.kv_heads,
        layer_config.head_dim,
        layer_config.rms_norm_eps,
    );
    let q = rope_cpu(
        &q,
        tokens,
        layer_config.q_heads,
        layer_config.head_dim,
        all_positions[0] as usize,
        layer_config.rope_theta,
    );
    let k = rope_cpu(
        &k,
        tokens,
        layer_config.kv_heads,
        layer_config.head_dim,
        all_positions[0] as usize,
        layer_config.rope_theta,
    );
    let attention = attention_cpu(
        &q,
        &k,
        &v,
        all_positions,
        all_positions,
        layer_config.q_heads,
        layer_config.kv_heads,
        layer_config.head_dim,
        if layer_config.full_attention {
            0
        } else {
            layer_config.sliding_window
        },
    );
    let mut output = linear_cpu(&layer_weights.o_proj, &attention, tokens);
    output = norm_cpu(
        &output,
        Some(&layer_weights.post_attention_norm),
        tokens,
        layer_config.hidden_size,
        layer_config.rms_norm_eps,
    );
    let residual = (0..output.len())
        .map(|index| {
            half(f16::from_bits(input[index]).to_f32() + f16::from_bits(output[index]).to_f32())
        })
        .collect::<Vec<_>>();
    let normalized = norm_cpu(
        &residual,
        Some(&layer_weights.pre_ffn_norm),
        tokens,
        layer_config.hidden_size,
        layer_config.rms_norm_eps,
    );
    let gate = linear_cpu(&layer_weights.gate_proj, &normalized, tokens);
    let up = linear_cpu(&layer_weights.up_proj, &normalized, tokens);
    let activated = (0..gate.len())
        .map(|index| {
            let value = f16::from_bits(gate[index]).to_f32();
            let cubic = value * value * value;
            let gelu = 0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044715 * cubic)).tanh());
            half(gelu * f16::from_bits(up[index]).to_f32())
        })
        .collect::<Vec<_>>();
    let mut mlp = linear_cpu(&layer_weights.down_proj, &activated, tokens);
    mlp = norm_cpu(
        &mlp,
        Some(&layer_weights.post_ffn_norm),
        tokens,
        layer_config.hidden_size,
        layer_config.rms_norm_eps,
    );
    (0..residual.len())
        .map(|index| {
            half(
                (f16::from_bits(residual[index]).to_f32() + f16::from_bits(mlp[index]).to_f32())
                    * layer_weights.layer_scalar,
            )
        })
        .collect()
}

fn assert_layer_close(actual: &[u16], expected: &[u16], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, value) in actual.iter().enumerate() {
        let actual = f16::from_bits(*value).to_f32();
        let expected = f16::from_bits(expected[index]).to_f32();
        assert!(
            (actual - expected).abs() < 0.5,
            "{name}[{index}] expected {expected} actual {actual}"
        );
    }
}

#[test]
fn decoder_layer_matches_cpu_for_full_attention_chunks() {
    let layer_config = config(true);
    let layer_weights = weights(&layer_config, 101);
    let input: Vec<u16> = (0..3 * layer_config.hidden_size)
        .map(|index| value_at(index, 0.4))
        .collect();
    let expected = layer_cpu(&input, &[0, 1, 2], &layer_weights, &layer_config);
    let mut layer = DecoderLayer::new(layer_config, layer_weights).unwrap();
    let first = layer
        .forward(&input[..2 * layer_config.hidden_size], &[0, 1])
        .unwrap();
    let second = layer
        .forward(&input[2 * layer_config.hidden_size..], &[2])
        .unwrap();
    let mut actual = first;
    actual.extend(second);
    assert_layer_close(&actual, &expected, "full layer");
    assert_eq!(layer.next_position(), 3);
    assert!(layer.cache_bytes() > 0);
}

#[test]
fn decoder_layer_matches_cpu_for_sliding_attention_chunks() {
    let layer_config = config(false);
    let layer_weights = weights(&layer_config, 211);
    let input: Vec<u16> = (0..3 * layer_config.hidden_size)
        .map(|index| value_at(index, 1.4))
        .collect();
    let expected = layer_cpu(&input, &[0, 1, 2], &layer_weights, &layer_config);
    let mut layer = DecoderLayer::new(layer_config, layer_weights).unwrap();
    let first = layer
        .forward(&input[..2 * layer_config.hidden_size], &[0, 1])
        .unwrap();
    let second = layer
        .forward(&input[2 * layer_config.hidden_size..], &[2])
        .unwrap();
    let mut actual = first;
    actual.extend(second);
    assert_layer_close(&actual, &expected, "sliding layer");
}

#[test]
fn text_model_matches_sequential_layer_cpu_reference() {
    let layer_config = config(true);
    let layer_weights = vec![weights(&layer_config, 501), weights(&layer_config, 601)];
    let vocab_size = 256;
    let embedding: Vec<u16> = (0..vocab_size * layer_config.hidden_size)
        .map(|index| half(((index as f32 + 0.2) * 0.013).sin()))
        .collect();
    let final_norm = vec![half(1.0); layer_config.hidden_size];
    let lm_head = make_linear_with_rate(
        layer_config.hidden_size,
        vocab_size,
        6,
        Exl3Codebook::Mcg,
        701,
    );
    let mut expected_hidden = (0..3 * layer_config.hidden_size)
        .map(|index| {
            let token = index / layer_config.hidden_size;
            let column = index % layer_config.hidden_size;
            half(
                f16::from_bits(embedding[token * layer_config.hidden_size + column]).to_f32()
                    * (layer_config.hidden_size as f32).sqrt(),
            )
        })
        .collect::<Vec<_>>();
    for layer_weights in &layer_weights {
        expected_hidden = layer_cpu(&expected_hidden, &[0, 1, 2], layer_weights, &layer_config);
    }
    let expected_hidden = norm_cpu(
        &expected_hidden,
        Some(&final_norm),
        3,
        layer_config.hidden_size,
        layer_config.rms_norm_eps,
    );
    let expected_logits = linear_cpu(&lm_head, &expected_hidden, 3)
        .into_iter()
        .map(|value| half((f16::from_bits(value).to_f32() / 30.0).tanh() * 30.0))
        .collect::<Vec<_>>();

    let layers = layer_weights
        .clone()
        .into_iter()
        .map(|weights| DecoderLayer::new(layer_config, weights).unwrap())
        .collect();
    let mut model = TextModel::new(embedding, layers, final_norm, lm_head, 30.0).unwrap();
    let tokens = [1_u32, 2, 3];
    let first = model.forward(&tokens[..2], &[0, 1]).unwrap();
    let second = model.forward(&tokens[2..], &[2]).unwrap();
    let mut actual = first;
    actual.extend(second);
    assert_layer_close(&actual, &expected_logits, "text model");
    assert_eq!(model.layer_count(), 2);
    assert_eq!(model.vocab_size(), vocab_size);
    assert_eq!(model.next_position(), 3);
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and allocates the full text checkpoint"]
fn real_checkpoint_loads_dedicated_metal_buffers() {
    let path = std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required");
    let model = TextModel::load_from_dir(std::path::Path::new(&path)).unwrap();
    assert_eq!(model.layer_count(), 60);
    assert_eq!(model.hidden_size(), 5376);
    assert_eq!(model.vocab_size(), 262144);
    assert!(model.cache_bytes() > 0);
}

#[test]
fn decoder_layer_rejects_invalid_weights_and_positions() {
    let layer_config = config(true);
    let mut layer_weights = weights(&layer_config, 307);
    layer_weights.layer_scalar = f32::NAN;
    assert!(DecoderLayer::new(layer_config, layer_weights).is_err());

    let layer_config = config(true);
    let layer_weights = weights(&layer_config, 401);
    let mut layer = DecoderLayer::new(layer_config, layer_weights).unwrap();
    let input = vec![0; 128];
    let error = layer.forward(&input, &[1]).unwrap_err().to_string();
    assert!(error.contains("cache position"));
    let input_two = vec![0; 256];
    let error = layer.forward(&input_two, &[0, 2]).unwrap_err().to_string();
    assert!(error.contains("contiguous"));
}
