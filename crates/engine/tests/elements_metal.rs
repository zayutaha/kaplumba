use half::f16;
use kaplumba_kernels::elements::{ElementKernels, RopeRequest};

fn values(count: usize, offset: f32) -> Vec<u16> {
    (0..count)
        .map(|index| f16::from_f32((index as f32 + offset).sin() * 0.7).to_bits())
        .collect()
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

fn norm_expected(
    input: &[u16],
    weight: Option<&[u16]>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; rows * width];
    for row in 0..rows {
        let base = row * width;
        let sum = (0..width)
            .map(|index| {
                let value = f16::from_bits(input[base + index]).to_f32();
                value * value
            })
            .sum::<f32>();
        let inverse = 1.0 / (sum / width as f32 + eps).sqrt();
        for index in 0..width {
            let value = f16::from_bits(input[base + index]).to_f32() * inverse;
            output[base + index] = weight
                .map(|values| value * f16::from_bits(values[index]).to_f32())
                .unwrap_or(value);
        }
    }
    output
}

fn gelu_expected(value: f32) -> f32 {
    let coefficient = 0.797_884_6_f32;
    0.5 * value * (1.0 + (coefficient * (value + 0.044715 * value.powi(3))).tanh())
}

#[allow(clippy::too_many_arguments)]
fn rope_expected(
    input: &[u16],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
    partial_factor: f32,
    proportional: bool,
) -> Vec<f32> {
    let half = head_dim / 2;
    let rotated_pairs = if proportional {
        (partial_factor * head_dim as f32 / 2.0) as usize
    } else {
        half
    };
    let mut output = vec![0.0; input.len()];
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for column in 0..head_dim {
                let pair = if column < half { column } else { column - half };
                if pair >= rotated_pairs {
                    output[base + column] = f16::from_bits(input[base + column]).to_f32();
                    continue;
                }
                let angle =
                    (offset + token) as f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let cosine = angle.cos();
                let sine = angle.sin();
                let value = f16::from_bits(input[base + column]).to_f32();
                output[base + column] = if column < half {
                    value * cosine - f16::from_bits(input[base + half + pair]).to_f32() * sine
                } else {
                    value * cosine + f16::from_bits(input[base + pair]).to_f32() * sine
                };
            }
        }
    }
    output
}

#[test]
fn element_embedding_and_norms_match_cpu() {
    let kernels = ElementKernels::new().expect("Metal element kernels should compile");
    let tokens = [2_u32, 0, 3, 1, 2];
    let hidden = 5;
    let vocab = 4;
    let table = values(vocab * hidden, 0.3);
    let scale = (hidden as f32).sqrt();
    let mut expected = vec![0.0; tokens.len() * hidden];
    for (token_index, token) in tokens.iter().enumerate() {
        for column in 0..hidden {
            expected[token_index * hidden + column] =
                f16::from_bits(table[*token as usize * hidden + column]).to_f32() * scale;
        }
    }
    let actual = kernels
        .embedding_scaled(&tokens, &table, hidden, vocab, scale)
        .unwrap();
    assert_close(&actual, &expected, 1e-3, "embedding");

    let input = values(10, 0.7);
    let weight = values(5, 1.1);
    let expected_weighted = norm_expected(&input, Some(&weight), 2, 5, 1e-6);
    let actual_weighted = kernels.rms_norm(&input, &weight, 2, 5, 1e-6).unwrap();
    assert_close(&actual_weighted, &expected_weighted, 2e-3, "weighted norm");

    let expected_unweighted = norm_expected(&input, None, 2, 5, 1e-6);
    let actual_unweighted = kernels.rms_norm_unweighted(&input, 2, 5, 1e-6).unwrap();
    assert_close(
        &actual_unweighted,
        &expected_unweighted,
        2e-3,
        "unweighted norm",
    );
}

#[test]
fn element_residual_activation_and_softcap_match_cpu() {
    let kernels = ElementKernels::new().expect("Metal element kernels should compile");
    let left = values(16, 0.2);
    let right = values(16, 1.2);
    let expected_residual = left
        .iter()
        .zip(&right)
        .map(|(left, right)| f16::from_bits(*left).to_f32() + f16::from_bits(*right).to_f32())
        .collect::<Vec<_>>();
    let actual_residual = kernels.residual_add(&left, &right).unwrap();
    assert_close(&actual_residual, &expected_residual, 1e-3, "residual");

    let scale = 0.8125;
    let expected_scaled = left
        .iter()
        .map(|value| f16::from_bits(*value).to_f32() * scale)
        .collect::<Vec<_>>();
    let actual_scaled = kernels.layer_scale(&left, scale).unwrap();
    assert_close(&actual_scaled, &expected_scaled, 1e-3, "layer scale");

    let expected_gelu = left
        .iter()
        .map(|value| gelu_expected(f16::from_bits(*value).to_f32()))
        .collect::<Vec<_>>();
    let actual_gelu = kernels.gelu_tanh(&left).unwrap();
    assert_close(&actual_gelu, &expected_gelu, 2e-3, "gelu");

    let expected_gated = left
        .iter()
        .zip(&right)
        .map(|(gate, up)| {
            gelu_expected(f16::from_bits(*gate).to_f32()) * f16::from_bits(*up).to_f32()
        })
        .collect::<Vec<_>>();
    let actual_gated = kernels.gated_gelu(&left, &right).unwrap();
    assert_close(&actual_gated, &expected_gated, 2e-3, "gated gelu");

    let logits = values(16, 2.1);
    let softcap = 30.0;
    let expected_softcap = logits
        .iter()
        .map(|value| (f16::from_bits(*value).to_f32() / softcap).tanh() * softcap)
        .collect::<Vec<_>>();
    let actual_softcap = kernels.softcap(&logits, softcap).unwrap();
    assert_close(&actual_softcap, &expected_softcap, 2e-3, "softcap");
}

#[test]
fn element_gated_silu_matches_cpu() {
    let kernels = ElementKernels::new().expect("Metal element kernels should compile");
    let left = values(97, 0.6);
    let right = values(97, 1.9);

    // Reference is x / (1 + e^-x), i.e. PyTorch's `silu`, evaluated in fp32 and
    // narrowed to fp16 on both sides so the comparison is like-for-like.
    let expected_gated = left
        .iter()
        .zip(right.iter())
        .map(|(gate, up)| {
            let value = f16::from_bits(*gate).to_f32();
            f16::from_f32(value / (1.0 + (-value).exp())).to_f32()
                * f16::from_bits(*up).to_f32()
        })
        .collect::<Vec<_>>();
    let actual_gated = kernels.gated_silu(&left, &right).unwrap();
    assert_close(&actual_gated, &expected_gated, 2e-3, "gated silu");

    let expected_silu = left
        .iter()
        .map(|value| {
            let value = f16::from_bits(*value).to_f32();
            f16::from_f32(value / (1.0 + (-value).exp())).to_f32()
        })
        .collect::<Vec<_>>();
    let actual_silu = kernels.silu(&left).unwrap();
    assert_close(&actual_silu, &expected_silu, 2e-3, "silu");
}

#[test]
fn element_rope_matches_default_and_proportional_paths() {
    let kernels = ElementKernels::new().expect("Metal element kernels should compile");
    let default_input = values(3 * 2 * 256, 0.4);
    let expected_default = rope_expected(&default_input, 3, 2, 256, 0, 10_000.0, 1.0, false);
    let actual_default = kernels
        .rope(RopeRequest {
            input: &default_input,
            tokens: 3,
            heads: 2,
            head_dim: 256,
            offset: 0,
            theta: 10_000.0,
            partial_factor: 1.0,
            proportional: false,
        })
        .unwrap();
    assert_close(&actual_default, &expected_default, 3e-3, "default rope");

    let proportional_input = values(2 * 512, 0.9);
    let expected_proportional = rope_expected(
        &proportional_input,
        2,
        1,
        512,
        100_000,
        1_000_000.0,
        0.25,
        true,
    );
    let actual_proportional = kernels
        .rope(RopeRequest {
            input: &proportional_input,
            tokens: 2,
            heads: 1,
            head_dim: 512,
            offset: 100_000,
            theta: 1_000_000.0,
            partial_factor: 0.25,
            proportional: true,
        })
        .unwrap();
    assert_close(
        &actual_proportional,
        &expected_proportional,
        1e-2,
        "proportional rope",
    );
}

#[test]
fn element_rejects_invalid_inputs() {
    let kernels = ElementKernels::new().expect("Metal element kernels should compile");
    let error = kernels
        .embedding_scaled(&[0], &[0; 3], 2, 2, 1.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("embedding length"));

    let error = kernels
        .embedding_scaled(&[2], &[0; 4], 2, 2, 1.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("outside the vocabulary"));

    let error = kernels
        .rms_norm(&[0; 6], &[0; 5], 2, 5, 1e-6)
        .unwrap_err()
        .to_string();
    assert!(error.contains("RMSNorm input"));

    let error = kernels
        .rope(RopeRequest {
            input: &[0; 3],
            tokens: 1,
            heads: 1,
            head_dim: 3,
            offset: 0,
            theta: 10_000.0,
            partial_factor: 1.0,
            proportional: false,
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("head_dim must be even"));

    let error = kernels.softcap(&[0], 0.0).unwrap_err().to_string();
    assert!(error.contains("cap finite"));
}

/// The threadgroup-per-row norm must match the per-row one that it replaces.
///
/// The dispatch switches to the threadgroup variant below 32 rows, so both paths
/// have to agree or decode silently changes shape depending on batch size. Note
/// the reduction order differs (strided partials plus a tree, versus a serial
/// sum), so this compares with a tolerance rather than exactly.
#[test]
fn threadgroup_norm_matches_per_row_norm() {
    use kaplumba_kernels::elements::ElementKernels;
    let kernels = ElementKernels::new().expect("element kernels");
    for width in [128usize, 256, 512, 5376] {
        for rows in [1usize, 2, 7, 31, 32, 64, 134] {
            let count = rows * width;
            let input: Vec<u16> = (0..count)
                .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.31).to_bits())
                .collect();
            let weight: Vec<u16> = (0..width)
                .map(|i| half::f16::from_f32(1.0 + (i % 5) as f32 * 0.01).to_bits())
                .collect();
            let eps = 1e-6_f32;

            let per_row = kernels
                .rms_norm(&input, &weight, rows, width, eps)
                .expect("per-row norm");
            let tg = kernels
                .rms_norm_threadgroup(&input, &weight, rows, width, eps)
                .expect("threadgroup norm");

            assert_eq!(per_row.len(), count);
            assert_eq!(tg.len(), count);
            let mut worst = 0.0_f32;
            for (index, (a, b)) in per_row.iter().zip(&tg).enumerate() {
                let a = half::f16::from_bits(*a).to_f32();
                let b = half::f16::from_bits(*b).to_f32();
                let scale = a.abs().max(1e-3);
                let diff = (a - b).abs() / scale;
                worst = worst.max(diff);
                assert!(
                    diff < 5e-2,
                    "width={width} rows={rows} index={index} per_row={a} threadgroup={b} rel={diff}"
                );
            }
            println!("norm width={width:>5} rows={rows:>4} worst_rel={worst:.2e}");
        }
    }
}

#[test]
fn gated_silu_is_not_gated_gelu() {
    // spec/dflash-correctness T4 sabotage evidence. The drafter's `hidden_act`
    // is `silu`; the engine already had a `gated_gelu` for the MTP drafter.
    // Feeding GeLU where SiLU belongs is a *silent* quality error -- nothing
    // crashes, the draft just gets worse. So pin the two apart by the number
    // that actually matters: the largest disagreement over the range a real
    // activation row covers, which must dwarf the 2e-3 fp16 tolerance
    // `assert_close` uses.
    let kernels = ElementKernels::new().expect("Metal element kernels should compile");
    let one = vec![f16::from_f32(1.0).to_bits(); 64];
    let gate = (-32..32)
        .map(|value| f16::from_f32(value as f32 * 0.25).to_bits())
        .collect::<Vec<_>>();
    let silu = kernels.gated_silu(&gate, &one).unwrap();
    let gelu = kernels.gated_gelu(&gate, &one).unwrap();
    let worst = (0..gate.len())
        .map(|index| {
            let a = f16::from_bits(silu[index]).to_f32();
            let b = f16::from_bits(gelu[index]).to_f32();
            let input = f16::from_bits(gate[index]).to_f32();
            (input, (a - b).abs())
        })
        .fold((0.0_f32, 0.0_f32), |(at, worst), (input, diff)| {
            if diff > worst { (input, diff) } else { (at, worst) }
        });
    assert!(
        worst.1 > 0.1,
        "silu and gelu-tanh only differ by {} at input {}, so swapping them \
         would not be caught by any test in this file",
        worst.1,
        worst.0
    );
}
