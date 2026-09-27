use std::path::PathBuf;

use half::f16;
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_engine::generate::{GenerationConfig, GenerationError, generate, run_benchmark};
use kaplumba_engine::metrics::BenchmarkPlan;
use kaplumba_engine::model::{
    DecoderLayer, DecoderLayerWeights, LayerConfig, QuantizedLinear, RopeMode, TextModel,
};
use kaplumba_engine::tokenizer::GemmaTokenizer;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};

fn half(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

fn linear(in_features: usize, out_features: usize, bits: u8, seed: u32) -> QuantizedLinear {
    let mask = (1_u32 << bits) - 1;
    let mut packed = Vec::new();
    for tile in 0..(in_features / 16) * (out_features / 16) {
        let mut states = vec![0_u16; 256];
        let mut value = seed + tile as u32 * 7919;
        for state in &mut states {
            value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *state = (value & mask) as u16;
        }
        packed.extend(pack_integer_states(&states, bits).unwrap().0);
    }
    QuantizedLinear::new(
        in_features,
        out_features,
        Exl3Rate::Integer(bits),
        Exl3Codebook::Mcg,
        packed,
        vec![half(0.01); in_features],
        vec![half(0.01); out_features],
    )
    .unwrap()
}

fn model_dir() -> PathBuf {
    std::env::var_os("KAPLUMBA_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/zayaantaha/.kaplumba/models/Gxma/Gxma"))
}

fn tokenizer() -> GemmaTokenizer {
    GemmaTokenizer::from_model_dir(&model_dir()).expect("checkpoint tokenizer should load")
}

fn model() -> TextModel {
    let config = LayerConfig {
        hidden_size: 128,
        intermediate_size: 256,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 128,
        max_sequence_len: 64,
        full_attention: true,
        sliding_window: 0,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_mode: RopeMode::Default,
        rope_partial_factor: 1.0,
    };
    let layer_weights = DecoderLayerWeights {
        q_proj: linear(128, 128, 3, 11),
        k_proj: linear(128, 128, 3, 12),
        v_proj: None,
        o_proj: linear(128, 128, 3, 13),
        gate_proj: linear(128, 256, 3, 14),
        up_proj: linear(128, 256, 3, 15),
        down_proj: linear(256, 128, 3, 16),
        q_norm: vec![half(1.0); 128],
        k_norm: vec![half(1.0); 128],
        input_norm: vec![half(1.0); 128],
        post_attention_norm: vec![half(1.0); 128],
        pre_ffn_norm: vec![half(1.0); 128],
        post_ffn_norm: vec![half(1.0); 128],
        layer_scalar: 0.9,
    };
    let layer = DecoderLayer::new(config, layer_weights).unwrap();
    let embedding = (0..65_536 * 128)
        .map(|index| half(((index as f32 + 0.3) * 0.011).sin()))
        .collect();
    let head = linear(128, 65_536, 6, 17);
    TextModel::new(embedding, vec![layer], vec![half(1.0); 128], head, 30.0).unwrap()
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn generation_streams_detokenized_text_and_reports_ids() {
    let tokenizer = tokenizer();
    let prompt_ids = tokenizer.encode_prompt("<bos>").unwrap();
    assert!(prompt_ids.iter().all(|id| *id < 65_536));
    let mut model = model();
    let config = GenerationConfig {
        max_new_tokens: 2,
        eos_ids: Vec::new(),
        ..GenerationConfig::default()
    };
    let mut chunks = String::new();
    let result = generate(&mut model, &tokenizer, "<bos>", &config, |chunk| {
        chunks.push_str(chunk);
        Ok(())
    })
    .unwrap();
    assert_eq!(result.prompt_tokens, prompt_ids.len());
    assert_eq!(result.generated_tokens, result.generated_ids.len());
    assert_eq!(chunks, result.text);
    assert!(result.text.is_char_boundary(0));
    assert!(!result.stopped_on_eos);
}

#[test]
fn benchmark_plan_runs_on_synthetic_model() {
    let mut model = model();
    let result = run_benchmark(
        &mut model,
        &[1, 2, 3],
        BenchmarkPlan {
            prompt_tokens: 3,
            warmup_steps: 1,
            measured_tokens: 2,
        },
    )
    .unwrap();
    assert_eq!(result.measured_tokens, 2);
    assert_eq!(result.warmup_steps, 1);
    assert!(result.prefill_tokens_per_second.is_finite());
    assert!(result.decode_tokens_per_second.is_finite());
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and runs the full resident model"]
fn real_checkpoint_generates_unicode_text() {
    let mut model = TextModel::load_from_dir(&model_dir()).unwrap();
    let tokenizer = tokenizer();
    let config = GenerationConfig {
        max_new_tokens: 1,
        eos_ids: Vec::new(),
        sampler: kaplumba_engine::sampler::SamplerConfig {
            temperature: 0.0,
            top_k: 64,
            top_p: 0.95,
        },
        ..GenerationConfig::default()
    };
    let mut output = String::new();
    let result = generate(&mut model, &tokenizer, "Hello", &config, |chunk| {
        output.push_str(chunk);
        Ok(())
    })
    .unwrap();
    println!("real output: {output:?}");
    assert_eq!(output, result.text);
    assert!(
        result
            .text
            .chars()
            .all(|character| !character.is_control() || character == '\n')
    );
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn generation_rejects_zero_tokens_and_context_overflow() {
    let tokenizer = tokenizer();
    let mut model = model();
    let config = GenerationConfig {
        max_new_tokens: 0,
        ..GenerationConfig::default()
    };
    assert!(matches!(
        generate(&mut model, &tokenizer, "a", &config, |_| Ok(())),
        Err(GenerationError::ZeroNewTokens)
    ));
    let config = GenerationConfig {
        max_new_tokens: 64,
        context_limit: 16,
        ..GenerationConfig::default()
    };
    assert!(matches!(
        generate(&mut model, &tokenizer, "a", &config, |_| Ok(())),
        Err(GenerationError::ContextExceeded { .. })
    ));
}
