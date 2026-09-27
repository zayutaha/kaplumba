use std::path::PathBuf;

use kaplumba_engine::sampler::{Sampler, SamplerConfig, SamplerError};
use kaplumba_engine::tokenizer::{Detokenizer, GemmaTokenizer, TokenizerError};

fn tokenizer() -> GemmaTokenizer {
    let path = std::env::var_os("KAPLUMBA_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/zayaantaha/.kaplumba/models/Gxma/Gxma"));
    GemmaTokenizer::from_model_dir(&path).expect("checkpoint tokenizer should load")
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn tokenizer_renders_verified_prompt_ids() {
    let tokenizer = tokenizer();
    let ids = tokenizer.encode_prompt("Hello").unwrap();
    assert_eq!(
        ids,
        vec![
            2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101
        ]
    );
    assert!(tokenizer.is_media_id(258_880));
    assert!(matches!(
        tokenizer.encode_prompt(""),
        Err(TokenizerError::EmptyPrompt)
    ));
    assert!(matches!(
        tokenizer.encode_prompt("look <|image|>"),
        Err(TokenizerError::MediaToken)
    ));
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn detokenizer_emits_valid_unicode_incrementally() {
    let tokenizer = tokenizer();
    let text = "é🙂 café";
    let ids = tokenizer.encode_text(text).unwrap();
    let expected = tokenizer.decode_all(&ids, true).unwrap();
    let mut detokenizer = Detokenizer::new(&tokenizer);
    let mut actual = String::new();
    for id in ids {
        if let Some(chunk) = detokenizer.push(id).unwrap() {
            actual.push_str(&chunk);
        }
    }
    if let Some(chunk) = detokenizer.finish().unwrap() {
        actual.push_str(&chunk);
    }
    assert_eq!(actual, expected);
    assert!(actual.contains('é'));
    assert!(actual.contains('🙂'));
    assert!(!actual.contains('\u{fffd}'));
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn tokenizer_suppresses_special_tokens_only_when_requested() {
    let tokenizer = tokenizer();
    assert_eq!(tokenizer.decode_all(&[1], true).unwrap(), "");
    assert!(!tokenizer.decode_all(&[1], false).unwrap().is_empty());
}

#[test]
fn sampler_is_greedy_and_seed_reproducible() {
    let mut greedy = Sampler::new(
        SamplerConfig {
            temperature: 0.0,
            top_k: 64,
            top_p: 0.95,
        },
        1,
    )
    .unwrap();
    assert_eq!(greedy.sample(&[1.0, 3.0, 3.0, 2.0]).unwrap(), 1);

    let config = SamplerConfig {
        temperature: 1.0,
        top_k: 2,
        top_p: 0.9,
    };
    let mut first = Sampler::new(config, 42).unwrap();
    let mut second = Sampler::new(config, 42).unwrap();
    let logits = [0.2, 1.4, 0.7, 0.1, 0.0];
    let first_ids = (0..8)
        .map(|_| first.sample(&logits).unwrap())
        .collect::<Vec<_>>();
    let second_ids = (0..8)
        .map(|_| second.sample(&logits).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(first_ids, second_ids);
}

#[test]
fn sampler_rejects_invalid_configuration_and_logits() {
    assert!(matches!(
        Sampler::new(
            SamplerConfig {
                temperature: -1.0,
                top_k: 1,
                top_p: 0.9
            },
            1
        ),
        Err(SamplerError::InvalidTemperature)
    ));
    assert!(matches!(
        Sampler::new(
            SamplerConfig {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.9
            },
            1
        ),
        Err(SamplerError::InvalidTopK)
    ));
    let mut sampler = Sampler::new(SamplerConfig::default(), 1).unwrap();
    assert_eq!(sampler.sample(&[]), Err(SamplerError::InvalidLogits));
    assert_eq!(
        sampler.sample(&[0.0, f32::NAN]),
        Err(SamplerError::InvalidLogits)
    );
}
