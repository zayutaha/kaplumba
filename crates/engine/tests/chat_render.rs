use std::path::PathBuf;

use kaplumba_engine::chat::{render_first_turn, render_next_turn, resolve_markers};
use kaplumba_engine::tokenizer::GemmaTokenizer;

fn tokenizer() -> GemmaTokenizer {
    let path = std::env::var_os("KAPLUMBA_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/zayaantaha/.kaplumba/models/Gxma/Gxma"));
    GemmaTokenizer::from_model_dir(&path).expect("checkpoint tokenizer should load")
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn chat_markers_resolve_to_checkpoint_ids() {
    let tokenizer = tokenizer();
    let markers = resolve_markers(&tokenizer).expect("markers should resolve");
    assert_eq!(
        (
            markers.bos,
            markers.eos,
            markers.pad,
            markers.turn_open,
            markers.turn_close,
            markers.channel_open,
            markers.channel_close,
        ),
        (2, 1, 0, 105, 106, 100, 101)
    );
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn first_turn_matches_encode_prompt() {
    // AC2: the chat's first turn is `--prompt` by construction. If this drifts,
    // the renderer changed and the parity claim is void.
    let tokenizer = tokenizer();
    for text in ["Hello", "The capital of France is", "count to three: 1,"] {
        assert_eq!(
            render_first_turn(&tokenizer, text).unwrap(),
            tokenizer.encode_prompt(text).unwrap(),
            "first-turn render diverged for {text:?}"
        );
    }
    assert!(render_first_turn(&tokenizer, "   ").is_err());
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn next_turn_is_exact_and_concatenates() {
    let tokenizer = tokenizer();
    let second = render_next_turn(&tokenizer, "Hello").unwrap();
    // The leading id closes the assistant's previous turn; 106 appears twice.
    assert_eq!(second[0], 106);
    assert_eq!(second.iter().filter(|id| **id == 106).count(), 2);
    assert!(render_next_turn(&tokenizer, "  ").is_err());

    // Concatenation is exact: both render boundaries are special-token
    // boundaries, so tokenizing the whole two-turn string must equal the
    // concatenated per-turn renders.
    let first = render_first_turn(&tokenizer, "Hi").unwrap();
    let full = "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>\
        <turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    let mut joined = first;
    joined.extend_from_slice(&second);
    assert_eq!(tokenizer.encode_text(full).unwrap(), joined);
}
