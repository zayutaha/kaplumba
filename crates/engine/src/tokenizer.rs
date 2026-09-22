//! In-engine tokenizer (T1.2): HF `tokenizers` crate loading `tokenizer.json`
//! from the model directory. Self-contained for CLI bench (no HTTP).

use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer.json not found in {0}")]
    Missing(PathBuf),
    #[error("failed to load tokenizer from {path}: {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("encode failed: {0}")]
    Encode(String),
    #[error("decode failed: {0}")]
    Decode(String),
}

/// Gemma 4 tokenizer wrapper.
pub struct GemmaTokenizer {
    inner: Tokenizer,
    path: PathBuf,
}

impl GemmaTokenizer {
    /// Load from `{dir}/tokenizer.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = dir.as_ref().join("tokenizer.json");
        if !path.is_file() {
            return Err(TokenizerError::Missing(path));
        }
        let inner = Tokenizer::from_file(&path).map_err(|source| TokenizerError::Load {
            path: path.clone(),
            source,
        })?;
        Ok(Self { inner, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Encode raw text (no chat template applied — caller owns template).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode with special tokens added (BOS etc. if template/post-processor requires).
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids to text (skip special tokens by default).
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Decode keeping special tokens visible.
    pub fn decode_with_special(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, false)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Vocabulary size (ids space).
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Id of a special token string, if present.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Ids expected by Gemma 4 text path (from tokenizer_config / added_tokens).
    pub fn gemma_special_ids(&self) -> GemmaSpecialIds {
        GemmaSpecialIds {
            pad: self.token_id("<pad>").unwrap_or(0),
            eos: self.token_id("<eos>").unwrap_or(1),
            bos: self.token_id("<bos>").unwrap_or(2),
            unk: self.token_id("<unk>").unwrap_or(3),
            turn_close: self.token_id("<turn|>").unwrap_or(106),
            turn_open: self.token_id("<|turn>").unwrap_or(105),
        }
    }
}

/// Special ids used by generation/stop logic (verified against tokenizer.json).
#[derive(Debug, Clone, Copy)]
pub struct GemmaSpecialIds {
    pub pad: u32,
    pub eos: u32,
    pub bos: u32,
    pub unk: u32,
    /// `<turn|>` — primary stop / turn boundary (config eos includes 106).
    pub turn_close: u32,
    pub turn_open: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        let d = std::env::var("KAPLUMBA_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let h = std::env::var("HOME").unwrap_or_else(|_| ".".into());
                PathBuf::from(format!("{h}/.kaplumba/models/Gxma/Gxma"))
            });
        d.join("tokenizer.json").is_file().then_some(d)
    }

    #[test]
    fn missing_tokenizer_errors() {
        let err = match GemmaTokenizer::from_dir("/nonexistent/model/dir") {
            Ok(_) => panic!("expected missing tokenizer error"),
            Err(e) => e,
        };
        assert!(matches!(err, TokenizerError::Missing(_)));
    }

    #[test]
    fn real_tokenizer_roundtrip_and_specials() {
        let Some(dir) = model_dir() else {
            eprintln!("skip: no tokenizer.json");
            return;
        };
        let tok = GemmaTokenizer::from_dir(&dir).expect("load tokenizer");
        assert_eq!(tok.vocab_size(), 262144, "vocab");

        let sp = tok.gemma_special_ids();
        assert_eq!(sp.pad, 0);
        assert_eq!(sp.eos, 1);
        assert_eq!(sp.bos, 2);
        assert_eq!(sp.unk, 3);
        assert_eq!(sp.turn_open, 105);
        assert_eq!(sp.turn_close, 106);

        let text = "The quick brown fox jumps over the lazy dog.";
        let ids = tok.encode(text).expect("encode");
        assert!(!ids.is_empty());
        let back = tok.decode(&ids).expect("decode");
        // HF decode may normalize whitespace slightly; require re-encode stability.
        let ids2 = tok.encode(&back).expect("re-encode");
        assert_eq!(ids, ids2, "roundtrip ids stable");

        // Special tokens as standalone strings stay atomic.
        let turn = tok.encode_with_special("<turn|>").expect("enc turn");
        assert_eq!(turn, vec![106], "got {turn:?}");
        let bos = tok.encode_with_special("<bos>").expect("enc bos");
        assert_eq!(bos, vec![2], "got {bos:?}");

        // token_id lookups
        assert_eq!(tok.token_id("<turn|>"), Some(106));
        assert_eq!(tok.token_id("<bos>"), Some(2));
    }
}
