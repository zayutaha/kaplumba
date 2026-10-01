use std::fs;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to read tokenizer file {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse tokenizer config: {0}")]
    Config(#[from] serde_json::Error),
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("tokenizer operation failed: {0}")]
    Operation(String),
    #[error("prompt is empty")]
    EmptyPrompt,
    #[error("prompt contains unsupported media tokens")]
    MediaToken,
    #[error("generated token id is outside the tokenizer vocabulary")]
    InvalidToken,
}

#[derive(Debug, Deserialize, Default)]
struct TokenizerConfig {
    image_token_id: Option<u32>,
    audio_token_id: Option<u32>,
    video_token_id: Option<u32>,
}

pub struct GemmaTokenizer {
    inner: Tokenizer,
    media_ids: Vec<u32>,
}

impl GemmaTokenizer {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, TokenizerError> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            TokenizerError::Load(format!("{}: {error}", tokenizer_path.display()))
        })?;
        let config_path = model_dir.join("tokenizer_config.json");
        let config = if config_path.exists() {
            let bytes = fs::read(&config_path).map_err(|source| TokenizerError::Io {
                path: config_path.display().to_string(),
                source,
            })?;
            serde_json::from_slice::<TokenizerConfig>(&bytes)?
        } else {
            TokenizerConfig::default()
        };
        let mut media_ids = Vec::new();
        for id in [
            config.image_token_id,
            config.audio_token_id,
            config.video_token_id,
        ]
        .into_iter()
        .flatten()
        {
            if !media_ids.contains(&id) {
                media_ids.push(id);
            }
        }
        for token in [
            "<|image|>",
            "<|audio|>",
            "<|video|>",
            "<image>",
            "<audio>",
            "<video>",
        ] {
            if let Some(id) = inner.token_to_id(token)
                && !media_ids.contains(&id)
            {
                media_ids.push(id);
            }
        }
        Ok(Self { inner, media_ids })
    }

    /// Vocabulary id for a literal piece (e.g. a chat-template marker), if the
    /// tokenizer knows it as a token. Used by the chat renderer to resolve
    /// markers without re-tokenizing them, and to fail loudly at startup if a
    /// checkpoint ever renumbers them.
    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.inner.token_to_id(piece)
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>, TokenizerError> {
        if prompt.is_empty() {
            return Err(TokenizerError::EmptyPrompt);
        }
        if [
            "<|image|>",
            "<|audio|>",
            "<|video|>",
            "<image>",
            "<audio>",
            "<video>",
        ]
        .iter()
        .any(|token| prompt.contains(token))
        {
            return Err(TokenizerError::MediaToken);
        }
        let rendered = format!(
            "<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
        let ids = self
            .inner
            .encode(rendered, false)
            .map_err(|error| TokenizerError::Operation(error.to_string()))?
            .get_ids()
            .to_vec();
        if ids.iter().any(|id| self.media_ids.contains(id)) {
            return Err(TokenizerError::MediaToken);
        }
        Ok(ids)
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {        if text.is_empty() {
            return Err(TokenizerError::EmptyPrompt);
        }
        if [
            "<|image|>",
            "<|audio|>",
            "<|video|>",
            "<image>",
            "<audio>",
            "<video>",
        ]
        .iter()
        .any(|token| text.contains(token))
        {
            return Err(TokenizerError::MediaToken);
        }
        let ids = self
            .inner
            .encode(text, false)
            .map_err(|error| TokenizerError::Operation(error.to_string()))?
            .get_ids()
            .to_vec();
        if ids.iter().any(|id| self.media_ids.contains(id)) {
            return Err(TokenizerError::MediaToken);
        }
        Ok(ids)
    }

    pub fn decode_all(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|error| TokenizerError::Operation(error.to_string()))
    }

    pub fn is_media_id(&self, id: u32) -> bool {
        self.media_ids.contains(&id)
    }

    pub fn stream_decode(
        &self,
        ids: &[u32],
        callback: &mut impl FnMut(&str) -> Result<(), TokenizerError>,
    ) -> Result<(), TokenizerError> {
        let mut stream = self.inner.decode_stream(true);
        for id in ids {
            if let Some(chunk) = stream
                .step(*id)
                .map_err(|error| TokenizerError::Operation(error.to_string()))?
            {
                callback(&chunk)?;
            }
        }
        Ok(())
    }
}

pub struct Detokenizer<'a> {
    tokenizer: &'a GemmaTokenizer,
    ids: Vec<u32>,
    emitted: String,
}

impl<'a> Detokenizer<'a> {
    pub fn new(tokenizer: &'a GemmaTokenizer) -> Self {
        Self {
            tokenizer,
            ids: Vec::new(),
            emitted: String::new(),
        }
    }

    pub fn push(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {
        if self.tokenizer.is_media_id(id) {
            return Err(TokenizerError::MediaToken);
        }
        self.ids.push(id);
        let mut stream = self.tokenizer.inner.decode_stream(true);
        let mut decoded = String::new();
        for token in &self.ids {
            if let Some(chunk) = stream
                .step(*token)
                .map_err(|error| TokenizerError::Operation(error.to_string()))?
            {
                decoded.push_str(&chunk);
            }
        }
        let common = self
            .emitted
            .chars()
            .zip(decoded.chars())
            .take_while(|(left, right)| left == right)
            .count();
        if common == self.emitted.chars().count() && decoded.len() >= self.emitted.len() {
            let suffix = decoded[self.emitted.len()..].to_string();
            self.emitted = decoded;
            if suffix.is_empty() {
                Ok(None)
            } else {
                Ok(Some(suffix))
            }
        } else {
            self.emitted = decoded;
            Ok(Some(String::new()))
        }
    }

    pub fn finish(&mut self) -> Result<Option<String>, TokenizerError> {
        let decoded = self.tokenizer.decode_all(&self.ids, true)?;
        if decoded == self.emitted {
            return Ok(None);
        }
        let common = self
            .emitted
            .chars()
            .zip(decoded.chars())
            .take_while(|(left, right)| left == right)
            .count();
        let suffix = decoded.chars().skip(common).collect::<String>();
        self.emitted = decoded;
        if suffix.is_empty() {
            Ok(None)
        } else {
            Ok(Some(suffix))
        }
    }
}
