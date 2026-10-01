use thiserror::Error;

use crate::generate::DEFAULT_CONTEXT_LIMIT;
use crate::model::{ModelError, TextModel};
use crate::sampler::{Sampler, SamplerConfig, SamplerError};
use crate::tokenizer::{Detokenizer, GemmaTokenizer, TokenizerError};

#[derive(Debug, Error)]
pub enum ChatError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
    #[error("message must not be empty")]
    EmptyMessage,
    #[error(
        "context limit reached: {used} tokens committed, this turn needs {need}, limit is {limit}; /reset to start over"
    )]
    ContextOverflow {
        used: usize,
        need: usize,
        limit: usize,
    },
}

/// The chat-template markers and their vocabulary ids, resolved through the
/// tokenizer at session start and asserted against the checkpoint's known
/// values. See `~/.kaplumba/models/Gxma-repacked/chat_template.jinja`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatMarkers {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub turn_open: u32,
    pub turn_close: u32,
    pub channel_open: u32,
    pub channel_close: u32,
}

/// Resolve the seven markers through `tokenizer` and assert they are the
/// checkpoint's known ids. A checkpoint swap that renumbers them would
/// otherwise render a subtly wrong template with no error anywhere.
pub fn resolve_markers(tokenizer: &GemmaTokenizer) -> Result<ChatMarkers, ChatError> {
    let lookup = |piece: &str| {
        tokenizer
            .token_id(piece)
            .ok_or_else(|| ChatError::Tokenizer(TokenizerError::Operation(format!(
                "chat marker {piece} is not in the tokenizer vocabulary"
            ))))
    };
    let markers = ChatMarkers {
        bos: lookup("<bos>")?,
        eos: lookup("<eos>")?,
        pad: lookup("<pad>")?,
        turn_open: lookup("<|turn>")?,
        turn_close: lookup("<turn|>")?,
        channel_open: lookup("<|channel>")?,
        channel_close: lookup("<channel|>")?,
    };
    let expected = ChatMarkers {
        bos: 2,
        eos: 1,
        pad: 0,
        turn_open: 105,
        turn_close: 106,
        channel_open: 100,
        channel_close: 101,
    };
    if markers != expected {
        return Err(ChatError::Tokenizer(TokenizerError::Operation(format!(
            "chat marker ids {markers:?} do not match the checkpoint's known ids {expected:?}"
        ))));
    }
    Ok(markers)
}

/// Render turn 1 by delegating to `GemmaTokenizer::encode_prompt`, so the
/// chat's first turn is byte-identical to `--prompt` by construction (AC2).
pub fn render_first_turn(tokenizer: &GemmaTokenizer, text: &str) -> Result<Vec<u32>, ChatError> {
    if text.trim().is_empty() {
        return Err(ChatError::EmptyMessage);
    }
    Ok(tokenizer.encode_prompt(text)?)
}

/// Render turn _n_ > 1: the terminator the model produced but never fed to the
/// cache, then the new user turn and a fresh model header. Tokenized in one
/// `encode_text` call; both ends are special-token boundaries, so concatenating
/// this onto the committed prefix is exact.
pub fn render_next_turn(tokenizer: &GemmaTokenizer, text: &str) -> Result<Vec<u32>, ChatError> {
    if text.trim().is_empty() {
        return Err(ChatError::EmptyMessage);
    }
    let delta = format!(
        "<turn|>\n<|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    );
    Ok(tokenizer.encode_text(&delta)?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub text: String,
}

#[derive(Debug, Clone, Copy)]
pub struct TurnStats {
    pub prefill_tokens: usize,
    pub generated_tokens: usize,
    pub decode_tok_s: f64,
}

fn argmax_logits(logits: &[u16]) -> u32 {
    let mut best = 0;
    let mut best_val = half::f16::from_bits(logits[0]).to_f32();
    for (index, bits) in logits.iter().enumerate().skip(1) {
        let val = half::f16::from_bits(*bits).to_f32();
        if val > best_val {
            best_val = val;
            best = index;
        }
    }
    best as u32
}

/// One conversation: owns the turn counter and the transcript, borrows the
/// model and tokenizer. The GPU KV cache persists across turns — that is the
/// whole feature — so `turn` only ever prefills the new turn's tokens at their
/// absolute positions. `reset` is the only thing that clears the cache.
pub struct ChatSession<'a> {
    model: &'a mut TextModel,
    tokenizer: &'a GemmaTokenizer,
    sampler: Sampler,
    greedy: bool,
    max_new_tokens: usize,
    eos_ids: Vec<u32>,
    context_limit: usize,
    turns: usize,
    messages: Vec<ChatMessage>,
    last_stats: Option<TurnStats>,
}

impl<'a> ChatSession<'a> {
    pub fn new(
        model: &'a mut TextModel,
        tokenizer: &'a GemmaTokenizer,
        sampler: SamplerConfig,
        seed: u64,
        max_new_tokens: usize,
        eos_ids: Vec<u32>,
        context_limit: usize,
    ) -> Result<Self, ChatError> {
        // Fail loudly at startup on a renumbered checkpoint, before any turn.
        let _ = resolve_markers(tokenizer)?;
        let sampler = Sampler::new(sampler, seed)?;
        model.reset_gpu_caches();
        Ok(Self {
            model,
            tokenizer,
            sampler,
            greedy: false,
            max_new_tokens,
            eos_ids,
            context_limit,
            turns: 0,
            messages: Vec::new(),
            last_stats: None,
        })
    }

    /// Same as `new`, but temperature 0 selects greedy argmax instead of the
    /// sampler, matching `generate_gpu`'s dispatch.
    pub fn new_greedy(
        model: &'a mut TextModel,
        tokenizer: &'a GemmaTokenizer,
        sampler: SamplerConfig,
        seed: u64,
        max_new_tokens: usize,
        eos_ids: Vec<u32>,
        context_limit: usize,
    ) -> Result<Self, ChatError> {
        let mut session = Self::new(
            model,
            tokenizer,
            sampler,
            seed,
            max_new_tokens,
            eos_ids,
            context_limit,
        )?;
        session.greedy = true;
        Ok(session)
    }

    pub fn turns(&self) -> usize {
        self.turns
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn context_used(&self) -> usize {
        self.model.gpu_next_position()
    }

    pub fn context_limit(&self) -> usize {
        self.context_limit
    }

    pub fn last_stats(&self) -> Option<TurnStats> {
        self.last_stats
    }

    pub fn reset(&mut self) {
        self.model.reset_gpu_caches();
        self.turns = 0;
        self.messages.clear();
        self.last_stats = None;
    }

    /// Run one turn: prefill only the new tokens, decode until EOS or the
    /// token budget, stream text through `on_text`.
    ///
    /// The stopping token is **not** forwarded to the cache, matching
    /// `generate_gpu`: after a turn,
    /// `context_used() == base + prefill + generated - 1` when the turn ended
    /// on EOS. Forwarding the terminator would shift every later position.
    pub fn turn<F>(&mut self, text: &str, mut on_text: F) -> Result<String, ChatError>
    where
        F: FnMut(&str) -> Result<(), String>,
    {
        let ids = if self.turns == 0 {
            render_first_turn(self.tokenizer, text)?
        } else {
            render_next_turn(self.tokenizer, text)?
        };
        let base = self.model.gpu_next_position();
        let need = ids.len();
        if base + need + self.max_new_tokens > self.context_limit {
            return Err(ChatError::ContextOverflow {
                used: base,
                need: need + self.max_new_tokens,
                limit: self.context_limit,
            });
        }
        let positions: Vec<u32> = (0..need as u32).map(|i| base as u32 + i).collect();
        let mut logits = Vec::new();
        self.model
            .forward_buffer_into_vec(&ids, &positions, &mut logits)?;
        let mut detokenizer = Detokenizer::new(self.tokenizer);
        let mut reply = String::new();
        let mut generated: Vec<u32> = Vec::new();
        let decode_started = std::time::Instant::now();
        for _ in 0..self.max_new_tokens {
            let token = if self.greedy {
                argmax_logits(&logits)
            } else {
                let last: Vec<f32> = logits
                    .iter()
                    .map(|bits| half::f16::from_bits(*bits).to_f32())
                    .collect();
                self.sampler.sample(&last)?
            };
            if self.eos_ids.contains(&token) {
                break;
            }
            generated.push(token);
            if let Some(chunk) = detokenizer.push(token)? {
                if !chunk.is_empty() {
                    reply.push_str(&chunk);
                    on_text(&chunk).map_err(ChatError::from_output)?;
                }
            }
            // A failed write is surfaced through `from_output`, not as a model
            // failure.
            let position = (base + need + generated.len() - 1) as u32;
            self.model
                .forward_buffer_into_vec(&[token], &[position], &mut logits)?;
        }
        let decode_s = decode_started.elapsed().as_secs_f64();
        if let Some(chunk) = detokenizer.finish()? {
            if !chunk.is_empty() {
                reply.push_str(&chunk);
                on_text(&chunk).map_err(ChatError::from_output)?;
            }
        }
        self.messages.push(ChatMessage {
            role: ChatRole::User,
            text: text.to_string(),
        });
        self.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            text: reply.clone(),
        });
        self.turns += 1;
        self.last_stats = Some(TurnStats {
            prefill_tokens: need,
            generated_tokens: generated.len(),
            decode_tok_s: if decode_s > 0.0 {
                generated.len() as f64 / decode_s
            } else {
                0.0
            },
        });
        Ok(reply)
    }
}

impl ChatError {
    /// The output closure can only fail with a string (usually a flush
    /// error), which has no structured variant here; surface it as a
    /// tokenizer-level failure of the turn, not a model failure.
    fn from_output(message: String) -> Self {
        ChatError::Tokenizer(TokenizerError::Operation(message))
    }
}

/// Re-export for callers that build a session with default limits.
pub fn default_context_limit() -> usize {
    DEFAULT_CONTEXT_LIMIT
}
