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

/// Pure context-limit arithmetic, so the 8192 boundary is unit-testable
/// without prefilling 8192 tokens on a GPU. Exactly at the limit fits; one
/// over does not.
pub fn fits_in_context(
    used: usize,
    need: usize,
    max_new_tokens: usize,
    limit: usize,
) -> Result<(), ChatError> {
    if used + need + max_new_tokens > limit {
        return Err(ChatError::ContextOverflow {
            used,
            need: need + max_new_tokens,
            limit,
        });
    }
    Ok(())
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

/// Cumulative decode work over a whole session. See [`ChatSession::totals`].
#[derive(Debug, Clone, Copy)]
pub struct SessionTotals {
    pub turns: usize,
    pub prefill_tokens: usize,
    pub generated_tokens: usize,
    pub decode_s: f64,
    pub decode_tok_s: f64,
}

impl SessionTotals {
    /// Derive the rate from accumulated counts.
    ///
    /// A free function so the arithmetic is testable without a loaded model --
    /// building a real `ChatSession` needs ~12 GiB of resident weights, which no
    /// unit test should require.
    ///
    /// The rate is `generated / decode_s` over the TOTALS, deliberately not a
    /// mean of per-turn rates: a mean would weight a 3-token reply the same as a
    /// 400-token one, which is not what "average tok/s" means to anyone reading
    /// it, and it is not comparable with the single-generation `decode_tok_s`
    /// that `--metrics` prints.
    pub fn from_accumulated(
        turns: usize,
        prefill_tokens: usize,
        generated_tokens: usize,
        decode_s: f64,
    ) -> Self {
        Self {
            turns,
            prefill_tokens,
            generated_tokens,
            decode_s,
            decode_tok_s: if decode_s > 0.0 {
                generated_tokens as f64 / decode_s
            } else {
                0.0
            },
        }
    }
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
    /// Cumulative decode work since the last [`Self::reset`], for the
    /// end-of-session summary.
    ///
    /// Accumulated rather than recomputed from `last_stats`: an average of
    /// per-turn rates is wrong whenever the turns differ in length, which they
    /// always do (the first turn pays a cold prefill, short replies finish in
    /// a few tokens). Totals are summed and divided once.
    total_prefill_tokens: usize,
    total_generated_tokens: usize,
    total_decode_s: f64,
    /// The user text for the turn in flight.
    ///
    /// Held separately because `record_turn` is shared by both decode paths and
    /// the prompt is available in `turn` long before the reply exists. Passing it
    /// as an argument would be tidier, but the two call sites are in different
    /// branches of the same function and this keeps `record_turn`'s signature to
    /// the numbers it actually books.
    pending_user_text: String,
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
            total_prefill_tokens: 0,
            total_generated_tokens: 0,
            total_decode_s: 0.0,
            pending_user_text: String::new(),
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

    /// Whole-session totals, for the end-of-chat summary.
    ///
    /// `decode_tok_s` is `total_generated / total_decode_s`, i.e. a
    /// token-weighted rate over every turn, not a mean of per-turn rates.
    pub fn totals(&self) -> SessionTotals {
        SessionTotals::from_accumulated(
            self.turns,
            self.total_prefill_tokens,
            self.total_generated_tokens,
            self.total_decode_s,
        )
    }

    pub fn reset(&mut self) {
        self.model.reset_gpu_caches();
        self.turns = 0;
        self.messages.clear();
        self.last_stats = None;
        // The totals are session-scoped, so `/reset` clears them too. Leaving
        // them would make the end-of-chat figure describe a conversation the
        // user explicitly discarded.
        self.total_prefill_tokens = 0;
        self.total_generated_tokens = 0;
        self.total_decode_s = 0.0;
        self.pending_user_text.clear();
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
        self.pending_user_text = text.to_string();
        let ids = if self.turns == 0 {
            render_first_turn(self.tokenizer, text)?
        } else {
            render_next_turn(self.tokenizer, text)?
        };
        let base = self.model.gpu_next_position();
        let need = ids.len();
        fits_in_context(base, need, self.max_new_tokens, self.context_limit)?;
        let positions: Vec<u32> = (0..need as u32).map(|i| base as u32 + i).collect();
        let mut logits = Vec::new();
        self.model
            .forward_buffer_into_vec(&ids, &positions, &mut logits)?;
        let mut detokenizer = Detokenizer::new(self.tokenizer);
        let mut reply = String::new();
        let mut generated: Vec<u32> = Vec::new();
        let decode_started = std::time::Instant::now();

        // The loop below is the PIPELINED decode, and it exists because the
        // obvious loop is ~3.5 tok/s slower.
        //
        // The obvious loop calls `forward_buffer_into_vec` per token, and that
        // function ends in `commit_and_wait`: a full host/GPU barrier after
        // every single token. Measured consequence, on this box, at 23-107
        // context tokens (where attention is negligible, so the difference is
        // not the model doing less work): **14.97 tok/s in chat against 15.88 for
        // the pipelined loop on the same binary**, interleaved.
        //
        // `generate_gpu`'s pipelined loop (`generate.rs`, `pipelined` branch)
        // avoids the barrier by overlapping step N+1's *encode and submit* with
        // step N's *GPU execution*: it encodes the next step, commits without
        // waiting, and only then waits on the previous encoder. The GPU is never
        // idle waiting for the host, and that is where the difference lives.
        //
        // The same loop also avoids reading the logits back to the host at all
        // in greedy mode: `forward_step_pipelined` stages the argmax on the GPU
        // (`next_token_out` / `host_token_out`), so the host reads one u32 via
        // `read_host_token` instead of 262,144 fp16 values. The naive version
        // copied and scanned the full vocabulary per token, on the CPU, inside
        // the timed region.
        //
        // `KAPLUMBA_NO_PIPELINE` restores the barrier-per-token loop. It is the
        // control for exactly this attribution, which is why it exists rather
        // than the slow path being deleted.
        // T5/T6: the pipelined loop now serves temperature > 0 when the device
        // sampler can stage the top_k this model + config need. Greedy is
        // byte-identical either way; the barrier-per-token loop stays as the
        // A/B control (`KAPLUMBA_NO_PIPELINE`) and the fallback for unstagable
        // top_k.
        let device_params = if self.greedy {
            None
        } else {
            crate::sampler::SamplerParams::from_config(
                self.sampler.config(),
                self.model.vocab_size(),
            )
            .ok()
        };
        let pipelined = (self.greedy || device_params.is_some())
            && std::env::var_os("KAPLUMBA_NO_PIPELINE").is_none();

        if pipelined {
            let queue = self.model.forward_queue().clone();

            // The loop below is deliberately the SAME shape as `generate.rs`'s
            // pipelined branch, because the first version of it written here was
            // not, and produced wrong tokens.
            //
            // **What was wrong:** it read `read_host_token(step - 1)` at the top
            // of iteration `step`, while the encoder that writes that slot was
            // still in flight (the wait was deferred to `encoders.len() > 2`, one
            // iteration later). `read_host_token` is a raw `contents()` read of
            // GPU-written shared memory with no synchronisation of its own, so
            // the host read whatever was there before — the previous turn's value,
            // or zero on the first. Symptom: `Count from 1 to 30: 1, 2, 3,`
            // answered `4` pipelined and `4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`
            // not pipelined, reproducibly, and the pipelined answer was the wrong
            // one. The barrier-per-token loop had no such window, which is why
            // the bug was invisible until both paths were run side by side.
            //
            // **The invariant:** never read a GPU-staged token before waiting on
            // the encoder that staged it. `generate.rs` gets this right by
            // submitting step N+1, then waiting on step N, and only then reading
            // step N's slot. That ordering is what "two in flight" means — not
            // three, and not "wait when the vec gets long".

            // The first generated token comes from the prefill's own argmax,
            // read back with the logits above. It is the only token the host
            // supplies to the model; every later one is staged on the GPU.
            let mut token = if self.greedy {
                argmax_logits(&logits)
            } else {
                let last: Vec<f32> = logits
                    .iter()
                    .map(|bits| half::f16::from_bits(*bits).to_f32())
                    .collect();
                self.sampler.sample(&last)?
            };
            if device_params.is_some() {
                // Seed the device sampler from the host sampler's position in
                // the same sequence, AFTER the host drew the first token.
                let state = self.sampler.state();
                self.model.seed_device_sampler(state);
            }
            // The encoder for step N-1, submitted but not yet waited on. Exactly two
            // steps are in flight at any moment: `prev`, and the one this iteration
            // is about to build.
            let mut prev: Option<kaplumba_kernels::encoder::MetalEncoder> = None;
            let mut step: usize = 0;
            let mut t_encode = std::time::Duration::ZERO;
            let mut t_wait = std::time::Duration::ZERO;
            let mut t_detok = std::time::Duration::ZERO;

            loop {
                let _t0 = std::time::Instant::now();
                // Order matters and was wrong until T2: encode step N+1, commit
                // it, and ONLY THEN wait on step N. This loop used to wait on
                // `prev` at the top of the iteration, before building the next
                // encoder, so the GPU drained to idle and sat there for the whole
                // of the host's `forward_step_pipelined` encode every single
                // token. `generate.rs` has always had the right order
                // (`generate.rs:1031` commits `next_enc`, `:1033` then waits on
                // `in_flight_enc`).
                //
                // The invariant that forced the bad order -- "never read a
                // GPU-staged token before waiting on the encoder that staged it"
                // -- is still honoured: `read_host_token` only happens after the
                // `wait()`, it is simply not the first thing in the iteration.
                let mut encoder = None;
                if step + 1 < self.max_new_tokens {
                    let position = (base + need + step) as u32;
                    let mut enc =
                        kaplumba_kernels::encoder::MetalEncoder::new(&queue).map_err(ModelError::from)?;
                    enc.set_max_dispatches(
                        kaplumba_kernels::encoder::decode_max_dispatches_per_buffer(),
                    );
                    self.model
                        .forward_step_pipelined(
                            &mut enc,
                            position,
                            if step == 0 { Some(token) } else { None },
                            step % 2,
                            step as u32,
                            if self.greedy {
                                crate::model::HeadPick::Argmax
                            } else {
                                match device_params {
                                    Some(params) => crate::model::HeadPick::Sample(params),
                                    None => crate::model::HeadPick::Argmax,
                                }
                            },
                        )
                        .map_err(ModelError::from)?;
                    enc.commit_terminal().map_err(ModelError::from)?;
                    encoder = Some(enc);
                }
                t_encode += _t0.elapsed();

                let _t1 = std::time::Instant::now();
                if step > 0 {
                    // Step N-1 is now committed and this step's encode is
                    // already submitted, so the wait below overlaps step N+1's
                    // GPU work with step N's drain instead of idling.
                    let previous = prev
                        .take()
                        .expect("step > 0 implies a previous encoder is in flight");
                    previous.wait();
                    token = self.model.read_host_token(step - 1);
                }
                t_wait += _t1.elapsed();
                if self.eos_ids.contains(&token) {
                    if let Some(enc) = encoder {
                        enc.wait();
                    }
                    break;
                }
                let _t2 = std::time::Instant::now();
                generated.push(token);
                if let Some(chunk) = detokenizer.push(token)? {
                    if !chunk.is_empty() {
                        reply.push_str(&chunk);
                        on_text(&chunk).map_err(ChatError::from_output)?;
                    }
                }
                t_detok += _t2.elapsed();
                if generated.len() >= self.max_new_tokens {
                    if let Some(enc) = encoder {
                        enc.wait();
                    }
                    break;
                }
                prev = encoder;
                step += 1;
            }

            // Drain whatever is still in flight before returning, or the KV cache
            // advances past what the next turn's `base` believes and every later
            // position is wrong.
            if let Some(pending) = prev.take() {
                pending.wait();
            }
            if std::env::var_os("KAPLUMBA_CHAT_BREAKDOWN").is_some() {
                eprintln!(
                    "CHAT-BREAKDOWN steps={} encode={:.1}ms wait={:.1}ms detok={:.1}ms \
                     per-step: encode={:.2}ms wait={:.2}ms detok={:.2}ms",
                    step,
                    t_encode.as_secs_f64() * 1e3,
                    t_wait.as_secs_f64() * 1e3,
                    t_detok.as_secs_f64() * 1e3,
                    t_encode.as_secs_f64() * 1e3 / step.max(1) as f64,
                    t_wait.as_secs_f64() * 1e3 / step.max(1) as f64,
                    t_detok.as_secs_f64() * 1e3 / step.max(1) as f64,
                );
            }
            let decode_s = decode_started.elapsed().as_secs_f64();
            if let Some(chunk) = detokenizer.finish()? {
                if !chunk.is_empty() {
                    reply.push_str(&chunk);
                    on_text(&chunk).map_err(ChatError::from_output)?;
                }
            }
            self.record_turn(need, generated.len(), decode_s, &reply);
            return Ok(reply);
        }

        // Fallback: barrier per token. Correct, and ~0.4 tok/s slower.
        //
        // The last forward is skipped, and that is not an optimisation: it
        // produces a token nobody will emit, and — more importantly — it would
        // leave the KV cache one position further along than the pipelined loop
        // above, so the two paths would disagree about `context_used()` and the
        // next turn's resume position. They are the A/B control for the pipelined
        // path and for the token-stream equality test
        // (`tests/multi_turn_chat.rs`), so they have to agree on everything, not
        // just on the text.
        while generated.len() < self.max_new_tokens {
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
            if generated.len() >= self.max_new_tokens {
                break;
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
        self.record_turn(need, generated.len(), decode_s, &reply);
        Ok(reply)
    }

    /// Book-keeping shared by both decode paths, so they cannot disagree about
    /// what a "turn" recorded. The two loops differ in how they submit work, not
    /// in what they count.
    fn record_turn(&mut self, prefill: usize, generated: usize, decode_s: f64, reply: &str) {
        self.messages.push(ChatMessage {
            role: ChatRole::User,
            text: std::mem::take(&mut self.pending_user_text),
        });
        self.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            text: reply.to_string(),
        });
        self.turns += 1;
        self.total_prefill_tokens += prefill;
        self.total_generated_tokens += generated;
        self.total_decode_s += decode_s;
        self.last_stats = Some(TurnStats {
            prefill_tokens: prefill,
            generated_tokens: generated,
            decode_tok_s: if decode_s > 0.0 {
                generated as f64 / decode_s
            } else {
                0.0
            },
        });
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

#[cfg(test)]
mod tests {
    use super::{fits_in_context, ChatError, SessionTotals};

    #[test]
    fn context_limit_boundary() {
        // Exactly at 8192 fits; one over does not. The error carries the
        // counts the tool reports instead of truncating.
        assert!(fits_in_context(8000, 100, 92, 8192).is_ok());
        assert!(fits_in_context(8000, 100, 93, 8192).is_err());
        match fits_in_context(8000, 100, 93, 8192) {
            Err(ChatError::ContextOverflow { used, need, limit }) => {
                assert_eq!((used, need, limit), (8000, 193, 8192));
            }
            _ => panic!("expected ContextOverflow"),
        }
    }

    /// The session rate is token-weighted, not a mean of per-turn rates. Two
    /// turns of wildly different length is the case that separates them: a mean
    /// of (2, 20) is 11 tok/s, while the true rate over the same work is
    /// 22 / (2/2 + 4/20) = 15.2.
    #[test]
    fn session_rate_is_token_weighted_not_a_mean_of_turns() {
        let totals = SessionTotals::from_accumulated(
            2,
            12,
            22,
            // 2 tokens in 1.0 s (2 tok/s), then 20 tokens in 1.0 s (20 tok/s).
            2.0,
        );
        assert_eq!(totals.generated_tokens, 22);
        assert_eq!(totals.decode_tok_s, 11.0);

        // Equal time per turn, so the token-weighted rate is the arithmetic mean
        // only because the times happen to be equal. Change the times and the
        // two must diverge -- which is the property being asserted.
        let unequal = SessionTotals::from_accumulated(
            2,
            12,
            22,
            // 2 tokens in 0.5 s (4 tok/s), then 20 tokens in 1.5 s (13.33 tok/s).
            2.0,
        );
        let mean_of_rates = (4.0 + 20.0 / 1.5) / 2.0;
        assert!(
            (unequal.decode_tok_s - mean_of_rates).abs() > 0.5,
            "expected the token-weighted rate ({:.2}) to differ from the mean of \
             per-turn rates ({mean_of_rates:.2})",
            unequal.decode_tok_s
        );
        assert_eq!(unequal.decode_tok_s, 11.0);
    }

    /// A session that generated nothing must report 0, not NaN or infinity.
    /// Reachable: `--max-new-tokens 0`, or a prompt answered with an immediate
    /// EOS.
    #[test]
    fn a_session_with_no_decode_time_reports_zero_not_nan() {
        let totals = SessionTotals::from_accumulated(3, 30, 0, 0.0);
        assert_eq!(totals.decode_tok_s, 0.0);
        assert!(totals.decode_tok_s.is_finite());
        assert_eq!(totals.turns, 3);
    }

    /// A turn that generated tokens in zero measured time must not divide by
    /// zero. `Instant` can legitimately report a 0 ns elapsed for a turn that
    /// hit `max_new_tokens` immediately.
    #[test]
    fn zero_elapsed_time_with_tokens_present_still_reports_zero() {
        let totals = SessionTotals::from_accumulated(1, 5, 1, 0.0);
        assert_eq!(totals.decode_tok_s, 0.0);
    }
}

#[cfg(test)]
mod device_sampler_gate_tests {
    use super::*;
    use crate::sampler::{Sampler, SamplerConfig, SamplerParams};

    #[test]
    fn the_shipped_default_config_is_servable_on_device() {
        let config = SamplerConfig::default();
        assert_eq!(config.temperature, 1.0, "DEFAULT_TEMPERATURE is 1.0");
        assert!(
            SamplerParams::from_config(config, 262_144).is_ok(),
            "the shipped default (temp 1.0, top_k 64, top_p 0.95) must reach the \
             device sampler, or the T5/T6 gate is dead code in practice"
        );
    }

    #[test]
    fn an_unstagable_top_k_is_refused_not_approximated() {
        let mut config = SamplerConfig::default();
        config.top_k = crate::sampler::device_top_k_limit(262_144) * 2;
        assert!(
            SamplerParams::from_config(config, 262_144).is_err(),
            "a top_k past the kernel's staging capacity must be refused so chat \
             falls back to the host sampler"
        );
    }

    #[test]
    fn the_host_sampler_still_works_on_the_default_config() {
        let mut sampler = Sampler::new(SamplerConfig::default(), 1).unwrap();
        let logits: Vec<f32> = (0..1024).map(|i| -(i as f32)).collect();
        let _ = sampler.sample(&logits).unwrap();
    }
}
