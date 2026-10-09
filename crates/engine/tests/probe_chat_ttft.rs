//! Unit probe and structural invariant verification for sub-300ms TTFT in `--chat`.
//!
//! Validates:
//! 1. Immediate Token 1 streaming: Token 1 must be emitted to `on_text` before
//!    entering the pipelined decode loop and before any decode step wait.
//! 2. `ttft_ms` timing: Recorded immediately after prefill completion and stored
//!    in `TurnStats`.

use kaplumba_engine::chat::TurnStats;

#[test]
fn test_turn_stats_ttft_fields() {
    let stats = TurnStats {
        prefill_tokens: 18,
        generated_tokens: 25,
        decode_tok_s: 15.8,
        ttft_ms: 142.5,
    };
    assert_eq!(stats.prefill_tokens, 18);
    assert_eq!(stats.generated_tokens, 25);
    assert_eq!(stats.decode_tok_s, 15.8);
    assert_eq!(stats.ttft_ms, 142.5);
    assert!(stats.ttft_ms > 0.0);
}

#[test]
fn test_token_1_streamed_before_decode_loop() {
    let src = std::fs::read_to_string(format!(
        "{}/src/chat.rs",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("failed to read crates/engine/src/chat.rs");

    let pipelined = src
        .split("if pipelined {")
        .nth(1)
        .and_then(|body| body.split("// Fallback: barrier per token").next())
        .expect("pipelined decode section");

    let (before_loop, in_loop) = pipelined
        .split_once("loop {")
        .expect("pipelined must contain loop {");

    // 1. Prefill completes before loop
    assert!(
        before_loop.contains("let mut token = if self.greedy {"),
        "prefill token argmax/sampling must happen before loop"
    );

    // 2. Active Token 1 streaming before loop
    let has_active_token1_streaming = before_loop.lines().any(|line| {
        let trimmed = line.trim();
        !trimmed.starts_with("//")
            && !trimmed.starts_with("/*")
            && !trimmed.starts_with('*')
            && trimmed.contains("on_text(&chunk)")
    });
    assert!(
        has_active_token1_streaming,
        "token 1 must have an active uncommented on_text(&chunk) call BEFORE loop begins! \
         Emitting inside or after the loop introduces a full decode step delay to TTFT."
    );

    // 3. Pipelined loop waits on previous step
    assert!(
        in_loop.contains("previous.wait()"),
        "loop must contain previous.wait() for step pipelining"
    );
}

#[test]
fn test_ttft_measured_immediately_after_prefill() {
    let src = std::fs::read_to_string(format!(
        "{}/src/chat.rs",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("failed to read crates/engine/src/chat.rs");

    let turn_fn = src
        .split("pub fn turn<F>")
        .nth(1)
        .and_then(|body| body.split("if pipelined {").next())
        .expect("turn function prefix");

    let turn_start = turn_fn
        .find("let turn_started = std::time::Instant::now();")
        .expect("turn_started timestamp recorded at start of turn");

    let prefill = turn_fn
        .find("forward_buffer_into_vec(&ids, &positions, &mut logits)")
        .expect("prefill forward execution");

    let ttft_calc = turn_fn
        .find("let ttft_ms = turn_started.elapsed().as_secs_f64() * 1000.0;")
        .expect("ttft_ms measured immediately after prefill forward");

    assert!(
        turn_start < prefill,
        "turn_started must precede prefill forward"
    );

    assert!(
        prefill < ttft_calc,
        "ttft_ms must be calculated after prefill forward"
    );
}
