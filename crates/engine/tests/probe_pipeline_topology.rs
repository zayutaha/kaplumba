//! B2 — decode-pipeline topology: a **static** map of the two decode loops, the
//! device-side token hand-off, and the true depth-1 blocker.
//!
//! # Scope and honesty
//!
//! This file runs **no GPU code and loads no checkpoint**. Every assertion is
//! either (a) a source assertion over `include_str!`ed files, or (b) pure
//! arithmetic over a timeline model written from reading those loops end to end.
//! Nothing here measures milliseconds. Where a millisecond figure appears it is
//! labelled `MEASURED-BY-OTHERS` and carries the file that recorded it; this
//! file does not launder another agent's number into its own.
//!
//! # Why it exists
//!
//! Three claims in the decode programme are load-bearing and none of them had
//! been checked against the tree at `641e114`:
//!
//! 1. **"There is no device-side token hand-off, so 1-deep is structural."**
//!    (`model.rs` doc on `forward_buffer_into_step`; restated in
//!    `P02-dispatch-overhead.md`.) The stated reason is that
//!    `ElementKernels::encode_greedy_pick` refuses destination index 0. That
//!    kernel is **not on the decode path**: the head now calls `encode_argmax` +
//!    `encode_argmax_to_token`, and `argmax_keys_to_token` writes `token_out[0]`
//!    with no destination-index guard at all. The hand-off exists, and the
//!    pipelined loop depends on it every step.
//! 2. **The 3.83 ms "barrier penalty" is one number with no decomposition.** The
//!    two arms of the A/B that produced it (`chat.rs`'s `pipelined` /
//!    `KAPLUMBA_NO_PIPELINE` branches) differ in more than the barrier, and one of
//!    the extra differences is a host scan of the whole 262,144-wide vocabulary
//!    that the temperature-1.0 loop never pays — while the same 512 KiB readback
//!    is billed a second time as its own line in `A5-budget.md` §4.
//! 3. **`decode_max_dispatches_per_buffer() -> u32::MAX` was only ever reasoned
//!    about for the pipelined path.** It is *also* set on the barrier-per-token
//!    path (`forward_buffer_into_vec`), which is the path that needs it.
//!
//! # What this file deliberately does NOT claim
//!
//! That a deeper pipeline would be fast. It shows the blocker is a *shared
//! activation scratch*, not the token, and then stops. Whether depth 2 pays is a
//! measurement; `spec/decode-ceiling/findings/B2-pipeline.md` argues from the
//! measured post-wait host tail that it probably does not, and
//! [`depth_two_is_worth_the_post_wait_tail_and_nothing_else`] pins that argument
//! so it cannot be quietly upgraded to a multi-millisecond claim.
//!
//! # Bit-exactness
//!
//! Nothing here changes a number a kernel computes. The file is a test.

use std::collections::BTreeSet;

// `include_str!` rather than `std::fs::read_to_string` on a path: the test must
// not pass because it ran from the wrong directory, and it must recompile when
// the source it describes changes.
const MODEL_RS: &str = include_str!("../src/model.rs");
const GENERATE_RS: &str = include_str!("../src/generate.rs");
const CHAT_RS: &str = include_str!("../src/chat.rs");
const ENCODER_RS: &str = include_str!("../../kernels/src/encoder.rs");
const ELEMENTS_RS: &str = include_str!("../../kernels/src/elements.rs");
const ELEMENTS_METAL: &str = include_str!("../../kernels/src/elements.metal");

/// Byte offset of the `n`-th occurrence of `needle`, 1-based.
fn nth(haystack: &str, needle: &str, n: usize) -> usize {
    haystack
        .match_indices(needle)
        .nth(n - 1)
        .unwrap_or_else(|| panic!("occurrence {n} of {needle:?} not found"))
        .0
}

/// The body of `elements.metal`'s `argmax_keys_to_token`, brace-matched from its
/// signature. Brace matching rather than "next blank line" because the kernel has
/// a `for` loop and two `if`s, and a line-based window silently grows to include
/// the next kernel — which is how a previous attempt at this assertion passed for
/// the wrong reason and then failed for the wrong reason.
fn argmax_keys_to_token_body() -> &'static str {
    let start = nth(ELEMENTS_METAL, "kernel void argmax_keys_to_token(", 1);
    let bytes = ELEMENTS_METAL.as_bytes();
    let mut depth = 0usize;
    for (offset, byte) in bytes[start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &ELEMENTS_METAL[start..start + offset + 1];
                }
            }
            _ => {}
        }
    }
    panic!("argmax_keys_to_token body is not brace-balanced");
}

/// Code lines of `file`, with `///` and `//` comment lines removed.
///
/// Used wherever the claim is about *code* rather than about documentation:
/// `model.rs` names `encode_greedy_pick` in a doc comment, and a naive
/// `contains` would read that as a call and reach the opposite conclusion.
fn code_lines(file: &str) -> String {
    file.lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("//")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn assert_before(file: &str, first: &str, second: &str) {
    let a = file
        .find(first)
        .unwrap_or_else(|| panic!("{first:?} not found"));
    let b = file
        .find(second)
        .unwrap_or_else(|| panic!("{second:?} not found"));
    assert!(
        a < b,
        "expected {first:?} (offset {a}) before {second:?} (offset {b})"
    );
}

// ---------------------------------------------------------------------------
// 1. The device-side token hand-off exists. The "1-deep is structural" note is
//    stale: it describes a kernel the decode head stopped calling.
// ---------------------------------------------------------------------------

/// `argmax_keys_to_token` writes the picked token at **index 0** of the next
/// step's token buffer, unconditionally, and stages the host-visible copy at
/// `host_tokens[step_idx]`.
///
/// This is exactly the dependency the "no device-side hand-off" note denies. If
/// anyone ever adds a destination-index guard here, the pipelined loop silently
/// embeds token 0 from step 2 onward — the symptom already recorded once for the
/// host-staged variant in the `forward_buffer_into` doc comment. This test is
/// what fails first.
#[test]
fn device_token_handoff_writes_index_zero() {
    let kernel = argmax_keys_to_token_body();

    assert!(
        kernel.contains("token_out[0] = token;"),
        "the hand-off no longer writes index 0:\n{kernel}"
    );
    assert!(
        kernel.contains("host_token_out[step_idx] = token;"),
        "the hand-off no longer stages the host-visible copy:\n{kernel}"
    );
    // A destination-index guard would be a `== 0` test on an *index* variable.
    // The kernel's only `== 0` is the threadgroup-lid guard, which is not one, so
    // asserting on `== 0` at all would be a false tripwire: assert on the two
    // guard *shapes* instead.
    for guard in ["row == 0", "idx == 0", "index == 0", "out_row == 0"] {
        assert!(
            !kernel.contains(guard),
            "argmax_keys_to_token grew a destination guard `{guard}`; the device \
             hand-off is blocked and the pipelined loop reads whatever the slot \
             held:\n{kernel}"
        );
    }
    assert_eq!(
        kernel.matches("== 0").count(),
        1,
        "unexpected extra `== 0` in argmax_keys_to_token:\n{kernel}"
    );
}

/// The blocker named in the stale note is real **for `encode_greedy_pick`**, and
/// that kernel is dead code on the decode path.
///
/// Asserted as a pair because either half alone misleads: `encode_greedy_pick`
/// really does refuse row 0 (`elements.rs`), so grepping for the refusal
/// concludes the pipeline is blocked; and `model.rs` really does not *call* it, so
/// the refusal never runs.
#[test]
fn the_row_zero_blocker_is_real_but_off_the_decode_path() {
    let start = nth(ELEMENTS_RS, "pub fn encode_greedy_pick(", 1);
    let body = &ELEMENTS_RS[start..start + 700];
    assert!(
        body.contains("if row == 0 {"),
        "encode_greedy_pick no longer refuses row 0; the stale note's mechanism \
         has changed, so the note needs rewriting rather than deleting"
    );
    assert!(
        body.contains("greedy pick row must be non-zero"),
        "encode_greedy_pick's row-0 error message changed shape"
    );
    assert!(
        !code_lines(MODEL_RS).contains("encode_greedy_pick"),
        "model.rs now CALLS encode_greedy_pick. If it is back on the decode path, \
         the 1-deep pipeline is blocked after all, and every claim in \
         spec/decode-ceiling/findings/B2-pipeline.md about depth is void."
    );
    // It must still be *mentioned*, or the doc comment that carries the stale
    // claim has been deleted and this file's premise needs revisiting.
    assert!(
        MODEL_RS.contains("encode_greedy_pick"),
        "the doc comment naming encode_greedy_pick is gone; nothing is asserting \
         the stale note any more, so it will not be corrected"
    );
}

// ---------------------------------------------------------------------------
// 2. What actually blocks depth >= 2: one shared activation scratch.
// ---------------------------------------------------------------------------

/// Every decode step writes the *same* `hidden_a` from the embedding and
/// ping-pongs the same pair across layers. Two steps in flight on one queue
/// would have step N+1's embedding overwrite the buffer step N's layer 1 is
/// still reading — a silent wrong-answer race, not a crash.
#[test]
fn activations_are_single_instance_per_model() {
    let embed = nth(MODEL_RS, "self.elements.encode_embedding_scaled(", 1);
    assert!(
        MODEL_RS[embed..embed + 400].contains("&self.scratch.hidden_a"),
        "the embedding no longer writes self.scratch.hidden_a"
    );

    let pingpong = nth(MODEL_RS, "let (input, output) = if index % 2 == 0 {", 1);
    let window = &MODEL_RS[pingpong..pingpong + 320];
    assert!(
        window.contains("&self.scratch.hidden_a, &self.scratch.hidden_b"),
        "the layer ping-pong is no longer over scratch.hidden_a/hidden_b"
    );
    assert!(
        window.contains("&self.scratch.hidden_b, &self.scratch.hidden_a"),
        "the layer ping-pong no longer alternates back"
    );

    // And the layer body receives the scratch as a *shared* borrow with no slot
    // parameter, so "give step N+1 its own scratch" is not expressible today.
    let layer = nth(MODEL_RS, "pub fn forward_buffer(\n        &mut self,", 1);
    let signature = &MODEL_RS[layer..layer + 900];
    assert!(
        signature.contains("scratch: &ForwardScratch"),
        "DecoderLayer::forward_buffer no longer takes `scratch: &ForwardScratch`"
    );
    assert!(
        !signature.contains("slot") && !signature.contains("instance"),
        "DecoderLayer::forward_buffer grew a slot/instance parameter. Depth >= 2 is \
         no longer blocked by the shared scratch, and \
         spec/decode-ceiling/findings/B2-pipeline.md must be rewritten."
    );
}

/// The per-step input slots are two, which is exactly the depth-1 window. The host
/// writes `query_positions_decode[slot]` at *encode* time, so a depth-D pipeline
/// needs D positions in flight, not two.
#[test]
fn input_slots_are_exactly_the_depth_one_window() {
    assert!(
        MODEL_RS.contains("pub query_positions_decode: [Buffer; 2],"),
        "query_positions_decode is no longer a 2-slot array"
    );
    assert!(
        MODEL_RS.contains("pub token_ids_decode: [Buffer; 2],"),
        "token_ids_decode is no longer a 2-slot array"
    );

    let start = nth(MODEL_RS, "pub fn forward_step_pipelined(", 1);
    let body = &MODEL_RS[start..start + 1400];
    assert!(
        body.contains("let slot = slot % 2;"),
        "forward_step_pipelined no longer folds the slot modulo 2"
    );
    assert!(
        body.contains("let next_slot = 1 - slot;"),
        "forward_step_pipelined no longer targets 1 - slot for the next token"
    );
    assert!(
        body.contains("self.scratch.query_positions_decode[slot].contents() as *mut u32"),
        "forward_step_pipelined no longer host-stages the position into the slot"
    );
}

// ---------------------------------------------------------------------------
// 3. The two decode loops, as text. Every difference between them that is not
//    "the barrier" is named here, so the 3.83 ms delta stops being atomic.
// ---------------------------------------------------------------------------

/// The temperature>0 loop is **fully serial**, and the shape is
/// `wait → copy → widen → sample → submit`.
///
/// The textual order inside one iteration is *sample then submit*, because the
/// `wait` lives at the end of the **previous** iteration, inside
/// `forward_buffer_into_vec`. So the barrier is the last thing before the loop
/// turns over and the sample is the first thing after — which is exactly the
/// serial chain, and the reason the GPU sits idle for the whole of it.
#[test]
fn temperature_path_is_wait_copy_widen_sample_submit() {
    // `generate_gpu` has two `for step in 0..config.max_new_tokens` loops in the
    // file; the one that matters is inside `generate_gpu`, after the
    // temperature==0 early return.
    let gate = nth(GENERATE_RS, "pub fn generate_gpu<F>(", 1);
    let loop_start = nth(&GENERATE_RS[gate..], "for step in 0..config.max_new_tokens {", 1) + gate;
    let body = &GENERATE_RS[loop_start..loop_start + 3000];

    // Within one iteration: the token is picked first, and only then is the
    // next step submitted. The submission is last, so its barrier is the last
    // thing before the loop turns over.
    assert_before(body, "half::f16::from_bits", "sampler.sample");
    assert_before(body, "sampler.sample", "forward_buffer_into_vec");
    assert_eq!(
        body.matches("forward_buffer_into_vec").count(),
        1,
        "the sampling loop body submits more than once per step; the serial \
         wait -> copy -> sample -> submit shape has changed"
    );
    // And the submit is the barrier, so nothing between it and the next
    // iteration's sample can overlap the GPU.
    assert!(
        GENERATE_RS[loop_start..loop_start + 3000].contains("decode_started"),
        "the sampling loop lost its step timer; the timed region no longer spans \
         submit-and-wait"
    );
}

/// `forward_buffer_into_vec` — the temperature>0 submit — waits **before** it
/// reads the row back, so the 512 KiB copy is host time inside the step, not
/// overlapped work.
#[test]
fn readback_happens_after_the_barrier_not_before_it() {
    let start = nth(MODEL_RS, "pub fn forward_buffer_into_vec(", 1);
    let body = &MODEL_RS[start..start + 3000];
    assert_before(body, "encoder.commit_and_wait()", "copy_nonoverlapping");
    assert!(
        body.contains("self.scratch.capped.contents() as *const u16"),
        "the readback source changed; it is no longer scratch.capped"
    );
}

/// The barrier lives inside `commit_and_wait`, so it is not something the caller
/// of an encoder can skip. That is why the temperature>0 loop cannot get any
/// overlap without changing which entry point it uses — and why the fix is
/// ungating the pipeline, not removing the barrier.
#[test]
fn the_barrier_is_inside_commit_and_wait_not_in_the_caller() {
    let start = nth(ENCODER_RS, "pub fn commit_and_wait(mut self)", 1);
    let body = &ENCODER_RS[start..start + 2200];
    assert!(
        body.contains("self.command_buffer.wait_until_completed();"),
        "commit_and_wait no longer waits on its own buffer"
    );
}

// ---------------------------------------------------------------------------
// 4. Disabling the roll: which paths it reaches, and why it cannot deadlock.
// ---------------------------------------------------------------------------

/// `decode_max_dispatches_per_buffer()` is set on **both** decode routes, so the
/// proposed constant reaches the barrier-per-token path — the one that needs it —
/// and not only the pipelined path the earlier reasoning covered.
#[test]
fn the_roll_threshold_reaches_the_non_pipelined_path() {
    let start = nth(MODEL_RS, "pub fn forward_buffer_into_vec(", 1);
    assert!(
        MODEL_RS[start..start + 1200].contains(
            "encoder.set_max_dispatches(kaplumba_kernels::encoder::decode_max_dispatches_per_buffer())"
        ),
        "forward_buffer_into_vec no longer sets the decode dispatch threshold, so \
         the proposed constant would not reach the non-pipelined path"
    );
    // generate.rs: three decode sites (two pipelined arms + the greedy
    // barrier-per-token arm), and separately one *prefill* site that must not be
    // swept up by the change.
    assert_eq!(
        GENERATE_RS.matches("decode_max_dispatches_per_buffer()").count(),
        3,
        "generate.rs no longer sets the decode threshold on exactly its three \
         decode sites; find out which moved before changing the constant"
    );
    // The prefill site must keep using the *other* knob. `encoder.rs` documents
    // why they are separate: folding them together makes the decode-threshold A/B
    // move prefill's too, which is the confound the A5 measurement plan warns
    // about. Count the call, not the identifier — the identifier also appears on
    // its own `fn` definition, which is how a naive count reads 2.
    assert_eq!(
        GENERATE_RS
            .matches("set_max_dispatches(prefill_max_dispatches_per_buffer())")
            .count(),
        1,
        "the prefill threshold call site moved or was duplicated. It is a separate \
         knob on purpose; folding it into the decode knob is the confound the A5 \
         measurement plan warns about."
    );
    assert!(
        GENERATE_RS.contains("fn prefill_max_dispatches_per_buffer() -> u32 {"),
        "the prefill threshold function is gone; find out where prefill's rolling \
         threshold now comes from before changing the decode constant"
    );
}

/// Every terminal commit records its buffers, so with rolling disabled every
/// buffer a step creates is still waited on. That is the safety argument for
/// raising the threshold: the number of waits falls, but no buffer becomes
/// un-waited, so there is nothing to deadlock against.
#[test]
fn every_commit_path_records_in_flight() {
    for needle in [
        "pub fn commit_terminal(&mut self)",
        "pub fn commit_stamped(&mut self",
        "pub fn commit_and_wait(mut self)",
    ] {
        let start = nth(ENCODER_RS, needle, 1);
        assert!(
            ENCODER_RS[start..start + 2600].contains("in_flight"),
            "{needle} does not mention in_flight; the 'nothing becomes un-waited' \
             argument for disabling the roll no longer holds"
        );
    }
    // `roll` is the only place a buffer is committed *and* recorded into
    // `in_flight`, and it is reachable only from `tick`, i.e. only on the
    // dispatch counter. So a threshold above the step's dispatch count removes
    // every commit except the terminal one.
    let tick = nth(ENCODER_RS, "fn tick(&mut self)", 1);
    let tick = &ENCODER_RS[tick..tick + 320];
    assert!(
        tick.contains("if self.dispatches >= self.max_dispatches {") && tick.contains("self.roll()"),
        "tick no longer gates the roll on the dispatch counter"
    );
    let roll = nth(ENCODER_RS, "fn roll(&mut self)", 1);
    assert!(
        ENCODER_RS[roll..roll + 700].contains("if self.in_flight.len() > max_in_flight()"),
        "roll no longer throttles on max_in_flight, so both the '7 waits per \
         step' count and the KAPLUMBA_MAX_IN_FLIGHT=1 race diagnostic change \
         meaning"
    );
}

// ---------------------------------------------------------------------------
// 5. Dispatch parity, and the one A/B-arm term the temperature>0 loop never pays.
// ---------------------------------------------------------------------------

/// The two arms issue the **same number** of head dispatches: the pipelined arm
/// gains `argmax_token` and skips `softcap`, the barrier arm the reverse. So the
/// 3.83 ms delta is not a dispatch-count artifact.
#[test]
fn head_dispatches_are_count_equal_across_the_arms() {
    let start = nth(MODEL_RS, "fn encode_lm_head(", 1);
    let head = &MODEL_RS[start..start + 4200];

    assert!(
        head.contains("let skip_softcap = pick.skips_softcap(out.is_some());"),
        "the softcap gate is no longer derived from pick + staging"
    );
    assert!(
        head.contains("if let Some(out) = out {"),
        "the argmax_token staging block moved"
    );
    let flattened: String = head.split_whitespace().collect();
    assert!(
        flattened.contains(".encode_argmax_to_token("),
        "encode_argmax_to_token is no longer the staged hand-off"
    );

    // pipelined: argmax (2) + argmax_token (1) + no softcap = 3
    // barrier:   argmax (2) + no argmax_token + softcap (1) = 3
    const PIPELINED: u32 = 2 + 1;
    const BARRIER: u32 = 2 + 1;
    assert_eq!(PIPELINED, BARRIER, "the parity argument is wrong");
}

/// The A/B that produced the 3.83 ms ran `chat.rs`'s two arms, and the barrier arm
/// there pays a **host argmax over the whole vocabulary** — a cost the
/// temperature>0 loop never pays, because it samples instead. So 3.83 is not a
/// clean barrier measurement, and `A5-budget.md` §4 bills the same 512 KiB readback
/// a second time as its own line item.
#[test]
fn chat_barrier_arm_pays_a_host_vocab_scan_the_temperature_path_never_pays() {
    let start = nth(CHAT_RS, "// Fallback: barrier per token.", 1);
    let arm = CHAT_RS[start..start + 1800].replace([' ', '\n'], "");
    assert!(
        arm.contains("argmax_logits(&logits)"),
        "chat.rs's barrier arm no longer host-scans the logits row, so it is now \
         a clean barrier measurement and the decomposition in B2-pipeline.md is \
         stale"
    );
    assert!(
        arm.contains(".forward_buffer_into_vec(&[token],&[position],&mutlogits)?"),
        "chat.rs's barrier arm no longer goes through forward_buffer_into_vec"
    );

    // The sampling loop in generate_gpu does mention argmax_logits, but only in
    // an arm unreachable at run time: generate_gpu early-returns to
    // generate_gpu_greedy whenever temperature == 0. Pin both halves.
    let gate = nth(GENERATE_RS, "pub fn generate_gpu<F>(", 1);
    let scoped = &GENERATE_RS[gate..];
    assert_before(
        scoped,
        "if config.sampler.temperature == 0.0 {",
        "for step in 0..config.max_new_tokens {",
    );
    let loop_start = nth(scoped, "for step in 0..config.max_new_tokens {", 1);
    let body = &scoped[loop_start..loop_start + 3000];
    assert!(
        body.contains("argmax_logits(&logits)"),
        "the sampling loop's dead greedy arm no longer mentions argmax_logits"
    );
    // `greedy` is computed before the loop, so the assertion has to reach above
    // it — the loop body alone will not find the declaration.
    let greedy_flag = nth(
        &scoped[..loop_start],
        "let greedy = config.sampler.temperature == 0.0;",
        1,
    );
    assert!(
        greedy_flag < loop_start,
        "the sampling loop's `greedy` flag is gone; the dead-arm argument needs \
         re-deriving"
    );
}

// ---------------------------------------------------------------------------
// 6. The post-wait host tail: the only thing a deeper pipeline could recover.
// ---------------------------------------------------------------------------

/// Operations the pipelined loop performs on the host **after** `wait()` and
/// **before** the next step's encode. This is the quantity a deeper pipeline
/// would take off the critical path. It is *enumerated*, not estimated, because
/// estimating it needs the GPU.
fn pipelined_host_tail_ops() -> BTreeSet<&'static str> {
    let start = nth(GENERATE_RS, "for step_idx in 1..total_to_generate {", 1);
    let body = &GENERATE_RS[start..start + 2600];
    let wait_at = body
        .find("in_flight_enc.wait();")
        .expect("pipelined loop has no in_flight_enc.wait()");
    let tail = &body[wait_at..];

    let mut ops = BTreeSet::new();
    if tail.contains("model.read_host_token(in_flight_step)") {
        ops.insert("read_host_token: 4-byte host read of GPU-written shared memory");
    }
    if tail.contains("detokenizer.push(step_token)") {
        ops.insert("Detokenizer::push: replays every id generated so far (quadratic)");
    }
    if tail.contains("on_text(&chunk)") {
        ops.insert("on_text: the caller's write to stdout, one syscall per chunk");
    }
    if tail.contains("segments.record(") {
        ops.insert("SegmentLog::record");
    }
    if tail.contains("pipeline_audit::assert_no_builds_since(") {
        ops.insert("pipeline_audit::assert_no_builds_since: one relaxed atomic load");
    }
    if tail.contains("config.eos_ids.contains(&step_token)") {
        ops.insert("eos_ids.contains");
    }
    ops
}

/// The tail is a fixed, short list of host work, and none of it is on the token
/// path — the next step's encode does not need the token. So a deeper pipeline's
/// whole prize is this list, and the list is small and mostly O(1) per token.
///
/// **Known limitation, stated rather than papered over:** this test recognises
/// tail operations by the calls it enumerates. It will fail if an item is
/// removed or if a *recognised* O(n) item joins, and it will **not** fail if
/// someone inserts host work this file has never heard of. It is a tripwire
/// against the known list drifting, not a proof that the list is complete. The
/// completeness claim needs a measurement, and this file takes none.
#[test]
fn the_depth_one_prize_is_a_small_host_tail() {
    let ops = pipelined_host_tail_ops();
    for expected in [
        "read_host_token: 4-byte host read of GPU-written shared memory",
        "Detokenizer::push: replays every id generated so far (quadratic)",
        "on_text: the caller's write to stdout, one syscall per chunk",
        "eos_ids.contains",
    ] {
        assert!(
            ops.contains(expected),
            "the post-wait host tail lost {expected:?}; it is now:\n{ops:#?}"
        );
    }
    // Exactly one member grows with the response length. If a second O(n) item
    // joins, the depth-1 argument's magnitude changes and this must fail.
    let growing = ops.iter().filter(|op| op.contains("replays every id")).count();
    assert_eq!(growing, 1, "a second O(n) item joined the tail:\n{ops:#?}");
}

// ---------------------------------------------------------------------------
// 7. A pure timeline model for both paths — arithmetic only, no measurement.
// ---------------------------------------------------------------------------

/// What one decode step is made of, for each of the two topologies.
///
/// `gpu_busy_ms` is device time for a step — the only input this model needs
/// from a measurement. `encode_ms` is host time to encode a step. `tail_ms` is
/// host time between the wait returning and the next submit. `readback_ms` is
/// host time after the barrier for the row copy, widen and sample.
///
/// The model is deliberately **not** a claim about where the milliseconds are.
/// It is the shape of the two schedules, so the two topologies can be compared
/// without anyone re-deriving the arithmetic from prose — and so the claims in
/// `B2-pipeline.md` are falsifiable rather than rhetorical.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Timeline {
    step_ms: f64,
    gpu_idle_ms: f64,
}

fn pipelined_step(gpu_busy_ms: f64, encode_ms: f64, tail_ms: f64) -> Timeline {
    // The host encodes step N+1 while the GPU runs step N, so the step is the
    // longer of the two, plus whatever the host does after the wait with nothing
    // queued behind it.
    Timeline {
        step_ms: gpu_busy_ms.max(encode_ms) + tail_ms,
        gpu_idle_ms: tail_ms,
    }
}

fn barrier_step(
    gpu_busy_ms: f64,
    encode_ms: f64,
    tail_ms: f64,
    readback_ms: f64,
) -> Timeline {
    // No overlap at all, and the readback sits inside the timed region.
    Timeline {
        step_ms: encode_ms + gpu_busy_ms + tail_ms + readback_ms,
        gpu_idle_ms: encode_ms + tail_ms + readback_ms,
    }
}

#[test]
fn timeline_model_separates_the_two_topologies() {
    let gpu_busy = 60.0;
    let encode = 6.8; // MEASURED-BY-OTHERS: P02/A5, 750 dispatches x 9.03 us
    let tail = 0.104; // MEASURED-BY-OTHERS: P06, 0.004 + detok ~0.1
    let readback = 3.47; // MEASURED-BY-OTHERS: P06/A4, sample 2.265 + widen + copy

    let p = pipelined_step(gpu_busy, encode, tail);
    let b = barrier_step(gpu_busy, encode, tail, readback);

    // The pipelined step hides the encode entirely, because 60 > 6.8.
    assert!((p.step_ms - 60.104).abs() < 1e-9, "the model lost the encode hiding");

    // The tail is in BOTH topologies, so it cancels: the barrier topology's
    // excess is exactly the encode it cannot hide plus the readback it must do.
    let excess = b.step_ms - p.step_ms;
    assert!(
        (excess - (encode + readback)).abs() < 1e-9,
        "the barrier topology must exceed the pipelined one by the encode plus the \
         readback: {excess} vs {}",
        encode + readback
    );
    assert_eq!(p.gpu_idle_ms, tail, "on the pipelined topology the only GPU idle is the tail");
    assert!(
        b.gpu_idle_ms > readback,
        "the barrier topology's idle must at least contain the readback"
    );
}

/// **The programme's own published inputs do not close, and this test pins that
/// fact rather than hiding it.**
///
/// `P02`/`A5` put per-step encode latency at 750 x 9.03 us = **6.77 ms**, and
/// `P06`/`A4` put the temperature>0 host chain at **~3.47 ms**. Fed into
/// [`timeline_model_separates_the_two_topologies`], those predict a gap between
/// the two topologies of **10.24 ms**. The measured gap is
/// 69.60 − 62.30 = **7.30 ms** (`A5-budget.md` §4). So **2.94 ms is
/// unaccounted for** — which is exactly the number `A5` §4 itself reports as
/// "the 1-deep pipeline already hides 2.94 ms", reached from the other direction.
///
/// The consequence is a statement about the ledger, not about the hardware: the
/// 3.83 ms "barrier penalty" and the 6.77 ms "per-dispatch latency" cannot both
/// be read as *host time on the critical path of the barrier arm*. At most one of
/// them is, and which one is not established. Until it is, "the 3.83 ms is the
/// barrier" is a label, not a decomposition.
#[test]
fn the_published_inputs_predict_a_gap_larger_than_the_measured_one() {
    let gpu_busy = 60.0;
    let encode = 6.77; // P02/A5: 750 dispatches x 9.03 us
    let tail = 0.104; // P06
    let readback = 3.47; // P06 sample 2.265 + widen + copy

    let predicted = barrier_step(gpu_busy, encode, tail, readback).step_ms
        - pipelined_step(gpu_busy, encode, tail).step_ms;
    let measured = 69.60 - 62.30; // A5-budget.md section 4

    assert!(
        predicted > measured + 2.0,
        "the inputs now agree to within 2 ms ({predicted} vs {measured}); either the \
         per-dispatch latency was re-measured, or the ledger was corrected. Delete \
         this test only when spec/decode-ceiling/findings/B2-pipeline.md section 5 \
         is updated to say which."
    );
    // And the shortfall is the size A5 reports as already-hidden, which is the
    // cross-check that makes this an inconsistency rather than a coincidence.
    assert!(
        (predicted - measured - 2.94).abs() < 0.15,
        "the shortfall is {shortfall}, not the 2.94 ms A5 reports as already hidden",
        shortfall = predicted - measured
    );
}

#[test]
fn timeline_model_degrades_when_the_encode_exceeds_the_gpu() {
    // The model must not claim the pipeline helps when the host is the slower
    // side. If encode_ms > gpu_busy_ms the step is host-bound and the "hiding"
    // argument is vacuous.
    let p = pipelined_step(6.0, 20.0, 0.1);
    assert_eq!(p.step_ms, 20.1);
    assert!(
        p.step_ms > 6.0 + 0.1,
        "a host-bound step must be charged the host cost, not the GPU cost"
    );
}

/// The prize of depth >= 2, in this model, is exactly the post-wait tail, because
/// with D steps queued the GPU never waits on the host again. Asserted so the
/// "depth 2 is worth ~0.1 ms" claim in `B2-pipeline.md` has a test behind it and
/// cannot be quietly upgraded to a multi-millisecond one.
#[test]
fn depth_two_is_worth_the_post_wait_tail_and_nothing_else() {
    let gpu_busy = 60.0;
    let encode = 6.8;
    let tail = 0.104;

    let one_deep = pipelined_step(gpu_busy, encode, tail);
    let two_deep = pipelined_step(gpu_busy, encode, 0.0);
    assert!(
        (one_deep.step_ms - two_deep.step_ms - tail).abs() < 1e-9,
        "depth 2 was credited with more than the post-wait tail: {} vs {}",
        one_deep.step_ms - two_deep.step_ms,
        tail
    );
    assert!(
        two_deep.step_ms >= gpu_busy,
        "depth 2 cannot make a step cheaper than the GPU work inside it"
    );
}