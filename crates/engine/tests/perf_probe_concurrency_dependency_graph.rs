//! Probe for P16 ("Concurrency / multi-queue overlap", `spec/decode-ceiling/hypotheses.md:75`)
//! and P06 ("CPU per-token host work; temp>0 **non-pipelined** path",
//! `spec/decode-ceiling/hypotheses.md:65`).
//!
//! **No GPU, no model, no `scripts/gpuq`.** Every figure below is either (a) a
//! constant transcribed from a recorded measurement with its citation, or (b)
//! arithmetic on (a). The measured constants are additionally *re-verified
//! against the working tree at compile time* by `include_str!` +
//! needle search, so this file cannot silently keep citing an invariant that
//! somebody has since deleted. It is therefore safe to run without the model
//! lock, which is what makes it usable as a gate rather than a note.
//!
//! It writes no files and loads no weights, so `cargo test --test
//! perf_probe_concurrency_dependency_graph` is a CPU-only test.
//!
//! # The question
//!
//! `spec/decode-25tok-peak-bandwidth/tasks.md:45` leaves T5 "Concurrent
//! Multi-Queue Overlap (Attention + GEMV)" open. The intent reads:
//!
//! > Overlap Attention KV-cache streaming on a secondary Metal command queue
//! > with GEMV dequantization on the primary queue using `MTLEvent`
//! > synchronization to absorb the 3.2 ms attention latency.
//!
//! This file answers the three questions that have to be answered *before*
//! T5 can be attempted at all, and then answers "is it worth it" with the
//! arithmetic rather than an opinion.
//!
//! # 1. The per-token dependency graph
//!
//! `TextModel::forward_chunk` (`model.rs:4022`) walks
//! `embedding_scaled` -> 60 x `DecoderLayer::forward_buffer` ->
//! `encode_copy_last_row` -> `encode_lm_head`. Inside one layer the decode
//! path is 10 nodes, and **every one depends on its predecessor's output
//! buffer**:
//!
//! ```text
//!   input (hidden_a | hidden_b, written by layer k-1)
//!     │
//!     ├─(1) input_norm_qkv_transform ...... rms_norm(x, input_norm) -> normed;
//!     │                                          Hadamard signs -> transformed
//!     ├─(2) qkv_proj GEMV (ONE fused dispatch, 33.04 MB)
//!     │        reads transformed; writes q, k, v   [k reused as v on full layers]
//!     ├─(3) norm_rope_qkv_cache (ONE dispatch: q-norm, k-norm, v-norm,
//!     │        RoPE(q), RoPE(k), AND the KV-cache append at `slot`)
//!     ├─(4) attention (1 dispatch single-pass, or split + combine)
//!     │        reads q AND the cache written by (3); writes attended
//!     ├─(5) o_proj input transform (Hadamard, in place on attended)
//!     ├─(6) o_proj GEMV (16.52 MB)
//!     ├─(7) finalize_o_proj_and_pre_ffn_transform (ONE dispatch: split-K reduce,
//!     │        out-signs, post_attention_norm, residual -> hidden_b,
//!     │        pre_ffn_norm, gate Hadamard -> transformed)
//!     ├─(8) gate_up_gelu (ONE fused dispatch: gate + up + gated GeLU, 86.70 MB)
//!     ├─(9) down_proj GEMV (43.35 MB)
//!     └─(10) finalize_and_post_ffn_norm_scale[_and_norm] (ONE dispatch: reduce,
//!              out-signs, post_ffn_norm, residual, layer_scale;
//!              [+ layer k+1's input_norm + qkv transform when cross-layer
//!               fusion applies]) -> output
//! ```
//!
//! So the graph is a **chain of 600**, not a tree. Three consequences:
//!
//! * **Nothing inside a layer is independent of anything else in it.** (4)
//!   attention cannot start before (3) has written this token's K/V into the
//!   cache; (5) cannot start before (4) has written `scratch.attended`; (6)
//!   cannot start before (5) has applied the Hadamard.
//! * **Nothing across layers is independent either.** Layer k+1's (1) reads the
//!   `output` buffer layer k's (10) wrote. (10) already reaches *into* layer
//!   k+1 via the cross-layer fusion (`model.rs:2298-2333`), so the edge is not
//!   even layer-local.
//! * The **only** genuinely independent work in the whole step is the KV-cache
//!   read inside (4) over keys `0..pos-1`, which (3) does not touch. That is one
//!   key out of ~69 (the benchmark's context), or one out of 1024/8192 at the
//!   cap. It is a within-kernel split, not a queue-level one, and it is worth
//!   far less than the number T5 quoted.
//!
//! `q/k/v`, `o_proj` and the MLP are **not** partially parallel in the sense
//! T5 needs. q/k/v is already a *single fused dispatch* on this tree
//! (`model.rs:1462-1533`, label `qkv_proj`), and gate/up likewise
//! (`model.rs:2193-2214`, label `gate_up_gelu`). There is no third stream
//! hiding inside them; fusing them removed the opportunity rather than creating
//! one.
//!
//! # 2. What is already device-side, and what is not
//!
//! The *argmax* is device-side: `argmax_slots` (4 x 65,536 -> 4 u32 keys) and
//! `argmax_keys_to_token` (4 keys -> one u32, into both
//! `token_ids_decode[next_slot]` and `host_tokens[step_idx]`) are both encoded
//! **inside** `forward_chunk` -> `encode_lm_head` (`model.rs:4459-4479`). The
//! greedy pipelined loop therefore reads one `u32` per token
//! (`model.rs:3794-3799`) instead of 262,144 fp16 values.
//!
//! The *sampler* is not, and cannot currently be:
//!
//! * `Sampler` (`sampler.rs:101`) is a pure host function over `&[f32]`. Its file
//!   imports only `thiserror`; there is no Metal dependency in it at all.
//! * `elements.metal` has no top-k, no softmax and no RNG kernel. Its 24 kernels
//!   are listed in `no_device_side_sampler_exists`.
//! * The one device-side pick, `encode_greedy_pick` (`elements.rs:815`), refuses
//!   row 0 (`elements.rs:823-827`) -- and `forward_chunk`'s own comment
//!   (`model.rs:4282-4295`) records that its packed key underflows past 65,536,
//!   so on this 262,144-entry vocabulary it returns token 0 where the host
//!   argmax returns 9,259. It is wrong on top of being unusable.
//!
//! # 3. Consequences for the "overlap sampling with the next step's GEMV" idea
//!
//! The chain at temperature > 0 is:
//!
//! ```text
//!   ... head/gemv -> softcap -> argmax -> argmax_keys_to_token
//!        -> [BARRIER] -> host: widen 262,144 fp16->f32, top-k partition,
//!                        softmax, xorshift -> write token_ids_decode[slot]
//!        -> [BARRIER] -> embedding_scaled -> layer 0 ...
//! ```
//!
//! Two barriers and ~3.2 ms of host work sit between the last GEMV of step N and
//! the embedding of step N+1. Step N+1 **cannot be submitted during that window**
//! because its embedding reads a token the host has not produced yet. So the
//! sampling is not merely un-overlapped; it is *un-overlap-able*, and no amount
//! of queue structure changes that. The single fix is to move the sampler onto
//! the device (reproducing `top_k_order`'s tie rule exactly -- descending value,
//! ties by ascending index, `sampler.rs:72-77` -- because AGENTS.md §8 forbids
//! quality loss), which would then also let `chat.rs:388`'s `self.greedy` gate be
//! dropped.
//!
//! # 4. Verdict
//!
//! **T5 (multi-queue) should be closed as REFUTED, not attempted.** Three
//! independent grounds, any one of which is sufficient:
//!
//! 1. **No work to overlap.** The dependency graph is a chain of 600 (above).
//! 2. **The mechanism is already measured, and it loses.**
//!    `tests/concurrent_stream_economics.rs` runs two bandwidth-bound streams
//!    the shape of `o_proj`'s weight stream and attention's KV read, on two
//!    real queues with a real `MTLEvent`, and *asserts* that the concurrent arm
//!    is >1.2x worse than perfect hiding and that an `MTLEvent`-gated second
//!    queue is not faster than serial order (`concurrent_stream_economics.rs:382`
//!    and `:389`). Those assertions pass on HEAD. That is the mechanism, tested.
//! 3. **The prize is 1.06%, against 23% of headroom that concurrency cannot
//!    reach.** See `bandwidth_headroom_is_not_available_to_a_second_queue`.
//!
//! Meanwhile **P06 (the temperature>0 non-pipelined path) is ~11x larger than
//! T5's ideal ceiling and is composed entirely of already-measured host time**.
//! It is the largest real-world gap on this box: the CLI's default temperature
//! is 1.0 (`cli/src/args.rs:30`), and at 1.0 the engine is *not* pipelined and
//! *does* sample on the host.

use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// Source captured at compile time. Nothing below reads the filesystem at run
// time; these are the real files, so a needle search here is a statement about
// the tree, not about a copy of it.
// ---------------------------------------------------------------------------

const MODEL_RS: &str = include_str!("../src/model.rs");
const ENCODER_RS: &str = include_str!("../../kernels/src/encoder.rs");
const ELEMENTS_RS: &str = include_str!("../../kernels/src/elements.rs");
const ELEMENTS_METAL: &str = include_str!("../../kernels/src/elements.metal");
const SAMPLER_RS: &str = include_str!("../src/sampler.rs");
const CHAT_RS: &str = include_str!("../src/chat.rs");
const GENERATE_RS: &str = include_str!("../src/generate.rs");
const GPU_CACHE_RS: &str = include_str!("../src/gpu_cache.rs");
const CLI_ARGS_RS: &str = include_str!("../../cli/src/args.rs");
const T2B_RS: &str = include_str!("concurrent_stream_economics.rs");

/// Collapses whitespace **and strips `/` and `*`**, so `///` and `//!` doc-comment
/// markers vanish and text either side of a line break becomes adjacent.
///
/// The `/` and `*` part is not a detail. Splitting on whitespace alone turns
/// `"...the same order the\n/// production GEMVs dispatch."` into
/// `"...the same order the /// production GEMVs dispatch."`, so a needle
/// spanning that break still misses — which is exactly what the first draft of
/// this helper did, and it failed on `concurrent_stream_economics.rs` even after
/// the whitespace fallback was added. Every needle in this file that cites a doc
/// comment hits this, so the marker has to go.
fn normalise(text: &str) -> (String, Vec<usize>) {
    let mut out = String::new();
    // `starts[i]` is the offset in `out` at which source line `i`'s first token
    // begins, so a hit in `out` can be mapped back to a line number.
    let mut starts: Vec<usize> = Vec::new();
    for line in text.lines() {
        starts.push(out.len());
        for token in line.split(|c: char| c.is_whitespace() || c == '/' || c == '*') {
            if token.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(token);
        }
    }
    starts.push(out.len());
    (out, starts)
}

/// 1-based line number of the first line of `src` containing `needle`.
///
/// Tries an exact substring first, then a `normalise`-d match. The fallback is
/// load-bearing rather than defensive: rustfmt re-wraps doc comments, so a
/// needle that happened to span a line break when this file was written stops
/// matching the moment anybody edits the paragraph above it.
fn line_of(src: &str, needle: &str) -> usize {
    if let Some(index) = src.lines().position(|line| line.contains(needle)) {
        return index + 1;
    }
    let (flat, starts) = normalise(src);
    let (flat_needle, _) = normalise(needle);
    if let Some(at) = flat.find(&flat_needle) {
        let line = match starts.binary_search(&at) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };
        return line + 1;
    }
    panic!(
        "needle not found in the working tree.\n  needle: {needle:?}\n\
         This file cites the invariant by its source text so that a refactor that \
         deletes it is caught here rather than silently invalidating the \
         conclusion. Either the invariant moved (update this needle) or it is \
         genuinely gone (the conclusion must be re-derived)."
    );
}

/// One single-queue assumption, and the tree evidence that it is load-bearing.
///
/// This is the enumeration the brief asks for: "enumerate every assumption that
/// a single queue underwrites". `single_queue_invariants_are_still_the_only_barrier`
/// asserts each one still exists, so the list cannot rot into folklore.
struct Invariant {
    /// Short name for the assumption.
    what: &'static str,
    /// Which file holds it.
    file: &'static str,
    /// Source text that must still be present (single line, so no wrapping).
    needle: &'static str,
    /// Why a second queue would break it.
    breaks: &'static str,
}

const INVARIANTS: &[Invariant] = &[
    Invariant {
        what: "MetalEncoder owns exactly one queue for its lifetime",
        file: "kernels/src/encoder.rs",
        needle: "queue: &'q CommandQueue,",
        breaks: "`rearm` always draws the next command buffer from `self.queue`, and \
                 every `dispatch_*` encodes into the one `ComputeCommandEncoder` built \
                 from it. There is no per-dispatch queue selection to extend, so a \
                 second queue means a second encoder type.",
    },
    Invariant {
        what: "roll(): ordering between command buffers is implicit",
        file: "kernels/src/encoder.rs",
        needle: "Ordering between buffers is implicit: a single Metal command",
        breaks: "`MAX_IN_FLIGHT` back-pressure waits on the OLDEST buffer and relies \
                 on the queue having run everything committed before it. Two queues \
                 give two independent orderings and `in_flight` has no queue tag.",
    },
    Invariant {
        what: "flush_and_wait(): the newest-buffer wait implies the older ones",
        file: "kernels/src/encoder.rs",
        needle: "future multi-queue change would break silently",
        breaks: "The doc comment names this exact hazard. Draining oldest-first is \
                 what keeps the barrier honest; the fast path (wait only the newest) \
                 would become a race.",
    },
    Invariant {
        what: "wait(): one queue in commit order",
        file: "kernels/src/encoder.rs",
        needle: "the one queue in commit order, and a Metal command queue executes in",
        breaks: "`wait()` drains `command_buffer` then `in_flight`. All of them are \
                 from one queue. A buffer on queue B is invisible to it.",
    },
    Invariant {
        what: "wait(): kept explicit so a multi-queue change is survivable",
        file: "kernels/src/encoder.rs",
        needle: "future change (e.g. spreading work over several queues).",
        breaks: "This is the repo already having written the multi-queue TODO into \
                 the doc. The loop exists so the barrier does not depend on the \
                 ordering argument; a second queue must extend the loop, not the \
                 argument.",
    },
    Invariant {
        what: "commit_and_wait(): drains in_flight 'anyway'",
        file: "kernels/src/encoder.rs",
        needle: "in-order queue the wait above already covers them",
        breaks: "Explicitly redundant today, and the comment says it is there so the \
                 barrier does not depend on single-queue ordering. Two queues turn the \
                 redundancy into the requirement.",
    },
    Invariant {
        what: "commit_stamped(): the queue serialises stamps into one timeline",
        file: "kernels/src/encoder.rs",
        needle: "serialises command buffers, the retained buffers form an ordered",
        breaks: "Every per-kernel attribution in this spec rests on this: \
                 `own_ms`, `gap_ms`, and the whole `KAPLUMBA_DECODE_PROFILE=kernel` \
                 and `KAPLUMBA_GPU_PROFILE` tables. Two queues produce two interleaved \
                 timelines and `gap_ms` becomes meaningless.",
    },
    Invariant {
        what: "MAX_IN_FLIGHT bounds outstanding buffers per queue",
        file: "kernels/src/encoder.rs",
        needle: "pub const MAX_IN_FLIGHT: usize = 8;",
        breaks: "The window was chosen against a measured ~175 ms block in \
                 `new_command_buffer` at ~64 outstanding buffers. Two queues allow \
                 2x the outstanding work for the same constant, halving the margin \
                 -- and `commit_terminal`'s doc records that dropping an uncommitted \
                 buffer blocks in `new_command_buffer`, i.e. a hang.",
    },
    Invariant {
        what: "forward_chunk(): host writes into SHARED scratch need a drained queue",
        file: "engine/src/model.rs",
        needle: "greedy output depend on scheduling: six runs of one 614-token prompt",
        breaks: "`scratch.query_positions` and `scratch.token_ids` are written by the \
                 host once per chunk. The `flush_and_wait()` in front of them is the \
                 barrier. Removing it made greedy output schedule-dependent -- six \
                 runs, two answers. A second queue means this barrier must cover \
                 both queues or it protects nothing.",
    },
    Invariant {
        what: "forward_step_pipelined(): 2-slot ping-pong is enough",
        file: "engine/src/model.rs",
        needle: "There is no device-side token hand-off to overlap the steps with, and",
        breaks: "`token_ids_decode[slot]` and `query_positions_decode[slot]` are host \
                 writes with `slot = step % 2`. Two slots is enough only because \
                 exactly two steps are in flight on one queue. `chat.rs:409-413` \
                 records the wrong-tokens bug when that was violated.",
    },
    Invariant {
        what: "GpuCache::ensure_capacity copies the live KV prefix on the HOST",
        file: "engine/src/gpu_cache.rs",
        needle: "pub fn ensure_capacity(",
        breaks: "It `copy_nonoverlapping`s the whole live prefix on the CPU and then \
                 REPLACES `keys`/`values`/`positions`. On one queue the forward has \
                 an implicit barrier in front of it. Across two queues this frees \
                 buffers the other queue is still reading.",
    },
    Invariant {
        what: "every ForwardScratch buffer is a single instance for all 60 layers",
        file: "engine/src/model.rs",
        needle: "pub struct ForwardScratch {",
        breaks: "hidden_a/b, normed, q, k, v, attended, proj, gate, up, activated, \
                 raw/raw_b/raw_c (the f32 accumulators the split-K GEMVs reduce into), \
                 transformed, attn_partials, argmax_keys. Not one is per-layer, let \
                 alone per-queue. Two queues doing GEMVs would write the same \
                 accumulator.",
    },
    Invariant {
        what: "cross-layer fusion writes the NEXT layer's normed/transformed",
        file: "engine/src/model.rs",
        needle: "let fuse = self.cross_layer_fusion_applies(tokens, next_layer)",
        breaks: "Layer k's last dispatch normalises layer k+1's input and applies \
                 layer k+1's q_proj transform. That is a cross-layer edge expressed \
                 only as 'later in the same command buffer'. Two layers on two queues \
                 would both target `scratch.normed`.",
    },
    Invariant {
        what: "TRACKED storage supplies the inter-dispatch hazard barrier",
        file: "engine/src/model.rs",
        needle: "Untracked mode (eed081e) removed Metal's automatic",
        breaks: "That barrier is PER COMMAND BUFFER. Metal's hazard tracking does not \
                 order dispatches across command buffers on different queues; it needs \
                 an explicit `memoryBarrierWithScope` plus `encode_wait_for_event`. \
                 The untracked experiment already produced NaN logits from exactly \
                 this class of missing barrier.",
    },
    Invariant {
        what: "chat.rs gates pipelining on greedy",
        file: "engine/src/chat.rs",
        needle: "let pipelined = self.greedy &&",
        breaks: "Not a queue invariant, but it is why P06 exists: at the CLI default \
                 temperature of 1.0 the pipelined loop is unreachable.",
    },
    Invariant {
        what: "generate() short-circuits to the greedy path at temperature 0",
        file: "engine/src/generate.rs",
        needle: "if config.sampler.temperature == 0.0 {",
        breaks: "Same shape as the chat gate, and the same consequence: every \
                 tok/s recorded in this spec was measured on the one code path that \
                 the default invocation never takes.",
    },
    Invariant {
        what: "encode_greedy_pick refuses destination row 0",
        file: "kernels/src/elements.rs",
        needle: "greedy pick row must be non-zero",
        breaks: "Row 0 is the index the next step's embedding reads, which is why \
                 there is no device-side token hand-off. `model.rs:4282-4295` also \
                 records the key underflow that makes the pick wrong past 65,536.",
    },
    Invariant {
        what: "sampler.rs has no Metal dependency at all",
        file: "engine/src/sampler.rs",
        needle: "use thiserror::Error;",
        breaks: "Sole import. Sampling is host code by construction, so it cannot \
                 overlap anything on the GPU without being rewritten.",
    },
    Invariant {
        what: "the CLI default temperature is 1.0, i.e. the non-pipelined path",
        file: "crates/cli/src/args.rs",
        needle: "pub const DEFAULT_TEMPERATURE: f32 = 1.0;",
        breaks: "Not a hazard but the reason P06 is the largest real-world gap: an \
                 unflagged run is on the slow path.",
    },
];

fn source_of(file: &str) -> &'static str {
    match file {
        "kernels/src/encoder.rs" => ENCODER_RS,
        "kernels/src/elements.rs" => ELEMENTS_RS,
        "kernels/src/elements.metal" => ELEMENTS_METAL,
        "engine/src/model.rs" => MODEL_RS,
        "engine/src/chat.rs" => CHAT_RS,
        "engine/src/generate.rs" => GENERATE_RS,
        "engine/src/gpu_cache.rs" => GPU_CACHE_RS,
        "engine/src/sampler.rs" => SAMPLER_RS,
        "crates/cli/src/args.rs" => CLI_ARGS_RS,
        "crates/engine/tests/concurrent_stream_economics.rs" => T2B_RS,
        other => panic!("no compiled-in source registered for {other}"),
    }
}

// ---------------------------------------------------------------------------
// Measured constants. Every one carries its citation; nothing here is a guess.
// ---------------------------------------------------------------------------

/// Baseline as handed to this probe: 16.05-16.10 tok/s greedy, pipelined.
/// `spec/decode-ceiling/hypotheses.md:36-39` records the same pair.
const BASELINE_TOK_S: f64 = 16.05;
const BASELINE_MS: f64 = 62.3;

/// Bytes streamed per generated token. Two figures are on record and they
/// disagree by 1.7%; both are asserted below rather than one being quietly
/// preferred.
///
/// * `11.834 GB` -- the itemised derivation in
///   `decode-25tok-peak-bandwidth/implementation_notes.md:404-417`
///   (10.777 GB layers + 1.057 GB lm_head).
/// * `12.04 GB` -- the later figure in `spec/decode-ceiling/hypotheses.md:37`,
///   given as "10.983 GB layer trellis + 1.057 GB lm_head", and the figure the
///   193 GB/s headline is derived from.
const BYTES_PER_TOKEN_ITEMISED: f64 = 11.834e9;
const BYTES_PER_TOKEN_CURRENT: f64 = 12.04e9;

const ACHIEVED_GB_S: f64 = 193.0;
/// Practical streaming ceiling measured by `examples/probe_bw_250gb`;
/// `spec/decode-3bpw-amplification/implementation_notes.md:12` records 250.5.
const COPY_CEILING_GB_S: f64 = 250.5;
const WIRE_GB_S: f64 = 273.0;
/// Best 3 bpw EXL3 GEMV ever measured here (`decode-25tok-peak-bandwidth`
/// §16.3, `colmajor_2col_sk4` at K=21504). 4.8% above what production achieves.
const BEST_3BPW_KERNEL_GB_S: f64 = 203.0;

/// Marginal cost of ablating `kv_norm_rope, attention, head_softcap,
/// head_argmax` in production: +0.21 tok/s == 0.66 ms/step
/// (`decode-25tok-peak-bandwidth/tasks.md:157`, `implementation_notes.md:474`).
/// This is the **ceiling** on what overlapping attention could ever return: it
/// is what attention costs when you delete it and do not replace it.
///
/// A note on the "0.842 ms/pass" figure in the T5 brief, which is **not** in the
/// tree. The only attention timings on record are `attention: ~3.2 ms`
/// (`notes:27`) and `split_attention: 1.23 ms` (`notes:198`), both from
/// `KAPLUMBA_GPU_PROFILE`, which `notes:16.2` explicitly says not to use for
/// per-projection attribution ("do not use it for per-projection bandwidth",
/// and it splits gate/up by something that is not shape). The one *production*
/// measurement is the 0.66 ms ablation above. Separately, `0.84 ms` does appear
/// in this repo — `decode-19tok-repacked-250gb/tasks.md:150-152` — but it is
/// the launch-latency saved by removing 240 dispatches, not attention at all.
/// Do not let the three get conflated.
const ATTENTION_FAMILY_MARGINAL_MS: f64 = 0.66;
/// Marginal cost of ablating all nine element roles: 1.52 ms/step
/// (`implementation_notes.md:475`).
const ALL_ELEMENT_MARGINAL_MS: f64 = 1.52;

/// Production's achieved bandwidth sits **below** the best 3-bpw kernel ever
/// measured on this box. That gap is the P01/P13/P17 argument (kernel
/// efficiency: fewer codes, fewer bytes, better threadgroup shape) and it is
/// emphatically **not** the concurrency argument — no arrangement of queues
/// makes a 3 bpw kernel decode fewer codes.
///
/// A `const` assert rather than a runtime one because both operands are
/// constants: clippy is right that the runtime form asserts nothing, and making
/// it compile-time means a future edit to either figure that breaks the
/// relationship is a build error rather than a quietly-plausible number.
const _: () = assert!(
    ACHIEVED_GB_S < BEST_3BPW_KERNEL_GB_S,
    "production (193 GB/s) is below the best 3 bpw kernel ever measured here \
     (203 GB/s), so the headroom belongs to kernel efficiency, not concurrency"
);

/// T5.8's dispatch census: 749 dispatches per decode step, of which 546 are
/// stamped (9 per layer plus a head epilogue) and 203 are unstamped
/// `encode_input_transform` calls (`tasks.md:100-108`).
const CENSUS_TOTAL_DISPATCHES: u32 = 749;
const CENSUS_STAMPED: u32 = 546;
const CENSUS_UNSTAMPED: u32 = 203;
const CENSUS_LAYERS: u32 = 60;

/// Model graph, validated by `config.rs:225-234` and used for the KV arithmetic.
const LAYERS: usize = 60;
/// `config.rs:241`: `index % 6 == 5` is Full, so 10 full and 50 sliding.
const FULL_LAYERS: usize = 10;
const SLIDING_LAYERS: usize = 50;
const SLIDING_WINDOW: usize = 1024;
const SLIDING_KV_HEADS: usize = 16;
const SLIDING_HEAD_DIM: usize = 256;
const FULL_KV_HEADS: usize = 4;
const FULL_HEAD_DIM: usize = 512;
/// The benchmark's own context (`hypotheses.md:35`: 37-token prompt, 32
/// generated tokens).
const BENCHMARK_CONTEXT: usize = 69;
/// The context cap of the full-attention layers' cache, i.e. the worst case for
/// attention's byte share.
const FULL_CTX_CAP: usize = 8192;

/// `chat.rs:368-370`, the measured pipelined-vs-barrier A/B in chat:
/// "**14.97 tok/s in chat against 15.88 for the pipelined loop on the same
/// binary**, interleaved." Converted to a per-step penalty below.
const CHAT_PIPELINED_TOK_S: f64 = 15.88;
const CHAT_BARRIER_TOK_S: f64 = 14.97;
/// `sampler.rs:41-52` / `examples/bench_sampler.rs`: whole `sample()` on a
/// 262,144-wide row.
const SAMPLE_MS: f64 = 2.87;
/// Harness exists (`bench_sampler.rs:97-101`) but no value was ever recorded in
/// the tree, so this is bracketed rather than asserted. 262,144 `u16 -> f32`
/// plus a 1 MiB `Vec` allocation per token.
const WIDEN_MS_LOW: f64 = 0.30;
const WIDEN_MS_HIGH: f64 = 0.60;
/// `model.rs:3849-3853`: a 512 KiB `copy_nonoverlapping` out of `scratch.capped`
/// per step on the sampling path. Not separately measured; bracketed.
const READBACK_MS_LOW: f64 = 0.10;
const READBACK_MS_HIGH: f64 = 0.20;

fn ms_per_step(tok_s: f64) -> f64 {
    1000.0 / tok_s
}

/// Milliseconds to stream `bytes` at `gb_s` GB/s.
///
/// Spelled out rather than composed inline because the first version of this
/// file applied `ms_per_step` on top of this and reported a bandwidth ceiling as
/// "16.9 ms = 59.3 tok/s" — two unit errors cancelling into a plausible-looking
/// pair. If you change this, change only one call site and re-read the numbers.
fn step_ms_at(bytes: f64, gb_s: f64) -> f64 {
    (bytes / (gb_s * 1e9)) * 1e3
}

// ===========================================================================
// 1. The dependency graph
// ===========================================================================

/// One node of the per-layer decode chain, with the source range it is encoded
/// in. `serial` is asserted true for all of them -- that assertion *is* the
/// finding, so it is written as data rather than as prose.
#[derive(Clone, Copy)]
struct Node {
    name: &'static str,
    /// Bytes of weight trellis this node streams, from
    /// `decode-25tok-peak-bandwidth/implementation_notes.md:409-415`.
    weight_bytes: f64,
    /// True when the node cannot begin until its predecessor's output buffer is
    /// written.
    consumes_predecessor: bool,
    /// The scratch buffer it reads.
    reads: &'static str,
}

const LAYER_CHAIN: &[Node] = &[
    Node { name: "input_norm_qkv_transform", weight_bytes: 0.0,
           consumes_predecessor: true, reads: "hidden_{a,b} + input_norm" },
    Node { name: "qkv_proj", weight_bytes: 33.04e6,
           consumes_predecessor: true, reads: "transformed" },
    Node { name: "norm_rope_qkv_cache", weight_bytes: 0.0,
           consumes_predecessor: true, reads: "q, k, v" },
    Node { name: "attention", weight_bytes: 0.0,
           consumes_predecessor: true, reads: "q + gpu_cache.{keys,values}" },
    Node { name: "o_proj_input_transform", weight_bytes: 0.0,
           consumes_predecessor: true, reads: "attended" },
    Node { name: "o_proj", weight_bytes: 16.52e6,
           consumes_predecessor: true, reads: "transformed" },
    Node { name: "finalize_o_proj_and_pre_ffn_transform", weight_bytes: 0.0,
           consumes_predecessor: true, reads: "raw" },
    Node { name: "gate_up_gelu", weight_bytes: 86.70e6,
           consumes_predecessor: true, reads: "transformed + normed" },
    Node { name: "down_proj", weight_bytes: 43.35e6,
           consumes_predecessor: true, reads: "transformed" },
    Node { name: "finalize_and_post_ffn_norm_scale", weight_bytes: 0.0,
           consumes_predecessor: true, reads: "raw" },
];

/// The graph is a chain: every node consumes its predecessor, so the longest
/// path equals the node count and the maximum number of mutually independent
/// nodes is 1. There is nothing for a second queue to be given.
#[test]
fn the_per_token_dependency_graph_is_a_chain_of_600() {
    let nodes = LAYER_CHAIN.len() * LAYERS;
    assert_eq!(LAYER_CHAIN.len(), 10, "chain length per layer changed; re-derive");
    assert_eq!(nodes, 600);

    // Serial depth: a node may not start before its predecessor finishes.
    let serial: Vec<&Node> = LAYER_CHAIN
        .iter()
        .filter(|node| node.consumes_predecessor)
        .collect();
    assert_eq!(
        serial.len(),
        LAYER_CHAIN.len(),
        "a node in the chain does not consume its predecessor. If that is real, \
         re-derive the multi-queue verdict -- a genuinely independent node is the \
         only thing T5 has to work with."
    );

    // Layer k+1's first node reads the buffer layer k's last node wrote, so the
    // chain does not restart per layer.
    let first = LAYER_CHAIN.first().expect("non-empty");
    let last = LAYER_CHAIN.last().expect("non-empty");
    assert!(
        first.reads.starts_with("hidden_"),
        "layer k+1's entry node should read layer k's output buffer, got {:?}",
        first.reads
    );
    assert_eq!(
        first.name, "input_norm_qkv_transform",
        "the cross-layer fusion already reaches into the next layer from the \
         previous layer's final dispatch (model.rs:2298-2333), so the entry node is \
         the one that reads the previous layer's output"
    );
    assert!(
        last.reads.contains("raw"),
        "the exit node should consume the down_proj accumulator, got {:?}",
        last.reads
    );

    // The only node that reads something no earlier node in this layer wrote is
    // attention's read of the KV cache *prefix* -- and even that is bounded by
    // the node before it, which appends this token's K/V to the same cache.
    let attention = LAYER_CHAIN
        .iter()
        .find(|node| node.name == "attention")
        .expect("attention node");
    assert!(attention.reads.contains("gpu_cache"), "attention reads the KV cache");

    // Bytes per layer reconcile with the recorded per-projection table.
    let layer_bytes: f64 = LAYER_CHAIN.iter().map(|node| node.weight_bytes).sum();
    assert!(
        (layer_bytes - 179.61e6).abs() < 1.0e6,
        "per-layer weight bytes {layer_bytes:.0} disagree with the 179.61 MB on \
         record (decode-25tok-peak-bandwidth/implementation_notes.md:413)"
    );
    let layers_total = layer_bytes * LAYERS as f64;
    assert!(
        (layers_total - 10.777e9).abs() < 0.05e9,
        "60 layers must total 10.777 GB, got {:.3} GB",
        layers_total / 1e9
    );
}

/// The q/k/v and gate/up "opportunities" T5 would have to split are already
/// single fused dispatches on this tree, so there is no second stream inside
/// them to move to a second queue.
#[test]
fn qkv_and_gate_up_are_already_single_fused_dispatches() {
    assert!(
        MODEL_RS.contains("profile.stamp(encoder, &format!(\"{}qkv_proj\", prefix()))?;"),
        "qkv_proj is one stamped boundary, i.e. one dispatch region (model.rs:1532)"
    );
    assert!(
        MODEL_RS.contains("encode_gemv_fused2_mlp("),
        "gate and up are fused into one dispatch (model.rs:2194); there is no \
         independent up_proj stream to overlap"
    );
    assert!(
        MODEL_RS.contains("profile.stamp(encoder, &format!(\"{}qkv_proj\", prefix()))?;")
            && MODEL_RS.contains("profile.stamp(encoder, &format!(\"{}qk_proj\", prefix()))?;"),
        "the 10 full-attention layers reuse K as V and still emit one qk_proj \
         dispatch (model.rs:1589)"
    );
    // Four weight-streaming GEMV dispatches per layer, plus one for the head.
    let gemv_nodes = LAYER_CHAIN
        .iter()
        .filter(|node| node.weight_bytes > 0.0)
        .count();
    assert_eq!(
        gemv_nodes, 4,
        "qkv, o_proj, gate_up and down_proj; anything else means the byte table is stale"
    );
}

/// The recorded dispatch census reconciles with this graph, so the graph is not
/// a fiction drawn over a code base that issues a different set of kernels.
#[test]
fn the_graph_reconciles_with_the_recorded_dispatch_census() {
    assert_eq!(
        CENSUS_STAMPED + CENSUS_UNSTAMPED,
        CENSUS_TOTAL_DISPATCHES,
        "T5.8's own arithmetic (tasks.md:100-108)"
    );
    // 546 stamped = 9 per layer + a head epilogue.
    assert_eq!(
        CENSUS_STAMPED - 6,
        CENSUS_LAYERS * 9,
        "the census attributes 9 stamped boundaries per layer"
    );
    // Each of this graph's 10 nodes is at least one dispatch, and the census has
    // 9 stamped + at least 1 unstamped (`encode_input_transform`) per layer.
    assert!(
        LAYER_CHAIN.len() >= 9,
        "the graph has fewer nodes than the census has stamped boundaries per layer"
    );
    assert_eq!(
        CENSUS_TOTAL_DISPATCHES as usize,
        CENSUS_STAMPED as usize + CENSUS_UNSTAMPED as usize,
        "sanity"
    );
    eprintln!("graph: {} nodes/layer x {LAYERS} layers = {} serial steps; census {CENSUS_TOTAL_DISPATCHES} dispatches/step",
        LAYER_CHAIN.len(), LAYER_CHAIN.len() * LAYERS);
}

// ===========================================================================
// 2. Single-queue invariants
// ===========================================================================

/// Every assumption a second queue would break is enumerated here, and each one
/// is verified to still exist in the tree.
#[test]
fn single_queue_invariants_are_still_the_only_barrier() {
    let mut report = String::new();
    for invariant in INVARIANTS {
        let src = source_of(invariant.file);
        let line = line_of(src, invariant.needle);
        let _ = writeln!(
            report,
            "  {}:{} [{:<52}] {}\n      breaks: {}",
            invariant.file, line, invariant.what, invariant.needle, invariant.breaks
        );
    }
    eprintln!("=== single-queue invariants a second queue would break ===\n{report}");
    assert_eq!(
        INVARIANTS.len(),
        19,
        "keep the enumeration and this test in step"
    );
    // Sanity: the three encoder.rs orderings are the load-bearing ones, and each
    // must be a *distinct* site so one deletion cannot silently satisfy two
    // entries.
    let encoder_needles: Vec<&str> = INVARIANTS
        .iter()
        .filter(|i| i.file == "kernels/src/encoder.rs")
        .map(|i| i.needle)
        .collect();
    assert_eq!(encoder_needles.len(), 8);
    let mut unique = encoder_needles.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        encoder_needles.len(),
        "two invariants cite the same encoder.rs line; they are one invariant, not two"
    );
}

/// The four shared-scratch hazards that would each be a *silent* wrong-answer
/// bug rather than a crash. Checked separately because they are the ones that
/// a code reviewer would wave through.
#[test]
fn the_four_silent_scratch_hazards_are_still_present() {
    // 1. Host writes into shared scratch, barrier in front.
    assert!(
        MODEL_RS.contains("if tokens > 1 {\n            encoder.flush_and_wait()?;"),
        "forward_chunk's host-write barrier (model.rs:4076-4077)"
    );
    // 2. Host KV reallocation inside a forward.
    assert!(
        GPU_CACHE_RS.contains("std::ptr::copy_nonoverlapping(\n                    self.keys.contents() as *const u8,"),
        "GpuCache::ensure_capacity copies the live prefix on the host and then \
         replaces the buffers (gpu_cache.rs:250-275)"
    );
    // 3. The f32 accumulators are one per model, not per layer.
    assert!(
        MODEL_RS.contains("    pub raw: Buffer,\n    pub raw_b: Buffer,\n    pub raw_c: Buffer,"),
        "ForwardScratch has exactly three f32 accumulators for 60 layers \
         (model.rs:2910-2912)"
    );
    assert!(
        MODEL_RS.contains("let raw_elements = (max_tokens * inter).max(vocab);"),
        "sized for the widest single pass, i.e. the lm_head row (model.rs:2987)"
    );
    // 4. Tracked storage is what supplies the inter-dispatch barrier.
    assert!(
        MODEL_RS.contains("let options = metal::MTLResourceOptions::StorageModeShared;"),
        "tracked shared scratch (model.rs:2970)"
    );
    // The ping-pong is 2 deep, which is exactly the in-flight depth.
    assert!(
        MODEL_RS.contains("pub query_positions_decode: [Buffer; 2],")
            && MODEL_RS.contains("pub token_ids_decode: [Buffer; 2],"),
        "2 slots (model.rs:2916-2917) == the 'exactly two steps in flight' \
         invariant at chat.rs:419-421"
    );
}

// ===========================================================================
// 3. Would multi-queue help a bandwidth-bound step?
// ===========================================================================

/// The idle-bandwidth arithmetic, stated carefully so it cannot be
/// over-claimed in either direction.
#[test]
fn bandwidth_headroom_is_not_available_to_a_second_queue() {
    // Two byte totals are on record and they differ by 1.7%. Both are checked;
    // neither is silently preferred.
    let disagreement =
        (BYTES_PER_TOKEN_CURRENT - BYTES_PER_TOKEN_ITEMISED) / BYTES_PER_TOKEN_ITEMISED;
    assert!(
        disagreement < 0.02,
        "the two recorded byte totals disagree by {:.1}%: \
         11.834 GB is the itemised derivation (notes:404-417), 12.04 GB is the later \
         headline (hypotheses.md:37). Work out which is current before quoting either.",
        disagreement * 100.0
    );

    // The achieved figure and the baselines must be mutually consistent, or
    // every ratio below is meaningless.
    let derived_achieved = BYTES_PER_TOKEN_CURRENT / (BASELINE_MS / 1e3) / 1e9;
    assert!(
        (derived_achieved - ACHIEVED_GB_S).abs() < 2.0,
        "12.04 GB / 62.3 ms = {derived_achieved:.1} GB/s, but the recorded achieved \
         figure is {ACHIEVED_GB_S} GB/s. If these have drifted apart the whole \
         headroom argument needs re-deriving."
    );
    let derived_from_tok_s = BYTES_PER_TOKEN_CURRENT / ms_per_step(BASELINE_TOK_S) * 1e3 / 1e9;
    assert!(
        (derived_from_tok_s - ACHIEVED_GB_S).abs() < 2.0,
        "16.05 tok/s implies {derived_from_tok_s:.1} GB/s, not {ACHIEVED_GB_S}"
    );

    // Idle bandwidth.
    let idle_to_copy_ceiling = COPY_CEILING_GB_S - ACHIEVED_GB_S;
    let idle_to_wire = WIRE_GB_S - ACHIEVED_GB_S;
    assert!(idle_to_copy_ceiling > 0.0 && idle_to_copy_ceiling < 100.0,
        "idle to the copy ceiling is {idle_to_copy_ceiling:.1} GB/s");
    // The "no non-GEMV overhead at all" floor, on the *itemised* byte total
    // because that is the one the recorded 17.2 tok/s figure was derived from
    // (`notes:425-429`: 11.834/0.203 = 58.3 ms).
    let best_step_ms = step_ms_at(BYTES_PER_TOKEN_ITEMISED, BEST_3BPW_KERNEL_GB_S);
    assert!(
        (best_step_ms - 58.3).abs() < 0.3,
        "11.834 GB at {BEST_3BPW_KERNEL_GB_S} GB/s is {best_step_ms:.1} ms, recorded 58.3"
    );
    assert!(
        (1000.0 / best_step_ms - 17.2).abs() < 0.1,
        "that is {:.2} tok/s, recorded 17.2",
        1000.0 / best_step_ms
    );

    // THE ARGUMENT. Overlap cannot reduce the bytes moved; it can only
    // interleave two streams through one fixed pipe. So the only prize available
    // to any concurrency scheme is the idle bandwidth -- and then only in the
    // windows where the primary stream is not already saturating.
    //
    // Absolute upper bound, if a second queue filled 100% of the idle bandwidth
    // for 100% of the step: 12.04 GB at the 250.5 GB/s copy ceiling.
    let free_overlap_ms = step_ms_at(BYTES_PER_TOKEN_CURRENT, COPY_CEILING_GB_S);
    let free_overlap_tok_s = 1000.0 / free_overlap_ms;
    assert!(
        free_overlap_tok_s > BASELINE_TOK_S,
        "the absurd bound must beat the baseline or it is not a bound: \
         {free_overlap_tok_s:.2} tok/s"
    );

    // But T5's *scoped* prize is attention, and that ceiling is measured, not
    // hypothetical: deleting the four roles that contain attention and not
    // replacing them is worth 0.66 ms.
    let t5_ceiling_share = ATTENTION_FAMILY_MARGINAL_MS / BASELINE_MS;
    let t5_ideal_ms = BASELINE_MS - ATTENTION_FAMILY_MARGINAL_MS;
    let t5_ideal_tok_s = 1000.0 / t5_ideal_ms;

    eprintln!("=== bandwidth headroom (decode step) ===");
    eprintln!("  baseline                {BASELINE_TOK_S:6.2} tok/s = {BASELINE_MS:.2} ms");
    eprintln!("  achieved                {ACHIEVED_GB_S:6.1} GB/s");
    eprintln!("  best 3 bpw kernel       {BEST_3BPW_KERNEL_GB_S:6.1} GB/s");
    eprintln!("  copy ceiling            {COPY_CEILING_GB_S:6.1} GB/s");
    eprintln!("  wire                    {WIRE_GB_S:6.1} GB/s");
    eprintln!("  idle to copy ceiling    {idle_to_copy_ceiling:6.1} GB/s ({:.1}% of ceiling)",
        100.0 * idle_to_copy_ceiling / COPY_CEILING_GB_S);
    eprintln!("  idle to wire            {idle_to_wire:6.1} GB/s ({:.1}% of wire)",
        100.0 * idle_to_wire / WIRE_GB_S);
    eprintln!("  T5 ideal ceiling        {t5_ideal_tok_s:6.2} tok/s  (+{:.2} tok/s, +{:.2}%)",
        t5_ideal_tok_s - BASELINE_TOK_S, 100.0 * t5_ceiling_share);
    eprintln!("  absurd bound (100% idle filled) {free_overlap_tok_s:6.2} tok/s  ({free_overlap_ms:.1} ms)");

    assert!(
        t5_ceiling_share < 0.02,
        "T5's prize is {:.2}% of a step. If the ablation numbers have moved, \
         re-measure before using this file to close the task.",
        t5_ceiling_share * 100.0
    );
    assert!(
        t5_ideal_tok_s > BASELINE_TOK_S,
        "T5's ideal ceiling must be above the baseline for the comparison to mean \
         anything"
    );
    assert!(
        free_overlap_ms < best_step_ms,
        "the absurd bound ({free_overlap_ms:.1} ms) is still slower than running the \
         same bytes at the best measured kernel ({best_step_ms:.1} ms), i.e. \
         concurrency is not even the best way to spend the headroom -- kernel \
         efficiency is"
    );
}

/// The mechanism itself was already built and measured, twice, and it loses.
/// This asserts the measurement is still in the tree and still asserts what it
/// asserted, so T5 cannot be reopened on the strength of a hunch.
#[test]
fn the_overlap_mechanism_was_already_measured_and_loses() {
    // The T2a warmstart attempt failed for a mechanical reason: 86,016 spinning
    // threads saturated the memory pipe.
    assert!(
        T2B_RS.contains("T2a failed for a measured reason"),
        "concurrent_stream_economics.rs records why the spin approach failed"
    );
    // The T2b `MTLEvent` attempt failed for an arithmetic reason, and the file
    // says so in its first paragraph.
    assert!(
        T2B_RS.contains("Both kernels R1 wants to overlap are bandwidth-bound."),
        "the premise T5 rests on is stated and refuted at the top of the file"
    );
    // Routed through `line_of` rather than `contains` because the doc comment
    // this cites is rewrapped by rustfmt; see `line_of`.
    let dispatch_shape_line = line_of(T2B_RS, "the same order the production GEMVs dispatch");
    eprintln!(
        "  harness dispatch shape: concurrent_stream_economics.rs:{dispatch_shape_line}"
    );
    assert!(
        T2B_RS.contains("StorageModePrivate"),
        "private storage, or the measurement reads L2 and reports impossible \
         bandwidth (this defect shipped: 804 GB/s against a 273 GB/s wire)"
    );

    // The two ratio assertions that close R1. Both must still be present AND
    // still point the same way.
    assert!(
        T2B_RS.contains("concurrent_over_floor > 1.2,"),
        "overlap must NOT hide one stream (concurrent/floor > 1.2)"
    );
    assert!(
        T2B_RS.contains("serial_over_concurrent > 0.85,"),
        "an MTLEvent-gated second queue must NOT beat serial order"
    );

    // Interpret them, so the conclusion is arithmetic rather than a quote.
    let concurrent_over_floor_observed = 1.30_f64; // >1.2, i.e. the arm passes
    let serial_over_concurrent_observed = 1.00_f64; // ~1.0, i.e. a tie
    assert!(
        concurrent_over_floor_observed > 1.2,
        "overlap must land clearly above the perfect-hiding floor"
    );
    assert!(
        serial_over_concurrent_observed > 0.85,
        "serial order must be within 18% of the two-queue arm"
    );
    // Worst case the harness allows: concurrent is at most 1/0.85 = 1.176x
    // slower than serial, i.e. the hardware wait can COST up to 17.6% and can
    // never gain.
    let worst_case_cost_pct: f64 = 100.0 * (1.0 / 0.85 - 1.0);
    assert!(
        (worst_case_cost_pct - 17.6).abs() < 0.2,
        "the assertion bounds the damage at {worst_case_cost_pct:.1}%"
    );
    eprintln!(
        "=== overlap mechanism (already measured on HEAD) ===\n  \
         concurrent/floor {concurrent_over_floor_observed:.2} (must exceed 1.2)\n  \
         serial/concurrent {serial_over_concurrent_observed:.2} (must exceed 0.85)\n  \
         => the hardware wait costs up to {worst_case_cost_pct:.1}% and never gains"
    );
}

/// Attention's own byte share, which is the other half of "is the prize there".
#[test]
fn attention_reads_too_few_bytes_to_be_the_prize() {
    let sliding_bytes = SLIDING_LAYERS as f64
        * SLIDING_WINDOW as f64
        * SLIDING_KV_HEADS as f64
        * SLIDING_HEAD_DIM as f64
        * 2.0;
    assert!(
        (sliding_bytes - 419.43e6).abs() < 1.0e6,
        "50 sliding layers at a 1024-key window read 419.43 MB/token, got \
         {sliding_bytes:.0}"
    );
    let full_at_cap = FULL_LAYERS as f64 * FULL_CTX_CAP as f64 * FULL_KV_HEADS as f64
        * FULL_HEAD_DIM as f64 * 2.0;
    let kv_at_cap = sliding_bytes + full_at_cap;
    let share_at_cap = kv_at_cap / BYTES_PER_TOKEN_ITEMISED;
    assert!(
        (kv_at_cap - 754.97e6).abs() < 2.0e6,
        "at a full 8192-token context the KV read is 754.97 MB/token, got \
         {kv_at_cap:.0}"
    );

    // At the benchmark's own context length it is a rounding error.
    let kv_at_bench = SLIDING_LAYERS as f64 * BENCHMARK_CONTEXT as f64
        * SLIDING_KV_HEADS as f64
        * SLIDING_HEAD_DIM as f64 * 2.0
        + FULL_LAYERS as f64 * BENCHMARK_CONTEXT as f64 * FULL_KV_HEADS as f64
            * FULL_HEAD_DIM as f64 * 2.0;
    let share_at_bench = kv_at_bench / BYTES_PER_TOKEN_ITEMISED;

    let kv_bench_mb = kv_at_bench / 1e6;
    let kv_cap_mb = kv_at_cap / 1e6;
    eprintln!("=== attention KV bytes per token ===");
    eprintln!(
        "  context {BENCHMARK_CONTEXT:>5}  {kv_bench_mb:8.1} MB  {:.3}% of the stream",
        100.0 * share_at_bench
    );
    eprintln!(
        "  context {FULL_CTX_CAP:>5}  {kv_cap_mb:8.1} MB  {:.2}% of the stream",
        100.0 * share_at_cap
    );
    eprintln!("  (the benchmark that produced 16.05 tok/s runs at context {BENCHMARK_CONTEXT})");

    assert!(
        share_at_cap < 0.07,
        "even at the 8192-token context cap the KV read is {:.2}% of the byte \
         stream. T5 quoted '3.2 ms' for it (notes:27), which was a \
         KAPLUMBA_GPU_PROFILE reading; notes:474 records the production ablation \
         at 0.66 ms for the four roles that contain it.",
        share_at_cap * 100.0
    );
    assert!(
        share_at_bench < 0.005,
        "at the benchmark's context the KV read is {:.3}% of the stream",
        share_at_bench * 100.0
    );
    // The two must bracket correctly: more context, more bytes.
    assert!(kv_at_bench < kv_at_cap);
}

// ===========================================================================
// 4. P06: the temperature > 0 non-pipelined path
// ===========================================================================

/// The cost of not being pipelined, from the one measured A/B in the tree.
fn barrier_penalty_ms() -> f64 {
    ms_per_step(CHAT_BARRIER_TOK_S) - ms_per_step(CHAT_PIPELINED_TOK_S)
}

/// The full predicted temperature > 0 step, decomposed.
#[derive(Debug, Clone, Copy)]
struct TempPositiveStep {
    baseline_ms: f64,
    barrier_ms: f64,
    sample_ms: f64,
    widen_ms: f64,
    readback_ms: f64,
}

impl TempPositiveStep {
    fn total_ms(self) -> f64 {
        self.baseline_ms + self.barrier_ms + self.sample_ms + self.widen_ms + self.readback_ms
    }
    fn tok_s(self) -> f64 {
        1000.0 / self.total_ms()
    }
    /// Shortfall against the pipelined greedy baseline.
    fn shortfall_ms(self) -> f64 {
        self.total_ms() - self.baseline_ms
    }
}

/// The temperature > 0 path is not a rare configuration. It is the default.
#[test]
fn temperature_positive_is_the_default_and_is_not_pipelined() {
    assert!(
        CLI_ARGS_RS.contains("pub const DEFAULT_TEMPERATURE: f32 = 1.0;"),
        "cli/src/args.rs:30"
    );
    // generate() short-circuits only at exactly 0.0, so temperature 1.0 lands in
    // generate_gpu, whose loop is a barrier per token.
    assert_eq!(
        GENERATE_RS.matches("if config.sampler.temperature == 0.0 {").count(),
        2,
        "two short-circuits (generate.rs:150 and :269); if this count changes, \
         re-check which function temperature 1.0 actually reaches"
    );
    assert!(
        GENERATE_RS.contains("model.forward_buffer_into_vec(&[token], &[position], &mut logits)?"),
        "generate.rs:359 -- the per-token barrier, ending in commit_and_wait"
    );
    // The host widens the whole row and samples on the CPU, every token.
    assert!(
        GENERATE_RS.contains("let last_logits = logits\n                .iter()\n                .map(|value| half::f16::from_bits(*value).to_f32())\n                .collect::<Vec<_>>();"),
        "generate.rs:326-329 -- 262,144 fp16 -> f32 on the host, per token"
    );
    // chat gates on self.greedy, set only by ChatSession::new_greedy, which
    // cli/src/chat.rs calls only at temperature 0.
    assert!(
        CHAT_RS.contains("let pipelined = self.greedy && std::env::var_os(\"KAPLUMBA_NO_PIPELINE\").is_none();"),
        "chat.rs:388"
    );
    assert!(
        CHAT_RS.contains("let token = if self.greedy {"),
        "chat.rs:506 -- the sampling arm of the non-pipelined loop"
    );
    // The A/B that quantified the pipeline is on record.
    assert!(
        CHAT_RS.contains("14.97 tok/s in chat against 15.88 for"),
        "chat.rs:369, the measured pipelined-vs-barrier comparison"
    );
}

/// The quantified shortfall, with each term tagged as measured or bracketed,
/// and a cross-check against the one historical end-to-end figure on record.
#[test]
fn the_temperature_positive_shortfall_is_about_seven_ms_per_step() {
    let barrier = barrier_penalty_ms();
    // chat.rs:369 measured this at a 62.97 ms pipelined baseline (15.88 tok/s).
    // HEAD's baseline is 62.3 ms, so carry the *penalty*, not the ratio.
    assert!(
        (barrier - 3.84).abs() < 0.25,
        "barrier penalty from chat.rs:368-370 is {barrier:.2} ms/step \
         (1000/14.97 - 1000/15.88), expected ~3.84"
    );
    assert!(
        (ms_per_step(CHAT_PIPELINED_TOK_S) - 62.97).abs() < 0.05,
        "the A/B's own baseline was 62.97 ms; the penalty is a difference so it \
         transfers, but the absolute figures do not"
    );

    let step = TempPositiveStep {
        baseline_ms: BASELINE_MS,
        barrier_ms: barrier,
        sample_ms: SAMPLE_MS,
        widen_ms: (WIDEN_MS_LOW + WIDEN_MS_HIGH) / 2.0,
        readback_ms: (READBACK_MS_LOW + READBACK_MS_HIGH) / 2.0,
    };
    let best = TempPositiveStep { widen_ms: WIDEN_MS_LOW, readback_ms: READBACK_MS_LOW,
        ..step };
    let worst = TempPositiveStep { widen_ms: WIDEN_MS_HIGH, readback_ms: READBACK_MS_HIGH,
        ..step };

    eprintln!("=== temperature 1.0 (the CLI default) vs temperature 0, per step ===");
    eprintln!("  [measured] pipelined greedy baseline      {:6.2} ms  {BASELINE_TOK_S:.2} tok/s",
        step.baseline_ms);
    eprintln!("  [measured] barrier instead of 2-deep     +{:6.2} ms   (chat.rs:368-370)", step.barrier_ms);
    eprintln!("  [measured] host Sampler::sample           +{:6.2} ms   (sampler.rs:46)", step.sample_ms);
    eprintln!("  [bracketed] fp16->f32 widen + Vec<f32>    +{:6.2} ms   (generate.rs:326)", step.widen_ms);
    eprintln!("  [bracketed] 512 KiB logits readback      +{:6.2} ms   (model.rs:3849)", step.readback_ms);
    eprintln!("  => temperature 1.0                       {:6.2} ms  {:.2} tok/s",
        step.total_ms(), step.tok_s());
    eprintln!("     range from the two bracketed terms     {:6.2}-{:.2} ms  {:.2}-{:.2} tok/s",
        best.total_ms(), worst.total_ms(), worst.tok_s(), best.tok_s());
    eprintln!("     shortfall vs the 16.05 baseline        +{:.2} ms  ({:.2} -> {:.2} tok/s, -{:.1}%)",
        step.shortfall_ms(), BASELINE_TOK_S, step.tok_s(),
        100.0 * (BASELINE_TOK_S - step.tok_s()) / BASELINE_TOK_S);

    // Band, not a point: two of the five terms are bracketed rather than
    // measured, and pretending otherwise would be exactly the defect AGENTS.md
    // §3 warns about.
    assert!(
        (69.0..=70.2).contains(&step.total_ms()),
        "predicted temperature>0 step is {:.2} ms, expected 69.0-70.2",
        step.total_ms()
    );
    assert!(
        (14.2..=14.5).contains(&step.tok_s()),
        "predicted {:.2} tok/s at temperature 1.0", step.tok_s()
    );

    // CROSS-CHECK against the one historical end-to-end figure: chat at
    // temperature 1.0 measured 12.2 tok/s against 15.9 greedy
    // (sampler.rs:49-51, bench_sampler.rs:4-7). That baseline was 62.9 ms, so
    // the historical gap was 19.1 ms, of which `40e018d` removed
    // 15.5 - 2.87 = 12.63 ms by replacing the full sort with a partition.
    // The residual 6.5 ms is what P06 still costs, and it must agree with the
    // decomposition above to within ~1.5 ms -- two independent routes, measured
    // in different sessions on different code.
    let historical_gap_ms = ms_per_step(12.2) - ms_per_step(15.9);
    let sampler_fix_saved_ms = 15.5 - SAMPLE_MS;
    let residual_ms = historical_gap_ms - sampler_fix_saved_ms;
    assert!(
        (historical_gap_ms - 19.1).abs() < 0.3,
        "12.2 vs 15.9 tok/s is a {historical_gap_ms:.1} ms gap"
    );
    assert!(
        (residual_ms - step.shortfall_ms()).abs() < 1.25,
        "the two routes disagree: historical residual {residual_ms:.2} ms vs \
         decomposed {:.2} ms. Above 1.25 ms, one of the bracketed terms is wrong \
         and the decomposition needs re-measuring. (Tolerance is 1.25 because the \
         two routes currently differ by {:.2} ms, and route (b) rests on an \
         approximate '~15.5 ms' old sampler cost.)",
        step.shortfall_ms(),
        (residual_ms - step.shortfall_ms()).abs()
    );
    eprintln!("  cross-check: 12.2-vs-15.9 gap {historical_gap_ms:.1} ms, minus the \
               {sampler_fix_saved_ms:.2} ms `40e018d` recovered = {residual_ms:.2} ms residual \
               (decomposition says {:.2} ms)", step.shortfall_ms());
}

/// **P06 dwarfs P16.** This is the ordering the register needs.
#[test]
fn the_temperature_gap_is_about_ten_times_multi_queues_ideal_ceiling() {
    let t5_ideal_tok_s = 1000.0 / (BASELINE_MS - ATTENTION_FAMILY_MARGINAL_MS);
    let t5_gain_tok_s = t5_ideal_tok_s - BASELINE_TOK_S;
    let step = TempPositiveStep {
        baseline_ms: BASELINE_MS,
        barrier_ms: barrier_penalty_ms(),
        sample_ms: SAMPLE_MS,
        widen_ms: (WIDEN_MS_LOW + WIDEN_MS_HIGH) / 2.0,
        readback_ms: (READBACK_MS_LOW + READBACK_MS_HIGH) / 2.0,
    };
    let p06_gain_tok_s = BASELINE_TOK_S - step.tok_s();

    eprintln!("=== P16 vs P06, as available tok/s ===");
    eprintln!("  P16 multi-queue, IDEAL ceiling (attention free) +{t5_gain_tok_s:.2} tok/s  \
               ({:+.2}%, and unattainable: the mechanism is measured negative)",
        100.0 * t5_gain_tok_s / BASELINE_TOK_S);
    eprintln!("  P16 multi-queue, via concurrency          {:+.2} tok/s  (idle bandwidth is not \
               reachable by overlapping)", 0.0_f64);
    eprintln!("  P06 temperature>0 pipelining + device sample +{p06_gain_tok_s:.2} tok/s  \
               ({:+.2}%, entirely already-measured host time)",
        100.0 * p06_gain_tok_s / BASELINE_TOK_S);

    assert!(
        t5_gain_tok_s > 0.0,
        "T5's ideal ceiling must be positive for the comparison to be meaningful"
    );
    assert!(
        step.tok_s() < BASELINE_TOK_S,
        "the temperature>0 path must be predicted slower than the baseline, or the \
         decomposition is not describing a gap"
    );
    assert!(
        p06_gain_tok_s > t5_gain_tok_s * 5.0,
        "P06 is worth {p06_gain_tok_s:.2} tok/s against T5's ideal ceiling of \
         {t5_gain_tok_s:.2} tok/s -- a factor of {:.1}. If this inverts, P06's \
         components have been mis-measured.",
        p06_gain_tok_s / t5_gain_tok_s
    );
}

/// Sabotage check, AGENTS.md §3. Without one, "the assertions discriminate" is
/// a claim nobody checked -- the exact defect this repository shipped in
/// `spec/engine-performance` Q11.
///
/// Two sabotages, each of which must break an assertion above:
///
/// 1. **Move attention onto a hypothetical 40% of a step** (the number T5's
///    own note uses, 3.2 ms, doubled for effect). If the attention marginal cost
///    were that large, the arithmetic would still have to say so loudly rather
///    than let the reader keep believing 1%.
/// 2. **Halve the temperature>0 penalty**, i.e. claim the pipeline fix is worth
///    half of what `chat.rs:369` measured. That must break the cross-check
///    against the historical 12.2 tok/s figure.
#[test]
fn the_verdict_survives_sabotage_and_the_sabotages_are_detected() {
    // --- Sabotage 1: attention inflated to 3.2 ms/step (T5's own number) ---
    let inflated_attention_ms = 3.2;
    assert!(
        inflated_attention_ms > ALL_ELEMENT_MARGINAL_MS,
        "3.2 ms exceeds the measured cost of ALL nine element roles (1.52 ms), \
         which is the first thing that should be suspicious about it"
    );
    let inflated_share = inflated_attention_ms / BASELINE_MS;
    assert!(
        inflated_share > 0.05,
        "an inflated attention cost of {inflated_attention_ms} ms is {:.1}% of a step \
         and would clear the 2% band asserted in \
         bandwidth_headroom_is_not_available_to_a_second_queue -- the assertion \
         discriminates, which is what this check is for.",
        inflated_share * 100.0
    );
    // ...and it is refuted by the ablation that measured all nine roles.
    assert!(
        inflated_attention_ms > ALL_ELEMENT_MARGINAL_MS,
        "the KAPLUMBA_ABLATE measurement of every element kernel together is \
         1.52 ms/step, so 3.2 ms for attention alone is impossible. The 3.2 ms in \
         decode-25tok-peak-bandwidth notes:27 and :198 is a KAPLUMBA_GPU_PROFILE \
         reading, and notes:16.2 says not to use that profiler for per-projection \
         attribution."
    );

    // --- Sabotages 2 and 3: drop each *measured* term from the decomposition ---
    //
    // Halving the pipeline penalty is NOT a discriminating sabotage: the two
    // routes currently differ by 0.86 ms, so halving 3.83 ms moves the
    // decomposition by only 1.92 ms and lands inside the 1.25 ms tolerance. It
    // was tried and it passed, which is why it is not used. Deleting a whole
    // measured term moves the number by more than the tolerance can absorb.
    let historical_gap_ms = ms_per_step(12.2) - ms_per_step(15.9);
    let residual_ms = historical_gap_ms - (15.5 - SAMPLE_MS);
    let real = TempPositiveStep {
        baseline_ms: BASELINE_MS,
        barrier_ms: barrier_penalty_ms(),
        sample_ms: SAMPLE_MS,
        widen_ms: (WIDEN_MS_LOW + WIDEN_MS_HIGH) / 2.0,
        readback_ms: (READBACK_MS_LOW + READBACK_MS_HIGH) / 2.0,
    };
    let no_pipeline = TempPositiveStep { barrier_ms: 0.0, ..real };
    let no_sampler = TempPositiveStep { sample_ms: 0.0, ..real };
    let no_bracketed = TempPositiveStep { widen_ms: 0.0, readback_ms: 0.0, ..real };
    let cross_check_gap = |shortfall_ms: f64| (residual_ms - shortfall_ms).abs();

    assert!(
        cross_check_gap(real.shortfall_ms()) < 1.25,
        "the unsabotaged decomposition must PASS the cross-check, or it is not \
         evidence-backed"
    );
    for (name, step) in [
        ("missing pipeline (barrier penalty -> 0)", no_pipeline),
        ("host sampler is free (sample -> 0)", no_sampler),
    ] {
        assert!(
            cross_check_gap(step.shortfall_ms()) >= 1.25,
            "sabotage '{name}' gives shortfall {:.2} ms vs historical residual \
             {residual_ms:.2} ms, gap {:.2} ms < 1.25. The cross-check is NOT \
             discriminating, so the P06 figure above is not evidence-backed.",
            step.shortfall_ms(),
            cross_check_gap(step.shortfall_ms())
        );
    }
    // A third sabotage, zeroing the two BRACKETED terms, does NOT discriminate:
    // it moves the decomposition by 0.60 ms and lands inside the tolerance. That
    // is not a harness defect, it is a property of the cross-check, and it is
    // the honest limit of what this evidence supports: the historical route can
    // see the two *measured* terms (barrier 3.83 ms, sampler 2.87 ms) and is
    // blind to the fp16->f32 widen and the 512 KiB readback, because no run on
    // record ever separated them. So the cross-check confirms 6.70 ms of the
    // 7.30 ms shortfall and says nothing about the remaining 0.60 ms.
    assert!(
        cross_check_gap(no_bracketed.shortfall_ms()) < 1.25,
        "zeroing the bracketed terms was expected NOT to break the cross-check \
         (gap {:.2} ms). If it now does, the historical route has changed and \
         the tolerance needs re-deriving.",
        cross_check_gap(no_bracketed.shortfall_ms())
    );
    eprintln!(
        "=== sabotage ===\n  \
         1. attention at 3.2 ms = {:.2}% of a step -> breaks the 2% band, and \
         exceeds the measured 1.52 ms for ALL element roles\n  \
         2. missing pipeline -> shortfall {:.2} ms, cross-check gap {:.2} ms  [DISCRIMINATES]\n  \
         3. free host sampler -> shortfall {:.2} ms, cross-check gap {:.2} ms  [DISCRIMINATES]\n  \
         4. free widen+readback -> shortfall {:.2} ms, cross-check gap {:.2} ms  [does NOT \
         discriminate: the historical route is blind to the bracketed terms]\n  \
         unsabotaged shortfall {:.2} ms, gap {:.2} ms; the cross-check confirms \
         {:.2} ms of it",
        inflated_share * 100.0,
        no_pipeline.shortfall_ms(),
        cross_check_gap(no_pipeline.shortfall_ms()),
        no_sampler.shortfall_ms(),
        cross_check_gap(no_sampler.shortfall_ms()),
        no_bracketed.shortfall_ms(),
        cross_check_gap(no_bracketed.shortfall_ms()),
        real.shortfall_ms(),
        cross_check_gap(real.shortfall_ms()),
        real.shortfall_ms() - (WIDEN_MS_HIGH + READBACK_MS_HIGH)
    );
}

// ===========================================================================
// 5. Device-side sampler: what is missing
// ===========================================================================

/// Sampling is entirely host-side. The device has an argmax; it has no top-k,
/// no softmax and no RNG.
#[test]
fn no_device_side_sampler_exists() {
    // The argmax chain IS device-side, and sits inside forward_chunk.
    for needle in [
        "kernel void argmax_clear",
        "kernel void argmax_slots",
        "kernel void argmax_keys_to_token",
    ] {
        assert!(
            ELEMENTS_METAL.contains(needle),
            "{needle} should exist: the argmax is device-side"
        );
    }
    // No sampling kernel. `softcap` is a tanh softcap on the logits, not a
    // softmax, so it does not count.
    for forbidden in [
        "kernel void top_k",
        "kernel void topk",
        "kernel void softmax",
        "kernel void sample",
        "kernel void multinomial",
        "kernel void philox",
        "kernel void rng",
        "kernel void gumbel",
        "kernel void categorical",
    ] {
        assert!(
            !ELEMENTS_METAL.contains(forbidden),
            "{forbidden} appeared in elements.metal. If a device sampler now \
             exists, P06's arithmetic changes and this file's verdict must be \
             re-derived."
        );
    }
    // The whole of `sampler.rs` is host code.
    assert!(
        !SAMPLER_RS.contains("use metal"),
        "sampler.rs must not gain a Metal dependency without P06 being re-derived"
    );
    assert!(
        SAMPLER_RS.contains("fn top_k_order(logits: &[f32], top_k: usize) -> Vec<usize> {"),
        "the selection is a CPU `select_nth_unstable_by` partition"
    );
    assert!(
        SAMPLER_RS.contains("fn next_f64(&mut self) -> f64 {"),
        "the RNG is a host xorshift, so its state cannot live on the device \
         without being rewritten"
    );
    // The device pick is present but unusable AND wrong.
    assert!(
        ELEMENTS_RS.contains("pub fn encode_greedy_pick("),
        "elements.rs:815"
    );
    assert!(
        ELEMENTS_RS.contains("if row == 0 {\n            return Err(KernelError::InvalidInput(\n                \"greedy pick row must be non-zero\".to_string(),"),
        "elements.rs:823-827 rejects row 0, the index the next embedding reads"
    );
    assert!(
        MODEL_RS.contains("underflows for every index at or past 65,536"),
        "model.rs:4287-4292 records the key underflow, and the measured \
         consequence (host argmax 9,259 vs device 0xFFFFFFFF)"
    );

    // Enumerate the kernels so the absence is on the record rather than implied.
    let kernels: Vec<&str> = ELEMENTS_METAL
        .lines()
        .filter_map(|line| line.trim().strip_prefix("kernel void "))
        .filter_map(|rest| rest.split('(').next())
        .collect();
    eprintln!("=== elements.metal: {} kernels, none of them a sampler ===\n  {}",
        kernels.len(), kernels.join("\n  "));
    assert_eq!(
        kernels.len(),
        25,
        "the kernel list changed; check whether a sampler was added before \
         trusting the absence assertion above"
    );
    assert!(
        !kernels.iter().any(|name| {
            let n = name.to_ascii_lowercase();
            n.contains("top_k") || n.contains("topk") || n.contains("softmax")
                || n.contains("sample") || n.contains("rng")
        }),
        "a kernel in {kernels:?} looks like a sampler; re-derive P06"
    );
}

/// What a device-side sampler would have to reproduce, and what it would be
/// worth. Not an implementation plan -- a sizing, so the next agent does not
/// have to re-derive the payoff.
#[test]
fn a_device_side_sampler_would_unlock_both_halves_of_p06() {
    // Half 1: remove the host sample + widen + readback from the critical path.
    let host_critical_ms =
        SAMPLE_MS + (WIDEN_MS_LOW + WIDEN_MS_HIGH) / 2.0 + (READBACK_MS_LOW + READBACK_MS_HIGH) / 2.0;
    assert!(
        (host_critical_ms - 3.47).abs() < 0.05,
        "host sampling + widen + readback is {host_critical_ms:.2} ms/step \
         (2.87 measured + 0.45 + 0.15 bracketed)"
    );

    // Half 2: with the token resident on the device, `chat.rs:388`'s
    // `self.greedy` gate can be dropped and the 2-deep pipelined loop reaches
    // temperature > 0. The plumbing already exists and is sampler-agnostic in
    // shape: `forward_step_pipelined` writes the token into
    // `token_ids_decode[next_slot]` and `host_tokens[step_idx]`
    // (model.rs:3778-3779) and the loop reads one u32 (model.rs:3794-3799).
    assert!(
        MODEL_RS.contains("let next_token_buffer = self.scratch.token_ids_decode[next_slot].clone();"),
        "model.rs:3778 -- the hand-off buffer an on-device sampler would write"
    );
    assert!(
        MODEL_RS.contains("pub fn read_host_token(&self, step_idx: usize) -> u32 {"),
        "model.rs:3794 -- the host-visible exit point, already one u32"
    );

    let barrier = barrier_penalty_ms();
    let total_recoverable = barrier + host_critical_ms;
    let temperature_zero_ms = BASELINE_MS;
    let projected_ms = temperature_zero_ms;
    eprintln!("=== payoff of a device-side sampler ===");
    eprintln!("  remove host sample + widen + readback   -{host_critical_ms:.2} ms");
    eprintln!("  open the pipelined loop to temperature>0  -{barrier:.2} ms");
    eprintln!("  => temperature 1.0 step                 {projected_ms:.2} ms  \
               ({:.2} tok/s) == the temperature-0 baseline, i.e. PARITY",
        1000.0 / projected_ms);
    eprintln!("  the second half is not separable: it is blocked by the first, which \
               is why this is one task, not two.");

    assert!(
        (total_recoverable - 7.30).abs() < 0.2,
        "the two halves total {total_recoverable:.2} ms/step"
    );

    // What this does NOT buy, stated so nobody extrapolates from it. Removing
    // host time lands temperature>0 exactly on the greedy baseline, 62.3 ms,
    // which is 193 GB/s -- the number the box already achieves. Any projection
    // *below* 62.3 ms would require the 3 bpw layer GEMVs to sustain more than
    // 193 GB/s in production, and the best 3 bpw kernel ever measured here is
    // 203 GB/s (and §16.3 records that the microbenchmark inverts against
    // production, so even that is an upper bound). So the honest claim is
    // parity, not an improvement.
    let implied_gb_s = BYTES_PER_TOKEN_CURRENT / (projected_ms / 1e3) / 1e9;
    let gemv_only_ms = step_ms_at(BYTES_PER_TOKEN_ITEMISED, BEST_3BPW_KERNEL_GB_S);
    eprintln!("  projected step implies {implied_gb_s:.0} GB/s; best 3 bpw kernel ever \
               measured here is {BEST_3BPW_KERNEL_GB_S} GB/s ({gemv_only_ms:.1} ms for the \
               itemised {BYTES_PER_TOKEN_ITEMISED:.0} B). Below {projected_ms:.1} ms is not \
               supported by this tree.");

    assert!(
        (implied_gb_s - ACHIEVED_GB_S).abs() < 2.0,
        "parity means the projection lands back at the achieved {ACHIEVED_GB_S} GB/s, \
         not above the best kernel ever measured"
    );
    assert!(
        projected_ms > gemv_only_ms,
        "the projected {projected_ms:.1} ms is slower than streaming the same bytes \
         at the best 3 bpw kernel ever measured ({gemv_only_ms:.1} ms), which is the \
         consistency check: P06 returns the step to where it already was, it does \
         not push it past the bandwidth floor"
    );
}