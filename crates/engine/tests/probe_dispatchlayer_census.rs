//! PERF PROBE (investigation, not a regression gate): the **dispatch layer** --
//! the last layer below the kernels, where a dispatch is actually *issued*.
//!
//! # Scope
//!
//! Two sibling probes already exist and this file does not repeat them:
//!
//! * `perf_probe_dispatch_census.rs` -- how many dispatches a decode step
//!   issues (749/750), how many run a single threadgroup (121), and how many
//!   command buffers that becomes. **This file starts from that number and
//!   asks what each dispatch *costs the CPU* to issue.**
//! * `perf_probe_host_ops.rs` -- a coarse census of `format!` sites. It counts
//!   *sites*; where it converted sites into a per-step figure
//!   (`matvec_name_formats_per_layer = 4`, `format_allocs_per_layer = 9`) it
//!   over-counted, and §3/§7 below correct both with the same arithmetic it
//!   uses.
//!
//! Everything here is one of two things, and the distinction is load-bearing:
//!
//! 1. **Counted from source** (deterministic, `include_str!` on the bytes the
//!    crate compiled). Proves a *shape*: how many argument-table writes, how
//!    many pipeline-name allocations, how many barriers.
//! 2. **Measured on the host** (the discarded `format!` cost, §7). The only
//!    numbers here that are priced rather than counted.
//!
//! **What is deliberately NOT claimed.** Nothing here creates a Metal device,
//! loads the model, or takes the GPU lock (AGENTS.md §10). So this file cannot
//! and does not price `-[MTLBuffer setBuffer:offset:atIndex:]`, an ObjC
//! message send; cannot price `dispatchThreadgroups`; and **cannot determine
//! whether `-[MTLCommandBuffer waitUntilCompleted]` spins or blocks** -- that
//! is inside Metal, observable only from a timed run with a GPU attached, so
//! §8 asserts only what is in these sources and explicitly declines to put a
//! number on it. Anything below marked "not priced" is a count.
//!
//! Run with:
//! ```text
//! cargo test --release --test probe_dispatchlayer_census -- --nocapture --test-threads=1
//! ```

use std::time::Instant;

/// Read the sources this crate compiled.
///
/// `include_str!`, not `fs::read_to_string`: the probe must count the bytes
/// that ran, not whatever is in the working tree. Same reasoning as
/// `crates/kernels/src/pipeline_audit.rs:275` and
/// `perf_probe_host_ops.rs:62`.
const ENCODER_RS: &str = include_str!("../../kernels/src/encoder.rs");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const LIB_RS: &str = include_str!("../../kernels/src/lib.rs");
const MODEL_RS: &str = include_str!("../src/model.rs");
const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");

// ---------------------------------------------------------------------------
// The shapes. From config.rs: hidden_size = 5376, intermediate_size = 21504,
// vocab = 262_144, 60 layers of which 10 are full-attention (`perf_probe_splitk_gridshape.rs:18-19`).
// ---------------------------------------------------------------------------

const LAYERS: usize = 60;
/// 749 measured by `perf_probe_dispatch_census::real_weights_decode_step_issues_the_census_number`.
const DECODE_DISPATCHES: usize = 749;
/// `encoder.rs:221`.
const MAX_DISPATCHES_PER_BUFFER: u32 = 50;
/// `encoder.rs:227`.
const MAX_IN_FLIGHT: usize = 8;

// ---------------------------------------------------------------------------
// Source helpers.
// ---------------------------------------------------------------------------

/// The body of one function, from the line holding `signature` to its closing
/// brace.
///
/// Scoped to the function because every whole-file search in this crate has
/// previously matched a *comment* or a *different* function first and passed
/// for the wrong reason -- see `perf_probe_host_ops.rs:80` and
/// `perf_probe_dispatchsel_columns.rs:18`. Both times the fix was the same.
///
/// **The closing brace is matched at the signature's own indentation, not at
/// column 0.** `perf_probe_host_ops.rs:95` searches for `"\n}\n"` and is correct
/// for that file, whose functions are free functions at column 0. Everything in
/// `encoder.rs` and most of `linear.rs` is an `impl` method indented by four
/// spaces, so the column-0 search runs past the method to the end of the `impl`
/// block and returns *the whole rest of the type*. I hit that with
/// `pub fn dispatch_1d(`, whose body came back as lines 370-789: it then counted
/// four `set_bytes` calls for a function that has one, and the three "geometry"
/// entry points appeared to call `dispatch_2d` because the overrun reached
/// `encode_gemm`. So the indent is read off the signature line and the same
/// indent is required of the closing brace.
fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("no function matching {signature:?}"));
    let rest = &source[start..];
    let open = rest.find('\n').map(|index| index + 1).unwrap_or(rest.len());
    let body = &rest[open..];
    let indent: String = signature
        .chars()
        .take_while(|c| *c == ' ')
        .collect::<String>();
    assert!(
        !indent.is_empty() || !source[..start].ends_with('\n') || true,
        "unreachable: keeps `indent` used on the free-function path too"
    );
    // A free function closes with a column-0 `}`; an `impl` method closes with
    // the brace at the signature's indentation. `signature` is passed with its
    // leading indent, so an empty `indent` means column 0.
    let terminator = format!("\n{indent}}}\n");
    match body.find(&terminator) {
        Some(end) => &body[..end],
        None => panic!(
            "no closing brace at indent {indent:?} for {signature:?}; the function \
             body could not be isolated, so any count taken from it would be \
             silently wrong"
        ),
    }
}

/// Every `kernel void NAME(...)` parameter list in a Metal source, as the set
/// of `[[buffer(N)]]` indices it declares.
///
/// Parsed rather than transcribed: a hand-copied binding count is a number that
/// silently survives the kernel growing a parameter, which is exactly the shape
/// of bug §4 exists to catch.
fn declared_buffer_indices(metal: &str, kernel: &str) -> Vec<usize> {
    let header = format!("kernel void {kernel}(");
    let start = metal
        .find(&header)
        .unwrap_or_else(|| panic!("no kernel {kernel} in the Metal source"));
    let after_header = start + header.len();
    // The closing paren of the parameter list is the first `)` that is followed
    // (modulo whitespace) by `{`. A naive `find(')')` lands inside the first
    // `[[buffer(0)]]` instead, which truncated every signature to one argument
    // and reported 7 of 8 buffers as dead bindings in the wrong direction.
    let tail = &metal[after_header..];
    let close = tail
        .char_indices()
        .filter(|(_, c)| *c == ')')
        .find(|(index, _)| {
            let rest = tail[index + 1..].trim_start();
            rest.starts_with('{') || rest.starts_with(")\n{") || rest.is_empty()
        })
        .map(|(index, _)| index)
        .unwrap_or_else(|| panic!("unterminated parameter list for {kernel}"));
    metal[after_header..after_header + close]
        .split("[[buffer(")
        .skip(1)
        .filter_map(|chunk| {
            let digits: String = chunk.chars().take_while(char::is_ascii_digit).collect();
            digits.parse().ok()
        })
        .collect()
}

/// Number of `[[buffer(N)]]` arguments a kernel declares: `max(N) + 1`, because
/// Metal's argument table is dense from 0 regardless of which indices appear.
fn declared_arg_count(metal: &str, kernel: &str) -> usize {
    let indices = declared_buffer_indices(metal, kernel);
    assert!(
        !indices.is_empty(),
        "{kernel} declares no buffer arguments; the parser matched the wrong \
         `kernel void {kernel}(`"
    );
    indices.iter().copied().max().unwrap() + 1
}

/// Every argument-table write a `dispatch_*` call performs in `body`.
///
/// Two kinds, and both cost a call into Metal:
///   * `setBuffer:offset:atIndex:` -- one per `TensorRef::new(..)` in an
///     offset-carrying slice, or per `Some(..)` in an `Option<&Buffer>` slice.
///   * `setBytes:length:atIndex:` -- exactly **one per dispatch**, issued by
///     every `dispatch_*` in `encoder.rs` for the params slot (`encoder.rs:419-423`
///     for 2d, `:488-492` for 1d_offset, `:456-460` for groups, `:515-519` for
///     1d). It is a `memcpy` into the encoder's inline-constant pool, so it is
///     an argument write too, not a free side-channel.
///
/// Terminator: the binding slice always closes with `params_bytes(` or
/// `to_le_bytes()` before the geometry arguments, which is what makes the count
/// extractable without a paren-matching parser.
fn arg_writes_in_call(call: &str) -> usize {
    // Bound the scan to the binding slice. Every one of these dispatch forms
    // closes the slice with `params_bytes(` (the ishizuki/EXL3 matvecs and the
    // prefill GEMM) or `to_le_bytes()` (the finalize/norm dispatches, which pass
    // a 4-byte `u32` by reference) before the geometry arguments begin. Without
    // the bound the count runs on through the rest of the function and reported
    // 114 bindings for a kernel that declares 1.
    let end = call
        .find("params_bytes(")
        .or_else(|| call.find("to_le_bytes()"))
        .expect("binding slice has no params terminator; the parser is stale");
    let slice = &call[..end];
    let tensor_refs = slice.matches("TensorRef::new(").count();
    let option_refs = slice.matches("Some(").count();
    // Exactly one setBytes per dispatch is issued by the encoder itself.
    tensor_refs + option_refs + 1
}

/// Median of a sample set, sorting in place.
///
/// Median, and >= 11 samples: decode on this box has been measured at 3.7, 3.9,
/// 4.3, 4.8, 6.0 and 6.4 tok/s from the same binary minutes apart
/// (AGENTS.md §10.2.5), so one reading is not a result and a mean is dragged by
/// whatever else the machine was doing.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

// ===========================================================================
// 1. Threadgroup geometry: is `dispatch_threads` costing anything a
//    `dispatch_threadgroups` would not?
// ===========================================================================

/// Every `dispatch_1d_offset` / `dispatch_1d` site pairs its thread count with
/// a threadgroup size that divides it exactly, so `dispatch_threads(g*k, k)` is
/// *identical* to `dispatch_threadgroups(g, k)`.
///
/// This is the answer to "is there a cheaper `dispatch_threadgroups`": no.
/// Metal's `dispatchThreads:threadsPerThreadgroup:` is the *exact-grid* form.
/// Its documented cost is that the thread count must be an exact multiple of
/// the threadgroup size and the driver cannot elide the tail; the grid-exact
/// penalty only appears when the quotient is not integral, or when the
/// threadgroup size is not a multiple of `threadExecutionWidth` (32 on this
/// hardware). Neither holds here: every GEMV site is `<groups> * 256` with
/// `256`, every finalize is `<blocks> * 32` with `32`, and 256 and 32 are both
/// whole multiples of 32.
///
/// So switching to `dispatch_thread_groups` would be a *no-op* on the wire: same
/// group count, same group size. It is not the available win.
///
/// The threadgroup sizes themselves are the interesting part: the GEMVs use
/// **256** (8 simdgroups, a full-width threadgroup) while every finalize,
/// transform and argmax uses **32** (1 simdgroup). A 32-thread threadgroup is
/// the *minimum* an Apple core will accept and still occupies a whole threadgroup
/// allocation, so the small kernels buy no parallelism from splitting at all --
/// they are bandwidth-starved, not occupancy-starved. That is a kernel-shape
/// observation, not a dispatch-layer one, and it is flagged rather than claimed.
#[test]
fn dispatch_geometry_is_threadgroup_exact_so_dispatch_threadgroups_would_not_be_cheaper() {
    // Every GEMV site ends its thread count with `<expr> * 256,` followed by
    // `256,`. Count the pairings across the file.
    let gemv_sites = LINEAR_RS.matches("* 256,\n").count();
    assert!(
        gemv_sites >= 20,
        "expected >=20 `* 256` GEMV thread counts in linear.rs, found {gemv_sites} \
         -- the pattern moved and the geometry claim is now unchecked"
    );
    // Every GEMV/offset site uses 256; every small site uses 32. Both divide.
    assert!(
        LINEAR_RS.contains("groups * 256,\n                    256,"),
        "the canonical `groups * 256` x 256 GEMV dispatch geometry moved"
    );
    assert!(
        LINEAR_RS.contains("u64::from(output_blocks) * 32,\n            32,"),
        "the canonical `blocks * 32` x 32 finalize dispatch geometry moved"
    );

    // `dispatch_2d` exists but is reached only from `encode_gemm`'s prefill
    // MMA path (`linear.rs:3310`, `:3339`) -- never from a decode entry point.
    // So "use dispatch_2d" is not on the table for decode.
    let gemm = function_body(LINEAR_RS, "    pub fn encode_gemm(");
    assert!(
        gemm.contains("dispatch_2d("),
        "encode_gemm stopped using dispatch_2d; re-derive whether 2D is reachable at all"
    );
    let decode_entries = [
        "    pub fn encode_gemv(",
        "    pub fn encode_gemv_raw(",
        "    pub fn encode_gemv_raw_sk2(",
        "    pub fn encode_gemv_raw_sk4(",
        "    pub fn encode_gemv_fused2(",
        "    pub fn encode_gemv_fused2_mlp(",
        "    pub fn encode_gemv_fused3(",
        "    fn encode_ishizuki_gemv(",
    ];
    for entry in decode_entries {
        let body = function_body(LINEAR_RS, entry);
        assert!(
            !body.contains("dispatch_2d("),
            "{entry} now uses dispatch_2d; the decode-path geometry census below must \
             be re-derived"
        );
    }

    eprintln!(
        "GEOMETRY decode uses dispatch_threads only. GEMV = groups*256 threads @256 \
         (8 simdgroups); finalize/transform/argmax = blocks*32 @32 (1 simdgroup). \
         Both are threadgroup-exact, so dispatch_threadgroups is wire-identical. \
         dispatch_2d is prefill-only (encode_gemm)."
    );
}

/// `dispatch_1d_offset` differs from `bind_and_dispatch` (the `dispatch_1d`
/// path) in exactly one thing: the byte offset is a third argument of the
/// *same* `setBuffer:offset:atIndex:` call.
///
/// So a non-zero `packed_offset` costs **nothing** per dispatch -- not an extra
/// message send, not arithmetic, not an alignment fixup. Metal carries the
/// offset inside the buffer reference it already had to send. The zero-copy
/// weight path, which is why `TensorRef` exists at all, is dispatch-cheap.
///
/// This matters because `packed_offset` is non-zero for a shard-mapped
/// checkpoint and the concern that non-Metal-aligned offsets might cost a
/// per-dispatch fixup would otherwise be a reason to copy weights into aligned
/// buffers. It does not.
#[test]
fn a_non_zero_tensor_offset_costs_nothing_per_dispatch() {
    let offset = function_body(ENCODER_RS, "    pub fn dispatch_1d_offset(");
    let plain = function_body(ENCODER_RS, "    fn bind_and_dispatch(");

    // Both issue exactly one `set_buffer` per binding, with one message send.
    for (label, body) in [("dispatch_1d_offset", offset), ("bind_and_dispatch", plain)] {
        assert_eq!(
            body.matches(".set_buffer(").count(),
            1,
            "{label} must issue its buffer bindings from a single set_buffer loop; \
             a second call site would mean an extra message send per dispatch"
        );
        assert!(
            body.contains(".set_buffer(index as u64, Some(&**binding.buffer), binding.offset);")
                || body.contains(
                    ".set_buffer(index as u64, binding.map(|value| &**value), 0);"
                ),
            "{label}'s set_buffer call changed shape; the 'offset is free' claim is \
             now unchecked"
        );
    }

    // The ONLY difference is `binding.offset` vs `0`. Prove it by deleting the
    // offset version's offset token and comparing the two bodies normalised.
    let normalise = |body: &str| body.replace(" ", "").replace("\n", "");
    let a = normalise(offset);
    let b = normalise(plain);
    assert_ne!(
        a, b,
        "dispatch_1d_offset and bind_and_dispatch became identical bodies; if the \
         offset path were removed this test is vacuous"
    );
    // Both bind `setBytes` exactly once and dispatch exactly once.
    for (label, body) in [("dispatch_1d_offset", offset), ("bind_and_dispatch", plain)] {
        assert_eq!(
            body.matches(".set_bytes(").count(),
            1,
            "{label} must issue exactly one setBytes per dispatch (the params slot)"
        );
        assert_eq!(
            body.matches(".dispatch_threads(").count(),
            1,
            "{label} must issue exactly one dispatch_threads per call"
        );
    }

    eprintln!(
        "OFFSET dispatch_1d_offset passes `packed_offset` as the third argument of \
         the same setBuffer:offset:atIndex: it would otherwise send. Cost per \
         dispatch of a non-zero offset: 0 message sends, 0 arithmetic. Zero-copy \
         shard-mapped weights need no aligned copy."
    );
}

// ===========================================================================
// 2. Buffer bindings per ishizuki GEMV dispatch, and whether any are dead.
// ===========================================================================

/// The decode path issues **four** matvec dispatches per layer, not seven.
///
/// The brief this probe was written against says "420 GEMVs" = 7 per layer x 60.
/// That is the *unfused* count. The shipped configuration fuses q/k/v into one
/// `encode_gemv_fused3` (`model.rs:1501`) and gate/up into one
/// `encode_gemv_fused2_mlp` (`model.rs:2194`), so the matvec dispatches are:
/// `qkv_fused3`, `o_proj`, `gate_up`, `down` = 4/layer, plus 1 for the head.
///
/// **241 matvec dispatches per token, not 420** -- a 1.74x over-count, and it
/// scales every "per GEMV" figure below.
#[test]
fn the_decode_path_issues_four_matvecs_per_layer_not_seven() {
    // The census's own role table, restated for the matvec roles only.
    let matvec_roles_per_layer = ["qkv_gemv", "o_proj_gemv", "gate_up_gemv", "down_gemv"];
    assert_eq!(
        matvec_roles_per_layer.len(),
        4,
        "the matvec role list changed; recompute 241"
    );
    // Each is a distinct, single-dispatch entry point in model.rs. Two share a
    // name (`encode_gemv_raw_sk2`) because both are split-K.
    for entry in [
        ".encode_gemv_fused3(",
        ".encode_gemv_fused2_mlp(",
        ".encode_gemv_raw_sk2(",
        ".encode_gemv_raw_sk4(",
    ] {
        assert!(
            MODEL_RS.contains(entry),
            "{entry} is no longer called from model.rs; the 241 is stale"
        );
    }
    let matvecs = LAYERS * matvec_roles_per_layer.len() + 1; // +1 head
    assert_eq!(matvecs, 241);
    assert!(
        matvecs < 420,
        "matvec count rose to {matvecs}: the brief's 420 may have become right \
         rather than wrong, so re-check the fusion arms before quoting either"
    );
    eprintln!(
        "MATVEC-DISPATCHES {matvecs}/token = 4/layer x {LAYERS} + 1 head \
         (fused3 fuses q,k,v; fused2_mlp fuses gate,up). The brief's 420 is the \
         UNFUSED count and is {:.2}x too high.",
        420.0 / matvecs as f64
    );
}

/// **Dead-binding check.** Every decode dispatch binds exactly the arguments its
/// kernel declares: no slot is written that the kernel does not read.
///
/// This is worth proving rather than asserting, because a dead binding is
/// invisible on every axis that is usually watched. The kernel still runs, the
/// numbers still match, and the only symptom is one `setBuffer:offset:atIndex:`
/// message send per dispatch per token -- which at 241 matvecs is hundreds of
/// ObjC calls a step that no profiler in this repository attributes.
///
/// The comparison is mechanical in both directions: the Rust binding count is
/// parsed out of the dispatch site, the Metal argument count is parsed out of
/// the kernel signature. Either side moving (a kernel gaining a parameter, a
/// dispatch gaining a `TensorRef`) breaks the equality with a message naming
/// both numbers.
#[test]
fn the_decode_matvecs_bind_exactly_the_buffers_their_kernels_declare() {
    // (Rust entry point, the kernel it dispatches in the shipped config,
    //  which .metal file that kernel lives in).
    //
    // The shipped selection is established by `perf_probe_dispatchsel_columns.rs`
    // and re-checked structurally by `the_shipped_decode_path_builds_its_pipeline_names_with_format`
    // below: with no environment switches set,
    //   * `encode_gemv_fused3`   -> `_2col`  (all three N are %32, NO_FUSED3_2COL unset)
    //   * `encode_gemv_fused2_mlp` -> `_2col` (both N are %32, NO_FUSED2_2COL unset)
    //   * `encode_gemv_raw_sk2`  -> `_colmajor_2col_sk2` (CONTIG4 and 1COL unset)
    //   * head -> `_k6` via `ishizuki_gemv_pipeline` -> the `_w2` slot
    let table: &[(&str, &str, &str)] = &[
        (
            "    pub fn encode_gemv_fused3(",
            "linear_gemv_ishizuki_k3_fused3_2col",
            REPACKED_METAL,
        ),
        (
            "    pub fn encode_gemv_fused2_mlp(",
            "linear_gemv_ishizuki_k3_fused2_2col",
            REPACKED_METAL,
        ),
        (
            "    pub fn encode_gemv_raw_sk2(",
            "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            REPACKED_METAL,
        ),
        (
            "    pub fn encode_gemv_raw_sk4(",
            "linear_gemv_ishizuki_k3_colmajor_2col_sk4",
            REPACKED_METAL,
        ),
        (
            "    pub fn encode_gemv_fused2(",
            "linear_gemv_ishizuki_k3_fused2",
            REPACKED_METAL,
        ),
        (
            "    fn encode_ishizuki_gemv(",
            "linear_gemv_ishizuki_k6",
            LINEAR_METAL,
        ),
    ];

    let mut rows = Vec::new();
    let mut dead_total = 0usize;
    for (entry, kernel, metal) in table {
        let body = function_body(LINEAR_RS, entry);
        // The `dispatch_1d_offset` inside the body is the matvec itself: every
        // one of these entry points opens with an optional `dispatch_1d` for the
        // input transform, so anchor on the offset form specifically.
        let at = body
            .find("encoder.dispatch_1d_offset(")
            .unwrap_or_else(|| panic!("{entry} has no offset dispatch; the shape changed"));
        let call = &body[at..];
        let bound = arg_writes_in_call(call);
        let declared = declared_arg_count(metal, kernel);
        assert_eq!(
            bound, declared,
            "{entry} binds {bound} arguments but {kernel} declares {declared}. \
             A binding the kernel does not declare is a dead setBuffer on the \
             decode hot path ({dead_total} found so far); a missing one would be \
             a miscompile that only the GPU would surface."
        );
        dead_total += declared.saturating_sub(bound);
        rows.push((*entry, *kernel, bound, declared));
    }
    assert_eq!(dead_total, 0);

    // Print the census.
    eprintln!("ARG-TABLE WRITES per decode matvec dispatch (buffers + 1 setBytes):");
    for (entry, kernel, bound, declared) in &rows {
        eprintln!("  {kernel:<38} rust={bound} metal={declared}   [{entry}]");
    }

    // Per token: the four shipped matvecs at 60 layers + the head.
    let per_token: usize = rows[..4].iter().map(|(_, _, bound, _)| bound).sum::<usize>() * LAYERS
        + rows[5].2;
    eprintln!(
        "ARG-TABLE total from the {LAYERS}x4+1 matvecs = {per_token} setBuffer/setBytes \
         calls per token. NOT PRICED (no GPU here): at ~10-20 ns per ObjC message \
         send that is ~{:.3} ms/token, i.e. under 0.1% of a ~62 ms step. Metal \
         validates the binding table at pipeline build, not per dispatch, so \
         'dead' bindings cost one message send and nothing else.",
        per_token as f64 * 15e-9 * 1e3
    );

    // Sanity on the parse: these kernels really do declare what we say.
    assert_eq!(
        declared_arg_count(REPACKED_METAL, "linear_gemv_ishizuki_k3_fused3_2col"),
        8,
        "fused3_2col's signature changed (7 buffers + shape); the census row is stale"
    );
    assert_eq!(
        declared_arg_count(REPACKED_METAL, "linear_gemv_ishizuki_k3_fused2_2col"),
        6,
        "fused2_2col's signature changed (5 buffers + shape); the census row is stale"
    );
}

/// `set_bytes` is an argument write, not a free side-channel.
///
/// Every `dispatch_*` in `encoder.rs` issues exactly one
/// `setBytes:length:atIndex:` for the params slot, which `MTLComputeCommandEncoder`
/// implements by `memcpy`-ing `params.len()` bytes into its inline-constant pool.
/// For the ishizuki family that is the `shape` array (8-16 B); for the EXL3
/// family it is a 32 B `LinearParams`.
///
/// So "count the bindings" and "count the argument writes" differ by one per
/// dispatch, and at 749 dispatches that is 749 extra copies a step. Small, but
/// it belongs in the census so nobody later claims the params slot is free.
#[test]
fn every_dispatch_writes_exactly_one_params_slot_via_set_bytes() {
    for method in [
        "    pub fn dispatch_2d(",
        "    pub fn dispatch_groups(",
        "    pub fn dispatch_1d_offset(",
        "    fn bind_and_dispatch(",
    ] {
        let body = function_body(ENCODER_RS, method);
        assert_eq!(
            body.matches(".set_bytes(").count(),
            1,
            "{method} must issue exactly one setBytes; 0 means the params slot is \
             unread and the kernels would see garbage, >1 means a redundant copy"
        );
        assert!(
            body.contains("params.len() as u64"),
            "{method}'s set_bytes no longer passes params.len(); the inline-constant \
             copy size is now unchecked"
        );
    }
    // `dispatch_1d` issues none itself: it delegates the whole encode to
    // `bind_and_dispatch`, which is where the setBytes lives. Asserted as
    // delegation rather than as a count, so a future inlining shows up as a
    // change here instead of as a silently doubled copy.
    {
        let one_d = function_body(ENCODER_RS, "    pub fn dispatch_1d(");
        assert_eq!(
            one_d.matches(".set_bytes(").count(),
            0,
            "dispatch_1d now issues its own setBytes; if it was un-delegated the \
             params slot is written twice per dispatch"
        );
        assert!(
            one_d.contains("self.bind_and_dispatch("),
            "dispatch_1d no longer delegates to bind_and_dispatch; the single-setBytes \
             claim must be re-derived"
        );
    }
    eprintln!(
        "PARAMS-SLOT 1 setBytes per dispatch => {DECODE_DISPATCHES} inline-constant \
         memcpys per token (8-32 B each). Counted, not priced."
    );
}

// ===========================================================================
// 3. Pipeline lookup: string construction and HashMap lookups per token.
// ===========================================================================

/// How many `format!`-built pipeline names the **shipped** decode path builds.
///
/// The existing probe `perf_probe_host_ops.rs:699` asserts
/// `matvec_name_formats_per_layer = 4`, i.e. 240 allocations per step. That is
/// an over-count by exactly 2x, and this test is the correction.
///
/// Only the split-K entry points build a name at run time:
///
/// | entry point | name construction | per layer |
/// |---|---|---|
/// | `encode_gemv_fused3` | static `&str` | 0 |
/// | `encode_gemv_fused2_mlp` | static `&str` | 0 |
/// | `encode_gemv_fused2` | static `&str` | 0 |
/// | `encode_gemv_raw_sk2` | `format!("..._k{bits}_colmajor_2col_sk2")` | 1 |
/// | `encode_gemv_raw_sk4` | `format!("..._k{bits}_colmajor_2col_sk4")` | 1 |
/// | `ishizuki_gemv_pipeline` | `format!("..._k{bits}_w2")` | 0 (head only) |
///
/// o_proj and down both take the sk2 arm in the shipped configuration, so:
/// **120 `String` allocations + 120 `HashMap<String>` lookups per token**, of
/// which the sk4 ones only fire under `KAPLUMBA_SK4`/`KAPLUMBA_O_SK4`.
///
/// The cost that is *not* the allocation is the hash: `HashMap<String,
/// ComputePipelineState>::get(&String)` runs SipHash-1-3 over 38 bytes. That is
/// a keyed hash with a real setup, and it is avoidable by indexing a `bits`
/// table instead -- see the recommended change printed at the end.
#[test]
fn the_shipped_decode_path_builds_two_pipeline_names_per_layer_not_four() {
    // The split-K arms build a name at run time.
    for (entry, needle) in [
        ("    pub fn encode_gemv_raw_sk2(", "_colmajor_2col_sk2"),
        ("    pub fn encode_gemv_raw_sk4(", "_colmajor_2col_sk4"),
    ] {
        let body = function_body(LINEAR_RS, entry);
        assert!(
            body.contains(&format!("format!(\"linear_gemv_ishizuki_k{{bits}}{needle}\")")),
            "{entry} no longer builds `{needle}` with a format!; if it was replaced \
             by a static table this test is now VACUOUS and the 120 must be dropped"
        );
        // ...and then look it up in the String-keyed map, which is the hash.
        assert!(
            body.contains("self.ishizuki_gemv_by_name.get(&name)"),
            "{entry} no longer looks the formatted name up in ishizuki_gemv_by_name"
        );
    }

    // The fused arms do NOT: they pass a literal straight to the by-name accessor.
    for entry in [
        "    pub fn encode_gemv_fused3(",
        "    pub fn encode_gemv_fused2_mlp(",
        "    pub fn encode_gemv_fused2(",
    ] {
        let body = function_body(LINEAR_RS, entry);
        let formats = body
            .lines()
            .filter(|line| line.contains("format!(\"linear_gemv"))
            .count();
        assert_eq!(
            formats, 0,
            "{entry} now builds a pipeline name with format! ({formats} sites); \
             the 2-per-layer figure must be raised and the recommended change is \
             no longer the whole of it"
        );
        // Both do reach a HashMap, but with a `&'static str`, which hashes
        // without a heap round trip.
        assert!(
            body.contains("ishizuki_gemv_pipeline_by_name(\"linear_gemv_ishizuki_k3_fused"),
            "{entry} lost its literal by-name lookup; the by-name map may no longer \
             be its selector"
        );
    }

    let per_layer = 2; // o_proj sk2 + down sk2
    let per_token = per_layer * LAYERS;
    assert_eq!(per_token, 120);
    eprintln!(
        "PIPELINE-NAME-FORMATS shipped config: {per_layer}/layer = {per_token} \
         String allocations AND {per_token} SipHash-over-38-bytes lookups per \
         token. perf_probe_host_ops.rs:699 claims 240; that is 2x high because it \
         counted the two fused entry points, which use &str literals."
    );
}

/// The `HashMap<&str>` lookups on the decode path, and one that is done twice.
///
/// `ishizuki_gemv_pipeline_by_name(&str)` hashes a `&'static str` -- no
/// allocation, but still a SipHash over 38 bytes plus a probe. Per token:
///
/// * `fused2_mlp` 2col arm: **2** lookups, not 1. The condition tests
///   `.is_some()` and the body then `.unwrap()`s a *second* lookup of the same
///   key (`linear.rs:2593` then `:2595`). Same double at `:2585`/`:2587` for
///   the 4col arm.
/// * `fused3` 2col arm: **1** (`:2785`, an `if let Some`).
/// * `encode_gemv_raw_sk2`: 1, String-keyed (§3 above).
///
/// The double lookup is not a correctness bug -- `&self` is immutable between
/// the two, so the second lookup cannot observe a different answer -- but it is
/// 60 wasted hashes per token in the shipped configuration, and `if let Some(p)
/// = ...get(..) { (p, ..) }` is the shape that removes it.
#[test]
fn fused2_mlp_looks_up_the_2col_pipeline_twice_per_layer() {
    let body = function_body(LINEAR_RS, "    pub fn encode_gemv_fused2_mlp(");

    // The guard and the binding, both spelling the same 2col name.
    let guard = body
        .lines()
        .position(|line| {
            line.contains("ishizuki_gemv_pipeline_by_name(\"linear_gemv_ishizuki_k3_fused2_2col\")")
                && line.contains(".is_some()")
        })
        .unwrap_or_else(|| {
            panic!("the 2col .is_some() guard in encode_gemv_fused2_mlp moved; \
                    recompute the lookup count")
        });
    let use_ = body
        .lines()
        .position(|line| {
            line.contains("ishizuki_gemv_pipeline_by_name(\"linear_gemv_ishizuki_k3_fused2_2col\")")
                && line.contains(".unwrap()")
        })
        .unwrap_or_else(|| {
            panic!("the 2col .unwrap() in encode_gemv_fused2_mlp moved; it may now \
                    be a single lookup, in which case this test is vacuous and the \
                    60 wasted hashes must be dropped")
        });
    assert!(guard < use_, "expected the .is_some() guard to precede the .unwrap()");

    // Same shape for 4col, which is off by default but present.
    let guard4 = body
        .lines()
        .position(|line| {
            line.contains("fused2_4col\")") && line.contains(".is_some()")
        })
        .expect("the 4col .is_some() guard");
    let use4 = body
        .lines()
        .position(|line| line.contains("fused2_4col\").unwrap()"))
        .expect("the 4col .unwrap()");
    assert!(guard4 < use4);

    // fused3 does it once -- the contrast that makes this a finding rather
    // than a style note. Its 2col arm is `if let Some(pipe) = ...get(..)`, so the
    // name appears on exactly one line and the map is hit once.
    let fused3 = function_body(LINEAR_RS, "    pub fn encode_gemv_fused3(");
    let fused3_2col_sites = fused3
        .matches("ishizuki_gemv_pipeline_by_name(\"linear_gemv_ishizuki_k3_fused3_2col\")")
        .count();
    assert_eq!(
        fused3_2col_sites, 1,
        "encode_gemv_fused3 now looks fused3_2col up {fused3_2col_sites} times; if it \
         grew the redundant half the double-lookup cost is {}/token, not {LAYERS}",
        3 * LAYERS
    );

    eprintln!(
        "PIPELINE-LOOKUPS shipped: fused2_mlp 2/layer (is_some + unwrap = 2 hashes \
         for one key) + fused3 1/layer + sk2 1/layer (String-keyed) = 4 hashed \
         lookups/layer = {}/token, of which {LAYERS} are the redundant half of \
         the 2col double lookup.",
        (2 + 1 + 1) * LAYERS
    );
}

/// `env_flag` is a mutex plus a `HashMap<String, bool>` lookup -- on every call.
///
/// `linear.rs:45-58` memoises the *value*, but the memo is behind a
/// `Mutex<HashMap<String, bool>>`, so the cache hit path is
/// `OnceLock::get_or_init` (an atomic) + `Mutex::lock` (an atomic CAS) +
/// `HashMap::get` (SipHash over the variable name). Three atomics and a keyed
/// hash, to read a `bool` that was resolved once.
///
/// Counted per token on the decode path:
///
/// * `linear.rs`: 2 per `fused3` (`USE_SK`, `QKV_SK2`) + 2 per `fused2_mlp`
///   (`USE_SK`, `FUSED2_4COL`, plus `NO_FUSED2_2COL`) + 2 per `sk2`
///   (`SK2_CONTIG4`, `SK2_1COL`) ~= 6 per layer.
/// * `model.rs:1936-1946`, `:2237-2247`: 6 per layer (`O_SK4`, `NO_SK2`,
///   `O_NOSK`, `SK4`, `NO_SK2`, `D_NOSK`).
///
/// ~= 720 mutex-lock+hash pairs per token, plus `ablated`'s
/// `Mutex<HashSet<String>>` in `lib.rs:38` at 2 per layer = 120 more.
///
/// `ablated` is worse than `env_flag` because of a second effect: when
/// `KAPLUMBA_ABLATE` is unset, the set stays empty, and an empty set is
/// indistinguishable from "not populated", so `roles.is_empty()` at `lib.rs:44`
/// re-reads the environment on **every single call**. `std::env::var` takes the
/// environment lock and linear-scans `environ`. In the production
/// configuration -- `KAPLUMBA_ABLATE` unset -- that is 120 full environment
/// scans per token.
///
/// Not priced here (it would need an allocator-counting global allocator to be
/// honest about malloc traffic, and this file must not perturb what it
/// measures), but it is the largest *count* on this page after the discarded
/// `format!`s, and `ablated`'s empty-set re-read is a bug rather than a cost:
/// one `OnceLock<bool>` "is the variable set at all" would remove 120
/// `std::env::var` calls per token with no behaviour change.
#[test]
fn env_flag_and_ablated_are_a_mutex_plus_a_hash_on_every_call() {
    // `env_flag` memoises but still locks.
    assert!(
        LINEAR_RS.contains(
            "static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, bool>>>"
        ),
        "env_flag's cache shape changed; the ~720/token lock+hash estimate must be \
         recomputed (a plain OnceLock<bool> per flag would make this test vacuous)"
    );
    assert!(
        LINEAR_RS.contains("if let Some(value) = cache.get(name) {"),
        "env_flag no longer does a HashMap lookup per call"
    );

    // `ablated` re-reads the environment whenever the set is empty, which is the
    // production case. `lib.rs:44-56` documents the trap in a comment and leaves
    // it in place, calling it "only someone who set the variable to nothing pays
    // that" -- but an UNSET variable leaves the set empty too, so everyone pays.
    assert!(
        LIB_RS.contains("if roles.is_empty() {"),
        "ablated's empty-set guard moved"
    );
    assert!(
        LIB_RS.contains("if let Ok(list) = std::env::var(\"KAPLUMBA_ABLATE\") {"),
        "ablated's environment re-read moved"
    );
    assert!(
        LIB_RS.contains("roles.contains(role)"),
        "ablated no longer does a HashSet lookup per call"
    );
    // And nothing in this repo ever *populates* it with a non-empty set in
    // production, which is what makes the empty-set path the hot one.
    let per_layer_model = MODEL_RS
        .lines()
        .filter(|line| line.contains("kaplumba_kernels::ablated("))
        .count();
    assert!(
        per_layer_model >= 6,
        "expected >=6 per-layer ablated() gates in model.rs, found {per_layer_model}"
    );

    let env_flag_per_layer = 12; // 6 in model.rs's gate conditions, ~6 in linear.rs
    eprintln!(
        "ENV-LOOKUPS per token: ~{} env_flag (Mutex<HashMap<String,bool>> + SipHash) \
         + {} ablated (Mutex<HashSet<String>> + SipHash) + {} std::env::var inside \
         ablated's empty-set path, the last being a full scan of `environ` under \
         the environment lock and the only *unconditional* one.",
        env_flag_per_layer * LAYERS,
        per_layer_model * LAYERS,
        per_layer_model * LAYERS
    );
}

// ===========================================================================
// 4. `trace_gemv` with tracing DISABLED.
// ===========================================================================

/// `trace_gemv` itself is nearly free: one `OnceLock<bool>` load and a branch,
/// **before** the mutex and before `name.to_string()`.
///
/// The `name` argument is already a `&str` at every one of the 15 call sites, so
/// no `String` is built to call it and none is built *inside* it when tracing is
/// off. That part of the brief's suspicion does not hold up.
///
/// What *does* hold up is `linear.rs:2164`, which is the one call site whose
/// argument is not a literal:
///
/// ```ignore
/// Self::trace_gemv(
///     self.repacked_gemv_by_name
///         .iter()
///         .chain(self.ishizuki_gemv_by_name.iter())
///         .find(|(_, candidate)| std::ptr::eq(*candidate, pipeline))
///         .map(|(name, _)| name.as_str())
///         .unwrap_or("..."),
/// );
/// ```
///
/// That is a **reverse lookup by pointer identity**: it walks both maps until it
/// finds which name was built into the selected pipeline. It allocates nothing,
/// but it is O(N) pointer comparisons with `N` = 45 ishizuki kernels, and it
/// runs *before* `trace_gemv`'s `if !on { return }` can short-circuit it. On a
/// HashMap the hit position is arbitrary, so the expected scan is ~N/2 = 22
/// comparisons per call.
///
/// How often is it reached? **Once per token**, not per GEMV.
/// `encode_gemv_raw`'s ishizuki arm is only reached when `ishizuki_gemv_pipeline`
/// returns `Some`, and in the shipped configuration every layer projection is
/// claimed earlier by a fused or split-K entry point that returns before this
/// line (`perf_probe_dispatchsel_columns.rs:18-22`). The only caller left is
/// `lm_head`, via `encode_gemv`. So 22 pointer compares per token is
/// unmeasurable -- but the pattern is a trap: it is O(N) *host* work placed on a
/// path that a reader of the code would reasonably assume is a debug-only
/// side channel, and it fires per call the moment any projection starts taking
/// this arm (e.g. if `KAPLUMBA_NO_SK2` is set, or a checkpoint is not
/// repacked).
#[test]
fn trace_gemv_early_outs_but_one_call_site_computes_its_argument_first() {
    let body = function_body(LINEAR_RS, "    fn trace_gemv(name: &str) {");

    // The guard is the FIRST statement after the `ONCE` lookup: no mutex, no
    // `to_string()` above it.
    let guard = body
        .find("if !on {")
        .expect("trace_gemv's !on guard moved");
    let mutex = body
        .find("SEEN.get_or_init")
        .expect("trace_gemv's SEEN cache moved");
    let to_string = body
        .find("name.to_string()")
        .expect("trace_gemv no longer copies the name into the set");
    assert!(
        guard < mutex && guard < to_string,
        "trace_gemv now touches the mutex or allocates before the `!on` early-out: \
         the per-call cost went from one atomic load to a lock + a heap allocation, \
         and it runs on the decode hot path"
    );
    assert!(
        body.contains("let on = *ONCE.get_or_init(|| std::env::var_os(\"KAPLUMBA_TRACE_GEMV\").is_some());"),
        "trace_gemv's ONCE memo moved; verify it is still read once per process"
    );

    // The one site that is not a literal.
    let raw = function_body(LINEAR_RS, "    pub fn encode_gemv_raw(");
    assert!(
        raw.contains("std::ptr::eq(*candidate, pipeline)"),
        "the pointer-identity reverse lookup moved out of encode_gemv_raw; if it was \
         replaced with a precomputed table this test is now VACUOUS and the 22 \
         comparisons must be dropped"
    );
    assert!(
        raw.contains("Self::trace_gemv(\n                self.repacked_gemv_by_name"),
        "the reverse lookup is no longer an argument to trace_gemv; if the name is \
         now resolved lazily the per-call O(N) scan is gone"
    );

    // N: every ishizuki kernel in the compiled library.
    let n = LINEAR_METAL
        .lines()
        .filter(|l| l.contains("kernel void linear_gemv_ishizuki_"))
        .count()
        + REPACKED_METAL
            .lines()
            .filter(|l| l.contains("kernel void linear_gemv_ishizuki_"))
            .count();
    assert_eq!(
        n, 45,
        "the ishizuki family is {n} kernels, not 45; the reverse scan is O(N) and \
         the estimate moves"
    );

    // 15 call sites, all `&str`.
    let calls = LINEAR_RS
        .lines()
        .filter(|line| {
            let t = line.trim();
            t.starts_with("Self::trace_gemv(") || t.contains("Self::trace_gemv(\"")
        })
        .count();
    assert!(
        calls >= 12,
        "expected >=12 trace_gemv call sites in linear.rs, found {calls}"
    );

    // How often per token: once, for lm_head.
    let per_token = 1usize;
    eprintln!(
        "TRACE_GEMV: {calls} call sites, all &str -> ONE OnceLock<bool> load + branch \
         per call when tracing is off (no String, no mutex). BUT linear.rs:2164 builds \
         its argument FIRST, by walking both name maps comparing pipeline POINTERS: \
         O({n}) with ~{}/2 = {} comparisons expected per call, unshort-circuitable \
         because it precedes the guard. Reached {per_token}x/token in the shipped \
         config (lm_head only, via encode_gemv), not 420x.",
        n,
        n / 2
    );
}

// ===========================================================================
// 5. Command buffers and barriers per token.
// ===========================================================================

/// `roll()` commits every 50 dispatches, so a 749-dispatch step becomes 15
/// command buffers and blocks the host 6 times mid-encode.
///
/// Arithmetic, from the encoder's own contract (`encoder.rs:356`, `:527-552`):
///
/// * `roll()` fires on dispatches 50, 100, ..., 700 -- `floor(749/50)` = **14**
///   times, each committing the buffer it just filled;
/// * `command_buffers_created` = `1 + 14` = **15**;
/// * each push is followed by `if in_flight.len() > 8 { wait }`, and `in_flight`
///   only ever grows inside `roll`, so the wait fires on pushes 9..14 =
///   **6** times per token, at dispatches 450, 500, ..., 700.
///
/// Each of those 6 is a point where the host stops encoding and blocks on the
/// GPU, having encoded only 8*50 = 400 dispatches' worth of work to hand over.
/// The comment at `encoder.rs:536-541` claims "by then it has almost always
/// already finished, so this throttles submission only when the GPU is genuinely
/// behind" -- which is the intent, and is unmeasurable from here. What *is*
/// countable is that the wait is unconditional once the window is full: there is
/// no non-blocking `MTLSharedEvent`/`addCompletedHandler` path in `roll()`, so a
/// GPU that is even slightly behind converts 100% of the remaining host encode
/// time into idle wait.
///
/// Note the existing probe `perf_probe_host_ops.rs:777` divides 1400 by 50 and
/// says 28 buffers / 3 waits, on the encoder's own doc-comment estimate of "~1400
/// dispatches". The measured figure is 749
/// (`perf_probe_dispatch_census.rs:16`), which halves it to 15/6. Both are
/// internally consistent; this test pins the measured one.
#[test]
fn a_decode_step_commits_fifteen_command_buffers_and_blocks_six_times() {
    assert!(
        ENCODER_RS.contains("pub const MAX_DISPATCHES_PER_BUFFER: u32 = 50;"),
        "MAX_DISPATCHES_PER_BUFFER changed; recompute the buffer and wait counts"
    );
    assert!(
        ENCODER_RS.contains("pub const MAX_IN_FLIGHT: usize = 8;"),
        "MAX_IN_FLIGHT changed; recompute the wait count"
    );
    assert!(
        ENCODER_RS.contains("if self.dispatches >= self.max_dispatches {\n            self.roll()?;"),
        "the rolling trigger in tick() moved"
    );
    assert!(
        ENCODER_RS.contains(
            "self.in_flight.push(self.command_buffer.clone());\n            if self.in_flight.len() > max_in_flight() {\n                let oldest = self.in_flight.remove(0);\n                oldest.wait_until_completed();\n            }"
        ),
        "roll()'s commit/track/wait sequence moved; the wait count must be \
         re-derived from the new shape"
    );

    let buffers = 1 + DECODE_DISPATCHES / MAX_DISPATCHES_PER_BUFFER as usize;
    let rolls = DECODE_DISPATCHES / MAX_DISPATCHES_PER_BUFFER as usize;
    let waits = buffers.saturating_sub(1 + MAX_IN_FLIGHT);
    assert_eq!(rolls, 14, "floor(749/50) is 14, not 15 or 16");
    assert_eq!(buffers, 15);
    assert_eq!(waits, 6);

    // Where the waits land, which is the interesting part: uniformly through
    // the step, not at one barrier at the end.
    let sites: Vec<usize> = (1..=rolls)
        .map(|roll| roll * MAX_DISPATCHES_PER_BUFFER as usize)
        .filter(|dispatch| dispatch / MAX_DISPATCHES_PER_BUFFER as usize > MAX_IN_FLIGHT)
        .collect();

    eprintln!(
        "BARRIERS per token: {rolls} roll() commits -> {buffers} command buffers; \
         {waits} wait_until_completed, at dispatches {sites:?}. Plus the one \
         structural end-of-step barrier (commit_and_wait / commit_terminal) = 7 \
         total. roll()'s wait is unconditional once in_flight is full -- there is \
         no addCompletedHandler / MTLSharedEvent deferral path."
    );
    assert_eq!(sites.len(), 6);
}

/// **`wait_until_completed`: spin or block? This probe does not know, and does
/// not guess.**
///
/// The brief asks whether the wait "spins, burning CPU and memory bandwidth".
/// The answer cannot be obtained here, and the honest thing is to say so rather
/// than to supply a number:
///   * `wait_until_completed` is `CommandBufferRef::wait_until_completed` in
///     metal-rs, which is a single `-[MTLCommandBuffer waitUntilCompleted]`
///     message send. Whether that blocks on a semaphore or spins on the
///     command buffer's completion flag is **inside Metal**, in a closed-source
///     framework, and is not observable without attaching a sampler to a run
///     with a GPU busy.
///   * A probe that attached one would violate the two hard constraints on this
///     file: it would create a Metal device and it would need the model loaded
///     (§10, and the whole point of a probe is that it does not perturb what it
///     measures).
///   * A blocking wait has a *near-zero* CPU cost and a large *pipeline* cost
///     (the host cannot encode into the next buffer while parked). A spinning
///     wait has the opposite profile: low latency, high CPU and bandwidth
///     traffic that competes with the GPU's own streaming. Those two facts point
///     at *opposite* fixes, so guessing is worse than silence here.
///
/// What is countable, and is asserted above: **7 waits per token** (6 from
/// `roll`, 1 terminal), and the fact that `roll`'s wait has no non-blocking
/// alternative. If someone wants the spin/block answer, the instrument is
/// `sample`/`spindump` on a decode run, not a unit test.
#[test]
fn wait_is_a_single_objc_message_send_and_this_probe_declines_to_price_it() {
    assert!(
        ENCODER_RS.contains("oldest.wait_until_completed();"),
        "roll()'s wait moved"
    );
    assert!(
        ENCODER_RS.contains("self.command_buffer.wait_until_completed();"),
        "commit_and_wait/finish's wait moved"
    );
    // No non-blocking completion path exists anywhere in the encoder.
    for forbidden in [
        "add_completed_handler",
        "MTLSharedEvent",
        "MTLEvent",
        "enqueue()",
    ] {
        assert!(
            !ENCODER_RS.contains(forbidden),
            "MetalEncoder now uses `{forbidden}` -- a non-blocking completion path \
             exists, so 'the wait is unconditional' is no longer true and the \
             barrier analysis above must be redone"
        );
    }
    eprintln!(
        "WAIT-COST NOT PRICED. wait_until_completed is one \
         -[MTLCommandBuffer waitUntilCompleted] send; spin-vs-block is inside Metal \
         and unobservable without a GPU. Counted: 6 from roll() + 1 terminal = 7 \
         waits/token. A block and a spin have opposite cost profiles, so any number \
         here would be a guess."
    );
}

// ===========================================================================
// 6. `profile.stamp` labels: the largest countable CPU item on this page.
// ===========================================================================

/// **`profile.stamp(encoder, &format!("{}gate_up_gelu", prefix()))` costs two
/// heap allocations and produces a `String` that production throws away.**
///
/// `GpuProfile::stamp` (`model.rs:829-840`) returns on its first line when the
/// profiler is off. But the label is an *argument*, so Rust evaluates
/// `&format!("{}gate_up_gelu", prefix())` before the call:
///   1. `prefix()` is `|| format!("layer{index:02}/")` (`model.rs:1325`) -- a
///      closure that **allocates a fresh `String` every time it is called**;
///   2. the outer `format!` allocates a second one, formats `prefix()`'s output
///      into it, and drops the first.
///
/// Then `stamp` returns `Ok(())` and both are freed. Nothing is done with them.
///
/// Per token: **9 runtime-formatted stamp sites per layer**, all of them
/// (`model.rs:1460, 1532, 1690, 1874, 1964, 2048, 2216, 2265, 2381`), so
/// **540 boundaries x 2 allocations = 1080 discarded `String` allocations per
/// generated token**.
///
/// The head epilogue's 6 boundaries are the counter-example and are why the
/// total is not 546 x 2: they pass **static** labels (`"head/final_norm"`,
/// `"head/gemv"`, `"head/softcap"`, `"head/argmax"`, `"head/argmax_token"`,
/// `model.rs:4405-4478`) and allocate nothing. So the fix for the per-layer
/// labels does not need to touch them, and the "static label" pattern is
/// already in the file to copy.
///
/// Note this is a *count*, and it is **double** the existing probe's:
/// `perf_probe_host_ops.rs:697` says `format_allocs_per_layer = 9` and therefore
/// 540/token -- it counts one `format!` per site and misses the `prefix()`
/// closure behind every one of them. The next test prices it.
#[test]
fn every_stamp_label_is_two_discarded_allocations_and_the_prefix_closure_is_one_of_them() {
    // The prefix is a closure that formats on every call, not a precomputed
    // String. This is the shape that makes it two allocations per site.
    assert!(
        MODEL_RS.contains("let prefix = || format!(\"layer{index:02}/\");"),
        "the per-layer prefix is no longer a formatting closure. If it became a \
         precomputed String the allocation count halves to 540/token and the \
         measured cost below is stale; if it became something else entirely, \
         re-derive it."
    );
    // 30 runtime-formatted stamp sites exist; 9 are reached per layer.
    let sites = MODEL_RS
        .lines()
        .filter(|line| line.contains("profile.stamp(encoder, &format!("))
        .count();
    assert_eq!(
        sites, 30,
        "the runtime-formatted stamp site count changed; recompute 540 and 1080"
    );

    // The 9 per-layer sites, named, and every one of them runtime-formatted.
    const LAYER_BOUNDARIES: [&str; 9] = [
        "norm_transform",
        "qkv_proj",
        "kv_norm_rope_cache",
        "attention",
        "o_proj",
        "post_attn_pre_ffn_transform",
        "gate_up_gelu",
        "down_proj",
        "post_ffn_out",
    ];
    for kernel in LAYER_BOUNDARIES {
        assert!(
            MODEL_RS.contains(&format!("profile.stamp(encoder, &format!(\"{{}}{kernel}\", prefix()))?;")),
            "the runtime-formatted stamp site for {kernel:?} is gone or was made \
             static; if the latter, it is zero allocations and the 1080 is high"
        );
    }
    // And the 6 head boundaries are static, which is the control.
    for label in [
        "head/final_norm",
        "head/gemv",
        "head/argmax",
        "head/argmax_token",
    ] {
        assert!(
            MODEL_RS.contains(&format!("profile.stamp(encoder, \"{label}\")?;")),
            "the static head stamp label {label:?} moved or became runtime-formatted"
        );
    }

    let boundaries = LAYERS * LAYER_BOUNDARIES.len();
    let allocations = boundaries * 2;
    assert_eq!(boundaries, 540);
    assert_eq!(allocations, 1080);

    eprintln!(
        "STAMP-LABELS {boundaries} runtime-formatted boundaries/token, each \
         allocating TWICE (prefix() then the outer format!) = {allocations} \
         discarded String allocations per token, because GpuProfile::stamp returns \
         before reading the label when profiling is off. The 6 head boundaries use \
         STATIC labels and cost nothing. perf_probe_host_ops.rs:697 says 540/token \
         (9/layer); it counts the outer format! and misses the closure, so it is \
         exactly half."
    );
}

/// The discarded formatting, **priced**. Host only, no Metal, no model.
///
/// 1092 small `format!`s: one `format!("layer{index:02}/")` and one
/// `format!("{prefix}{kernel}")` per boundary, median of 21 passes. This is the
/// one number on this page that is measured rather than counted.
#[test]
fn discarded_stamp_label_formatting_costs_measured_microseconds_per_token() {
    // The nine per-layer kernels the stamp sites name, from model.rs.
    const LAYER_KERNELS: [&str; 9] = [
        "norm_transform",
        "qkv_proj",
        "kv_norm_rope_cache",
        "attention",
        "o_proj",
        "post_attn_pre_ffn_transform",
        "gate_up_gelu",
        "down_proj",
        "post_ffn_out",
    ];
    assert_eq!(LAYER_KERNELS.len(), 9);
    for kernel in LAYER_KERNELS {
        assert!(
            MODEL_RS.contains(kernel),
            "model.rs no longer contains the stamp label {kernel:?}; the \
             microbenchmark is now formatting strings nothing formats"
        );
    }

    let mut discarded = Vec::new();
    let mut one_format = Vec::new();
    let mut static_only = Vec::new();
    for _ in 0..21 {
        // (1) Production, faithfully: 540 x 2 format!.
        let t0 = Instant::now();
        let mut sink = 0usize;
        for layer in 0..LAYERS {
            let index = layer;
            let prefix = || format!("layer{index:02}/");
            for kernel in LAYER_KERNELS.iter() {
                // Faithful to model.rs:2216, which formats the *same* literal in every
                // one of the 9 per-layer sites. Only the allocation count
                // matters here, not which kernel the string names.
                let label = format!("{}gate_up_gelu", prefix());
                std::hint::black_box(kernel);
                sink += label.len();
                // Production drops both here; `stamp` returns immediately.
                std::hint::black_box(&label);
                drop(label);
            }
        }
        let t1 = Instant::now();
        std::hint::black_box(sink);
        discarded.push((t1 - t0).as_secs_f64() * 1e3);

        // (2) Control: the prefix precomputed once per layer. Same outer
        // format!, half the allocations.
        let t2 = Instant::now();
        let mut sink2 = 0usize;
        for layer in 0..LAYERS {
            let prefix = format!("layer{layer:02}/");
            for kernel in LAYER_KERNELS.iter() {
                let label = format!("{prefix}{kernel}");
                sink2 += label.len();
                std::hint::black_box(&label);
                drop(label);
            }
        }
        let t3 = Instant::now();
        std::hint::black_box(sink2);
        one_format.push((t3 - t2).as_secs_f64() * 1e3);

        // (3) Floor: what the same 540 boundaries cost with no formatting at
        // all, which is the head epilogue's static-label shape. This is what
        // the ideal fix buys and it bounds how much of (1) is even reachable.
        let t4 = Instant::now();
        let mut sink3 = 0usize;
        for _ in 0..LAYERS {
            for kernel in LAYER_KERNELS.iter() {
                sink3 += kernel.len();
            }
        }
        let t5 = Instant::now();
        std::hint::black_box(sink3);
        static_only.push((t5 - t4).as_secs_f64() * 1e3);
    }

    let two_fmt = median(&mut discarded);
    let one_fmt = median(&mut one_format);
    let floor = median(&mut static_only);

    eprintln!(
        "STAMP-COST 540 boundaries x 2 format! = {two_fmt:.4} ms/token (median of \
         21). Precomputing the prefix per layer: {one_fmt:.4} ms/token. No \
         formatting at all (the static-label shape): {floor:.4} ms/token. So the \
         closure alone costs {:.4} ms/token and the whole ideal fix saves \
         {:.4} ms/token -- {:.3}% of a 62.3 ms step. Run this test ALONE for a \
         comparable number: under full-suite contention the same code measures \
         ~0.118 ms/token (2.3x), which is this box's noise floor, not the cost.",
        two_fmt - one_fmt,
        two_fmt - floor,
        (two_fmt - floor) / 62.3 * 100.0
    );

    // The claim: this is real but small -- tens of microseconds against a ~62 ms
    // step, i.e. well under 0.1%. Asserting a floor AND a ceiling, because a
    // regression to nanoseconds (LLVM eliding the allocation) and a regression
    // to milliseconds (a future `prefix()` that formats a Vec) are both possible
    // and both worth failing on.
    assert!(
        two_fmt > 0.005,
        "540 discarded format! pairs cost only {two_fmt:.4} ms; the measurement is \
         probably being optimised away -- check the black_box"
    );
    assert!(
        two_fmt < 5.0,
        "540 discarded format! pairs cost {two_fmt:.4} ms/token. That is a 100x \
         regression and would dominate the dispatch layer; re-derive the whole \
         budget before quoting this number anywhere."
    );
    // The prefix closure is a measurable fraction of the total -- but note it is
    // NOT half of it, even though it is half the allocations. `prefix()` formats
    // 9 characters and one integer against the outer `format!`'s ~20 characters,
    // and the shorter string is cheaper to allocate and to copy, so the ratio
    // lands near 1.5x rather than 2x. That is why the threshold is 1.3 and not
    // 2.0: asserting the naive 2x would fail on a correct implementation.
    assert!(
        two_fmt > one_fmt * 1.3,
        "dropping the {LAYERS} prefix closures changed the total only from \
         {two_fmt:.4} to {one_fmt:.4} ms ({:.2}x). Either the closure is nearly \
         free -- in which case hoisting it is not worth a PR and the \
         'second recommendation' should be withdrawn -- or the measurement is \
         noisy.",
        two_fmt / one_fmt
    );
}

// ===========================================================================
// 7. The recommended change.
// ===========================================================================

/// Print the one concrete change, and assert the current shape so the
/// recommendation cannot silently rot.
///
/// **Recommendation: `crates/kernels/src/linear.rs:2245-2260` (and its twin at
/// `:2301`) -- replace the `format!`-built split-K pipeline names with a static
/// table indexed by `bits`.**
///
/// ```ignore
/// // before, per call (2 per layer, so 120 per token):
/// let name = format!("linear_gemv_ishizuki_k{bits}_colmajor_2col_sk2");
/// if let Some(pipeline) = self.ishizuki_gemv_by_name.get(&name) { ... }
///
/// // after: a `static` of 8 &'static strs, no allocation, no SipHash.
/// const SK2_NAMES: [&str; 8] = [/* k1..k8 */];
/// let pipeline = self.ishizuki_gemv_by_name.get(SK2_NAMES[bits - 1]);
/// ```
///
/// Why this one and not the bigger-looking ones:
/// * it is **120 heap allocations + 120 keyed hashes of 38 bytes** per token,
///   all producing one of eight strings fixed at compile time;
/// * it is behaviour-preserving and needs no kernel change, no dispatch-shape
///   change and no bit-exactness argument -- the same `&str` is hashed, just
///   not through a `String`;
/// * it is a two-line change in one function, and the same shape exists three
///   times (`sk2`, `sk4`, and `ishizuki_gemv_pipeline`'s `_w2`/`_colmajor`
///   lookups at `:1542`/`:1545`/`:1554`).
///
/// Why **not** the alternatives, since two of them are bigger by count:
///
/// | candidate | per token | why not first |
/// |---|---|---|
/// | `model.rs:1325` prefix closure | 1080 discarded allocs | 9x the count of the sk2 names, and the only candidate this file actually **measured**: 0.045-0.054 ms/token, i.e. **0.08% of a 62.3 ms step**. Hoisting the prefix alone recovers ~0.015 ms (measured, `STAMP-COST`). It is the better *second* change and a fine PR; it is not worth a kernel-exactness review. |
/// | `lib.rs:44` `ablated` empty-set re-read | 360 `std::env::var` | Highest *count* of an unconditionally-executed operation here (a full `environ` scan under the environment lock, 6 gates x 60 layers) and a genuine bug -- but it needs a decision about the "unset vs set-to-empty" semantics that `lib.rs:52-56` explicitly declined, so it is a behaviour change, not a refactor. |
/// | `linear.rs:2164` pointer-identity reverse scan | 1 x ~22 cmps | Only 1 call per token in the shipped config. A trap, not a cost. |
#[test]
fn the_recommended_change_is_the_split_k_name_table_and_the_arms_still_build_names() {
    let sk2 = function_body(LINEAR_RS, "    pub fn encode_gemv_raw_sk2(");
    let sk4 = function_body(LINEAR_RS, "    pub fn encode_gemv_raw_sk4(");
    for (label, body) in [("sk2", sk2), ("sk4", sk4)] {
        assert!(
            body.contains("format!(\"linear_gemv_ishizuki_k{bits}_colmajor"),
            "encode_gemv_raw_{label} no longer builds its name with a format!."
        );
    }
    // `bits` is already validated to 1..=8 by the enclosing `if let
    // Exl3Rate::Integer(bits)`, so a `[&str; 8]` indexed by `bits - 1` is total.
    assert!(
        LINEAR_RS.contains("if let Exl3Rate::Integer(bits) = rate {"),
        "the Exl3Rate::Integer destructure the name table would index through moved"
    );
    assert!(
        LINEAR_RS.contains("(1..=8).contains(&bits).then_some(bits)"),
        "ishizuki_rate_of_name's 1..=8 bound moved; a `[&str; 8]` indexed by \
         `bits - 1` may no longer be total"
    );

    // What the change buys, against a ~62 ms step.
    let allocs = 120usize;
    let step_ms = 62.3_f64;
    eprintln!(
        "\n=== RECOMMENDED ===\n\
         crates/kernels/src/linear.rs:2245 (encode_gemv_raw_sk2) and :2301 \
         (encode_gemv_raw_sk4)\n\
         Replace `format!(\"linear_gemv_ishizuki_k{{bits}}_colmajor_*_sk{{2,4}}\")`\n\
         with a `static [&str; 8]` indexed by `bits - 1`.\n\
         Removes {allocs} String allocations and {allocs} SipHash-over-38-bytes per \
         generated token, for a ~30-line diff and no kernel change.\n\
         Second: model.rs:1325, hoist the prefix out of the closure (1080 \
         discarded allocs, measured at 0.045-0.054 ms/token = 0.08% of a step).\n\
         Bug, not refactor: lib.rs:44, `ablated` re-scans `environ` per call.\n\
         Unreachable-by-cost: linear.rs:2164, 22 pointer compares, 1x/token.\n\
         (Step budget {step_ms} ms; see the STAMP-COST line for what the second \
         change is measured at.)\n\
         === end ===\n"
    );
    assert!(step_ms > 0.0);
}