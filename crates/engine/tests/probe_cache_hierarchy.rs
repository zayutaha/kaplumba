//! PROBE (read-only investigation, not a regression gate): the cache hierarchy
//! between the GPU and DRAM on the decode weight stream.
//!
//! # Scope, and the one thing this file is not
//!
//! No model, no checkpoint, no `metal::Device`, no dispatch, no GPU. Pure
//! arithmetic plus `include_str!` of the production Metal and Rust sources.
//! Every claim below is either (a) arithmetic on constants that are quoted from
//! a cited `file:line`, or (b) a **textual** fact about the source. Neither is a
//! measurement, and per AGENTS.md §3.4 a coverage claim is not a claim until the
//! code has been sabotaged and the test watched to fail. Nothing here has been
//! sabotaged, because this file touches no production code — so read every
//! assertion as "the sources say they will do this", not as "this is observed".
//!
//! # Verdict summary (the deliverable)
//!
//! | question | answer |
//! |---|---|
//! | is cache pollution measurable here? | **No.** Zero hardware-counter API is linked; there is no L1/L2 miss or allocation counter anywhere in the tree |
//! | does a load hint exist? | **No.** `mem_flags` has exactly one member and it is `mem_threadgroup`; no `cache_action`, no `read(`, no texture/sampler, no evict-priority anywhere in the Metal SDK headers either |
//! | vector loads on the weight stream? | **Impossible, and structurally so.** `j0` is even for only 16/32 lanes and ≡0 mod 4 for only 8/32 — at *every* integer rate 1–7 bpw exactly half are even — and 2 words/lane is a provable floor, because the 8 codes are 16-bit slices advancing 3 bits each, so a lane needs a 37-bit window. Reached independently by sibling `probe_vecld_window_widening.rs` |
//! | is the KV evicted by the weight stream? | **Yes, unavoidably, and it does not matter.** 12.04 GB/token is 380–1500 full L2 turnovers; the KV cross-token reuse is already dead before any hint could protect it |
//! | does the 512 MB segment matter? | **No, for cache.** It is a *placement* constraint: `tensor_buffer` rejects any tensor that is not exactly one segment at offset 0, so there is no boundary inside a read range |
//!
//! # HEAD note
//!
//! Probed at `8e3fc2b`. The brief said `5e8ab73`; `spec/decode-3bpw-amplification`
//! and `crates/engine/tests/perf_probe_kv_scaling.rs:33` both flag the same stale
//! pin, so treat any `5e8ab73` line number in the brief as unreliable and use the
//! anchors in the comments below.
//!
//! # Two corrections to the brief, both load-bearing
//!
//! 1. **`linear_gemv_ishizuki_k3_colmajor_contig4` is NOT the live decode
//!    kernel.** `linear.rs:2133` takes `_colmajor_2col` first when
//!    `perm_mode == 1 && out_features % 32 == 0`, and `out % 32 == 0` holds for
//!    every projection in this model, so the by-name `contig4` lookup at
//!    `linear.rs:1542` is never reached. `contig4` is analysed here too, because
//!    it has the identical inner loop, but the live kernel is `_2col`.
//! 2. **"881 MB at 512 positions" is the *allocated* footprint, not the live
//!    one.** 880,803,840 B is 50 sliding layers x 1024 positions + 10 full layers
//!    x 512 positions (`gpu_cache.rs` capacities, `model.rs:1077`
//!    `INITIAL_FULL_CACHE = 512`). At the 37-token benchmark context the *live*
//!    KV is 33,350,320 B — 3.8% of the allocation. Every KV byte outside that is
//!    never written and never read, so it cannot be "evicted".
//!
//! # Where this overlaps sibling probes — read before quoting
//!
//! Two other probes in this tree already cover adjacent ground and agree:
//!
//! - `probe_vecld_window_widening.rs` — the weight-window load width. It reaches
//!   the same `uint2` legal for 16/32 lanes and `uint4` for 8/32 by a bit-level
//!   trace rather than the modular arithmetic used here. **Do not count this
//!   file's Q4/Q3(c) answer as a second independent result on that sub-question.**
//! - `probe_actvec_read_amplification.rs` — the activation operand. It proposes
//!   the one concrete change below (as `uint` + unpack) and proves the alignment
//!   precondition; see the comment on `activation_loads_are_4_byte_alignable_and_collapse_2_to_1`
//!   for the one refinement this file adds.
//!
//! What is **not** duplicated anywhere I can find: the counter-API negative
//! (nothing in this tree can measure a cache effect at all), the `mem_flags`
//! exhaustive negative, and the segment-size/placement argument.

use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Sources. `include_str!` is deliberate: a moved file becomes a *compile*
// error rather than a green test reading nothing (the "fails open" failure mode
// called out three times in `spec/decode-3bpw-amplification/implementation_notes.md:240`).
// ---------------------------------------------------------------------------

const LINEAR: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED: &str = include_str!("../../kernels/src/linear_repacked.metal");
const ATTENTION: &str = include_str!("../../kernels/src/attention.metal");
const ELEMENTS: &str = include_str!("../../kernels/src/elements.metal");
const COLMAJOR_SWEEP: &str = include_str!("../../kernels/src/colmajor_sweep.metal");
const BW_PROBE: &str = include_str!("../../kernels/src/bw_probe.metal");
const BW: &str = include_str!("../../kernels/src/bw.metal");
const CB: &str = include_str!("../../kernels/src/cb.metal");
const ALU: &str = include_str!("../../kernels/src/alu.metal");
const EXL3: &str = include_str!("../../kernels/src/exl3.metal");

const WEIGHTS_RS: &str = include_str!("../src/weights.rs");
const METAL_RS: &str = include_str!("../src/metal.rs");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");

fn all_metal() -> Vec<(&'static str, &'static str)> {
    vec![
        ("linear.metal", LINEAR),
        ("linear_repacked.metal", REPACKED),
        ("attention.metal", ATTENTION),
        ("elements.metal", ELEMENTS),
        ("colmajor_sweep.metal", COLMAJOR_SWEEP),
        ("bw_probe.metal", BW_PROBE),
        ("bw.metal", BW),
        ("cb.metal", CB),
        ("alu.metal", ALU),
        ("exl3.metal", EXL3),
    ]
}

/// Body of `kernel void <name>(` up to the first line that is exactly `}`.
fn kernel_body(source: &str, name: &str) -> String {
    let open = format!("kernel void {name}(");
    let start = source
        .find(&open)
        .unwrap_or_else(|| panic!("no kernel named {name}"));
    let rest = &source[start..];
    let end = rest.find("\n}\n").unwrap_or_else(|| {
        panic!("no closing brace for {name}");
    });
    rest[..=end].to_string()
}

fn line_of(source: &str, needle: &str) -> usize {
    source
        .lines()
        .position(|line| line.contains(needle))
        .map(|index| index + 1)
        .unwrap_or_else(|| panic!("anchor not found: {needle}"))
}

/// Metal source with `//` line comments and `/* */` block comments removed.
///
/// **Required, not cosmetic.** `linear_repacked.metal:36` contains the word
/// "sampler" in a prose comment ("what the sampler reads"), and several other
/// cache words appear in explanatory comments that *document* the absence of a
/// feature. Scanning raw text therefore reports the very comments that argue
/// there is no hint as evidence that there is one. The comment bodies are kept
/// as a separate string so a text claim can still be anchored to a comment.
fn strip_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let mut index = 0usize;
    let mut in_block = false;
    while index < bytes.len() {
        if in_block {
            if bytes[index] == b'*' && bytes.get(index + 1) == Some(&b'/') {
                in_block = false;
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            in_block = true;
            index += 2;
            continue;
        }
        out.push(bytes[index] as char);
        index += 1;
    }
    out
}

/// The Metal sources with comments removed -- what is safe to scan for tokens
/// that must not appear in *code*.
fn all_metal_code() -> Vec<(&'static str, String)> {
    all_metal()
        .into_iter()
        .map(|(name, source)| (name, strip_comments(source)))
        .collect()
}

/// Every `mem_flags::<member>` appearing in Metal *code* (comments stripped).
///
/// Factored out so `no_load_hint_mechanism_exists_in_the_language_or_in_this_tree`
/// and `the_cache_negative_predicates_can_actually_fail` run the *same* function;
/// a mutation test that re-implements the thing it is testing proves nothing.
fn mem_flags_members(code: &str) -> BTreeSet<String> {
    let mut members: BTreeSet<String> = BTreeSet::new();
    let mut from = 0usize;
    while let Some(at) = code[from..].find("mem_flags::") {
        let tail = &code[from + at + "mem_flags::".len()..];
        let member: String =
            tail.chars().take_while(|c| c.is_ascii_lowercase() || *c == '_').collect();
        if !member.is_empty() {
            members.insert(member);
        }
        from += at + 1;
    }
    members
}

/// Same, over every production `.metal` file with comments stripped.
fn mem_flags_members_all() -> BTreeSet<String> {
    let mut all = BTreeSet::new();
    for (_, code) in all_metal_code() {
        all.extend(mem_flags_members(&code));
    }
    all
}
// (`linear_repacked.metal`, kernel at the anchor asserted in
// `live_kernel_is_2col_not_contig4`).
// ---------------------------------------------------------------------------

/// `constexpr uint KA = 3u;` — the integer EXL3 rate of every layer projection.
const KA: u32 = 3;
/// `constexpr uint RING = 256u * KA;` — bits per 16x16 code tile.
const RING: u32 = 256 * KA;
/// `constexpr uint WORDS = RING / 32u;` — 32-bit words per tile.
const WORDS: u32 = RING / 32;
/// Bytes one 16x16 tile occupies. **96, not 128.**
const TILE_BYTES: u32 = WORDS * 4;
/// Lanes in an Apple simdgroup. `const uint lane = tid & 31u`.
const LANES: u32 = 32;
/// A lane owns 8 of the tile's 256 codes.
const CODES_PER_LANE: u32 = 8;

/// `linear_repacked.metal` `_2col`: `lo0 = (lane * 8u + 1u) * 3u + RING - 16u`.
///
/// The bit offset, within the tile's 768-bit ring, of lane `lane`'s 8th code.
/// This is the whole sliding window: `RING - 16u` walks backwards from the ring
/// end so the last code of the lane lands inside it.
fn lo0(lane: u32) -> u32 {
    (lane * CODES_PER_LANE + 1) * KA + RING - 16
}

/// Word index and intra-word shift of lane `lane`'s window.
fn window(lane: u32) -> (u32, u32, u32) {
    let lo = lo0(lane);
    let sh0 = lo % 32;
    let hi0 = lo / 32;
    let j0 = if hi0 >= WORDS { hi0 - WORDS } else { hi0 };
    let j1 = if j0 + 1 >= WORDS { j0 + 1 - WORDS } else { j0 + 1 };
    (sh0, j0, j1)
}

// ---------------------------------------------------------------------------
// The brief's measured baseline. `spec/decode-ceiling/hypotheses.md:26-27`,
// and independently `crates/engine/tests/perf_probe_kv_scaling.rs:70`
// (`per_token_weight_bytes() == 12_039_487_888`, asserted at `bench.rs:331`).
// ---------------------------------------------------------------------------

/// Bytes of weight read per decode token. 10.983 GB layers + 1.057 GB lm_head.
const BYTES_PER_TOKEN: u64 = 12_039_487_888;
/// `hypotheses.md:15` — the brief's 62.3 ms/step.
const STEP_MS: f64 = 62.3;
/// `AGENTS.md` §8.
const WIRE_GBPS: f64 = 273.0;

// ---------------------------------------------------------------------------
// 1. Is cache pollution measurable here?
// ---------------------------------------------------------------------------

/// **No counter API is linked, so no cache effect in this tree can be measured
/// without adding one.**
///
/// `MTLCounterSet` / `MTLDevice.sampleCounterBuffer` is Metal's only route to
/// L1/L2 miss, sector and eviction counters. Grepping the whole crate tree for
/// it returns nothing, and the `metal` crate's own API surface is not exercised
/// for it anywhere. So `KAPLUMBA_GPU_PROFILE=1`
/// (`crates/engine/src/decode_profile.rs:5`) is a **per-dispatch timestamp
/// table**, not hardware counters: it can attribute milliseconds to a kernel and
/// nothing can attribute them to a cache level.
///
/// The consequence for the brief: "193 GB/s of allocation pressure" is a
/// *derivation* from bytes/step, not an observation. It is a good derivation —
/// but nothing in this box would notice if the allocation were free, and
/// nothing would notice if it were 3x worse.
#[test]
fn no_hardware_counter_api_is_linked_so_cache_pollution_is_unmeasurable() {
    for (name, source) in all_metal_code() {
        for forbidden in [
            "MTLCounterSet",
            "CounterSampleBuffer",
            "sampleCounterBuffer",
            "MTLCommonCounter",
            "counterSampleBuffer",
            "access::read",
        ] {
            assert!(
                !source.contains(forbidden),
                "{name} references {forbidden}; the unmeasurable claim in this \
                 file's header would need revisiting"
            );
        }
    }
    // Same for the Rust side: no `sample_counter_buffer` call in the engine or
    // the kernel crate.
    for source in [WEIGHTS_RS, METAL_RS, LINEAR_RS] {
        assert!(!source.contains("sample_counter"));
        assert!(!source.contains("CounterSet"));
    }

    // The only profiler in the tree is timestamp-based. Assert it says so, so
    // that "unmeasurable" is anchored to a positive fact and not only to the
    // absence of one.
    assert!(
        LINEAR_RS.contains("trace_gemv"),
        "the dispatch tracer must still be the only profiling hook"
    );
}

/// The allocation-pressure arithmetic, spelled out so the number in the brief is
/// reproducible rather than asserted.
#[test]
fn allocation_pressure_is_a_derivation_from_bytes_per_step_and_not_a_reading() {
    let gbps = BYTES_PER_TOKEN as f64 / (STEP_MS * 1.0e6);
    assert!(
        (gbps - 193.3).abs() < 0.5,
        "12.04 GB / 62.3 ms = {gbps:.1} GB/s, expected ~193"
    );
    assert!(
        gbps < WIRE_GBPS,
        "if the stream ever reaches the wire it is not allocation-bound, and \
         the whole framing changes"
    );

    // Cache lines are a *quantisation* of this figure and the machine's line
    // size is not published by Apple, so state the answer as a range over the
    // two sizes that bracket every AGX datapath rather than picking one. AGENTS.md
    // §10.3 is explicit about this class of mistake: guessing a number and
    // presenting it as measured is the failure mode, not the arithmetic.
    for line in [64u64, 128] {
        let lines_per_token = BYTES_PER_TOKEN / line;
        let per_second = lines_per_token as f64 / (STEP_MS / 1000.0);
        // A 128 B line over 62.3 ms is 1.51e9 lines/s; on 20 cores at ~4 GHz
        // that is ~1.9 lines per cycle per core to *allocate*, sustained, on
        // every core -- and every one of them is a compulsory miss, because the
        // stream has no reuse (see `dram_traffic_is_exactly_1x_and_the_2_667x_is_requests_only`).
        // That is the real content of "193 GB/s of allocation pressure": it is
        // not free, and it is not obviously cheap either.
        assert!(
            per_second > 1.0e9,
            "{line} B lines: {per_second:.3e} allocations/s"
        );
        // At ~4 GHz across 20 cores the machine can retire 80e9 line
        // allocations/s if it had to, so the *rate* is not the issue -- the
        // issue would be the L2 tag/sector bandwidth they consume, which Metal
        // gives no way to read (test 1). Say so rather than implying the rate
        // alone proves a bottleneck.
        let budget_per_cycle_per_core = 80.0e9 / (per_second);
        assert!(
            budget_per_cycle_per_core > 1.0,
            "line allocations per cycle per core: {budget_per_cycle_per_core:.3}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Does a load hint exist?  Definitive negative.
// ---------------------------------------------------------------------------

/// **No. There is no streaming / non-temporal / evict-first load hint for
/// compute shaders in Metal, and this tree uses none of what little exists.**
///
/// The search is exhaustive over the ten production `.metal` files plus the
/// macOS SDK's Metal framework headers:
///
/// - `mem_flags` — the MSL enum whose only members are `mem_none`,
///   `mem_device`, `mem_threadgroup_imageblock` and `mem_threadgroup`. It is
///   the *only* memory-qualifier in the language and it appears in
///   `threadgroup_barrier` / `simdgroup_barrier` / `mem_flags`-scoped fences. It
///   is **not** a cache hint on a load and cannot be used as one.
/// - `read()`, `write()`, `read_action`, `write_action`, `cache_action` — these
///   are Metal 3 texture/`access::read` forms. Absent here, and absent from the
///   installed SDK headers (searched below).
/// - Eviction priority (`evict_first` / `evict_last` / anything containing
///   `evict`) — **zero hits across the entire Metal framework.** Apple's only
///   related API is `MTLResource.setPurgeableState`, which is whole-resource,
///   host-side, and would evict a 12 GB arena out from under the GPU.
/// - `texture2d` + `sampler` (the brief's option b) — absent. `MTLPixelFormat.h`
///   has no cache-layer or eviction enum either, so option (b) would not buy a
///   different cache path even if it were written.
#[test]
fn no_load_hint_mechanism_exists_in_the_language_or_in_this_tree() {
    // 2a. `mem_flags` — the only qualifier in the language — has exactly one
    // member used here, and it is a barrier scope.
    let members = mem_flags_members_all();
    assert_eq!(
        members.into_iter().collect::<Vec<_>>(),
        vec!["mem_threadgroup".to_string()],
        "a new mem_flags member appeared; re-read this test's claim"
    );

    // 2b. Nothing that names a cache policy, a texture, or a sampler -- in
    // *code*. Comments are stripped, because the comments in these files argue
    // at length that none of these exist and would otherwise self-trigger.
    for (name, source) in all_metal_code() {
        for forbidden in [
            "cache_action",
            "read_action",
            "write_action",
            "evict_first",
            "evict_last",
            "non_temporal",
            "nontemporal",
            "streaming_load",
            "cache_hint",
            "texture2d",
            "texture2d_array",
            "texturecube",
            "sampler",
            "L1_no_allocate",
            "L2_evict",
        ] {
            assert!(
                !source.contains(forbidden),
                "{name} contains {forbidden}: if a hint is now in use, the \
                 negative result in this file is stale"
            );
        }
    }

    // 2c. Positive anchor: the qualifier is used only for its one legal job. Every
    // `threadgroup_barrier` in the three hot files carries it -- there is not a
    // single unqualified barrier -- so `mem_flags` is being used as a barrier
    // scope and nothing else.
    let mut barriers = 0usize;
    let mut qualified = 0usize;
    for source in [REPACKED, ATTENTION, ELEMENTS] {
        barriers += source.matches("threadgroup_barrier(").count();
        qualified += source
            .matches("threadgroup_barrier(mem_flags::mem_threadgroup)")
            .count();
    }
    assert!(barriers >= 60, "only {barriers} barriers scanned");
    assert_eq!(
        barriers, qualified,
        "every threadgroup_barrier must carry mem_flags::mem_threadgroup"
    );
}

/// The SDK half of the negative, run only where the SDK is present so the test
/// does not become a machine-configuration failure. On this box it *is* present
/// (`MacOSX27.0.sdk`), and it finds nothing.
#[test]
fn the_metal_sdk_headers_contain_no_eviction_or_cache_action_api() {
    let Ok(sdk) = std::env::var("KAPLUMBA_PROBE_SDK") else {
        eprintln!("set KAPLUMBA_PROBE_SDK to $(xcrun --sdk macosx --show-sdk-path) to run this");
        return;
    };
    let headers = format!("{sdk}/System/Library/Frameworks/Metal.framework/Headers");
    let mut checked = 0usize;
    for entry in std::fs::read_dir(&headers).expect("Metal.framework Headers") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("h") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("header");
        checked += 1;
        let lowered = text.to_ascii_lowercase();
        for forbidden in ["evict", "cacheaction", "non_temporal"] {
            assert!(
                !lowered.contains(forbidden),
                "{} now contains {forbidden}; Metal may have grown a cache hint",
                path.display()
            );
        }
        // What *is* there, so the negative is not vacuous: whole-resource,
        // host-side purging and a CPU write-combined mode.
        if path.file_name().and_then(|n| n.to_str()) == Some("MTLResource.h") {
            assert!(text.contains("MTLPurgeableState"));
            assert!(text.contains("MTLCPUCacheModeWriteCombined"));
        }
    }
    assert!(checked > 10, "only {checked} headers scanned");
}

// ---------------------------------------------------------------------------
// 3/4. The load stream itself: width, amplification, and the vector-load floor.
// ---------------------------------------------------------------------------

/// Which kernel is live. `linear.rs:2133` short-circuits to `_2col`.
#[test]
fn live_kernel_is_2col_not_contig4() {
    let at = line_of(LINEAR_RS, "out_features % 32 == 0 && !env_flag(\"KAPLUMBA_NO_2COL\")");
    let arm = &LINEAR_RS.lines().nth(at + 1).expect("arm line").to_string();
    assert!(
        arm.contains("colmajor_2col"),
        "the first arm must still be _2col: {arm}"
    );
    // And `contig4` is only reached by the by-name fallback further up, which
    // this arm pre-empts whenever `out % 32 == 0` (always, for this model).
    assert!(LINEAR_RS.contains("linear_gemv_ishizuki_k{bits}_colmajor_contig4"));
    assert!(kernel_exists(REPACKED, "linear_gemv_ishizuki_k3_colmajor_2col"));
    assert!(kernel_exists(REPACKED, "linear_gemv_ishizuki_k3_colmajor_contig4"));
}

fn kernel_exists(source: &str, name: &str) -> bool {
    source.contains(&format!("kernel void {name}("))
}

/// **The load width is 32-bit, and it is already at the floor.**
///
/// `ring_word<true>` (`linear.metal`) is `((const device uint *)packed)[index]` —
/// one 32-bit load. The live kernel does not even call it: `_2col` inlines the
/// two-word read as `t0_0[j0], t0_0[j1]` on a `const device uint *`. So the
/// brief's question 4, "count the current load width in `ring_word` /
/// `trellis_word`", has the answer **4 bytes, everywhere, on the decode path**.
#[test]
fn the_decode_path_already_loads_32_bit_words_and_not_2_or_4_byte_scalars() {
    // `trellis_word` (linear.metal) is the 2x16-bit fallback for unaligned
    // data. It must still exist -- it is load-bearing for mapped checkpoints --
    // but the decode kernel must not be using it.
    let tw = kernel_or_fn_body(LINEAR, "static inline uint trellis_word");
    assert!(tw.contains("packed[index * 2u]"));
    assert!(tw.contains("packed[index * 2u + 1u]"));
    assert!(tw.contains("<< 16"), "trellis_word assembles the word big-endian");

    // `ring_word<true>`: the one 32-bit load.
    let rw = kernel_or_fn_body(LINEAR, "static inline uint ring_word");
    assert!(rw.contains("((const device uint *)packed)[index]"));

    // The live kernel casts to `const device uint *` itself and indexes words.
    let k2 = kernel_body(REPACKED, "linear_gemv_ishizuki_k3_colmajor_2col");
    assert!(k2.contains("(const device uint *)packed"));
    assert!(k2.contains("const uint stride = WORDS"));
    assert!(k2.contains("t0_0[j0]"), "the two-word read must be inlined words");

    // And `ring_word` / `exl3_decode8` are *not* in the live kernel: the decode
    // path inlines the same window arithmetic.
    assert!(!k2.contains("ring_word"));
    assert!(!k2.contains("exl3_decode8"));
}

fn kernel_or_fn_body(source: &str, signature: &str) -> String {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("no {signature}"));
    let rest = &source[start..];
    let end = rest.find("\n}\n").unwrap_or_else(|| panic!("no closing brace after {signature}"));
    rest[..=end].to_string()
}

/// **DRAM traffic is exactly 1x, and the 2.667x is requests only.**
///
/// Per 16x16 tile: 24 words / 96 bytes on the wire. Each of the 32 lanes reads
/// the two words at its own window offset, and the union of those 32 pairs
/// covers **all 24 words** — so every byte fetched is used, and no byte is
/// fetched twice from DRAM. That is what makes a cache-pollution fix
/// *impossible to value from DRAM bytes*: there are no wasted bytes to remove.
///
/// This independently reproduces `rate4_colmajor_bit_exact.rs`'s
/// `dram_line_traffic_is_1x_at_every_rate_so_the_amplification_is_requests_only`
/// and the derivation at `spec/decode-3bpw-amplification/implementation_notes.md:162-178`.
#[test]
fn dram_traffic_is_exactly_1x_and_the_2_667x_is_requests_only() {
    assert_eq!(TILE_BYTES, 96, "3 bpw tile is 96 B -- not 128");

    let mut covered: BTreeSet<u32> = BTreeSet::new();
    let mut requests = 0u32;
    let mut shifts: BTreeSet<u32> = BTreeSet::new();

    for lane in 0..LANES {
        let (sh0, j0, j1) = window(lane);
        covered.insert(j0);
        covered.insert(j1);
        requests += 2;
        shifts.insert(sh0);
    }

    assert_eq!(
        covered.len() as u32,
        WORDS,
        "lanes must cover all {WORDS} words, else DRAM over-fetches: {covered:?}"
    );
    assert_eq!(requests, 64);
    let amp = requests as f64 / WORDS as f64;
    assert!(
        (amp - 2.666_666_7).abs() < 1e-6,
        "amplification is {amp}, expected 64/24"
    );

    // No lane's window is word-aligned, so every one of them needs the real
    // funnel shift. This is what makes the window structurally 2 words wide.
    assert_eq!(
        shifts,
        [3u32, 11, 19, 27].into_iter().collect::<BTreeSet<_>>(),
        "sh0 must be {{3,11,19,27}}; 0 or 16 would mean a free single word"
    );

    // 96 B of tile means the 2.667x never reaches DRAM, and a 128 B line spans
    // two tiles, so no line is fetched for bytes that are not used.
    assert!(TILE_BYTES < 128);
    assert_eq!(TILE_BYTES % 16, 0, "a 96 B tile is 6 uint4 -- alignment permits that");
}

/// **Vector loads on the weight stream are structurally impossible, three ways
/// over. This is the deliverable arithmetic for the brief's question 4.**
///
/// Asking for `ulong2` (8 B) or `uint4` (16 B) requires natural alignment of the
/// *tile-relative word offset*, because the tile base is itself aligned:
///
/// - `stride = WORDS = 24` words = 96 B, and `tn * (tiles_k * WORDS)` is
///   `tn * tiles_k * 96` — 8- and 16-byte aligned for every `tn`. So the base
///   is never the problem; `j0` is.
/// - **`ulong2` is legal for 16 of 32 lanes.** `j0 % 2 == 0` holds for exactly
///   16. The other 16 are at a 4-byte-aligned address, and Metal requires
///   natural alignment for a vector load, so the load is ill-formed — not slow,
///   not miscompiled: ill-formed.
/// - **`uint4` is legal for 8 of 32 lanes.** `j0 % 4 == 0` holds for exactly 8.
///   Worse, even for those 8 it does not help: the lane wants 2 of the 4 words
///   it would fetch, and the 6 `uint4` that cover a 96 B tile are *all* wanted by
///   the simdgroup, so widening trades 2 requests for 4x the bytes per request on
///   8 lanes and 12x on 24.
/// - Rounding `j0` down to an even word does not rescue it: for the 16 odd lanes
///   that yields `(j0-1, j0)` and the wanted `j1 = j0+1` is then in the *next*
///   load, so it is 2 loads again.
#[test]
fn vector_loads_on_the_weight_stream_are_illegal_for_half_and_three_quarters_of_the_lanes() {
    let mut even = 0;
    let mut on_u16 = 0;
    let mut distinct_j0: BTreeSet<u32> = BTreeSet::new();
    for lane in 0..LANES {
        let (_, j0, _) = window(lane);
        distinct_j0.insert(j0);
        if j0 % 2 == 0 {
            even += 1;
        }
        if j0 % 4 == 0 {
            on_u16 += 1;
        }
    }
    assert_eq!(even, 16, "uint2 (ulong2) is 8-byte aligned for only 16/32 lanes");
    assert_eq!(on_u16, 8, "uint4 is 16-byte aligned for only 8/32 lanes");
    // And the offsets are genuinely scattered, not clustered: a repack that made
    // them even would have to change the layout, which T1 shows is impossible.
    assert_eq!(distinct_j0.len() as u32, 24, "24 distinct j0 across 32 lanes");

    // The alignment precondition, stated positively: the tile base IS aligned,
    // so the whole question reduces to `j0`.
    assert_eq!(TILE_BYTES % 16, 0);
    assert_eq!(WORDS * 4 % 16, 0);
}

/// **The floor is 2 words per lane, and that is arithmetic, not a guess.**
///
/// `decode_state_mcg` is
/// `value = value*0xCBAC1FED; value = 0x3b603b60 ^ (value & 0x8fff8fff);`
/// — it multiplies **before** masking, so all 16 bits of the window reach the
/// output and a lane cannot be reconstructed from its own 8 codes. The union of
/// the bits any lane depends on is **37 bits**, which exceeds one 32-bit word, so
/// every lane unconditionally needs 2 words and
/// `amp = 64 / (8 * KA)` — which is what the kernel already does.
///
/// This is `spec/decode-3bpw-amplification/tasks.md` **T1**, which is checked
/// and reports the same 37 from three directions. So the brief's
/// "does the access pattern allow vector loads" is answered *no, and no relayout
/// can change it*.
#[test]
fn two_words_per_lane_is_a_provable_floor_because_decode_consumes_37_bits() {
    // `extract_v_mcg_32` (linear_repacked.metal) builds a 64-bit window from
    // the two words and then reads code i as a **16-bit slice** of it:
    //
    //   v[0..5] = decode_state_mcg((A_hi >> (16 - 3i)) & 0xffff)   i = 0..5
    //   v[6]    = decode_state_mcg(((A_hi << 2) | (A_lo >> 30)) & 0xffff)
    //   v[7]    = decode_state_mcg(((A_hi << 5) | (A_lo >> 27)) & 0xffff)
    //
    // So code i touches window bits [16 - 3i, 16 - 3i + 16). The slices are 16
    // bits wide but advance by only `KA` = 3 bits each, so the union over the 8
    // codes is a span of
    //
    //     16 + (CODES_PER_LANE - 1) * KA  =  16 + 7*3  =  37 bits.
    //
    // 37 > 32, so one 32-bit word provably cannot supply a lane, and
    // `ceil(37/32) = 2` words is a floor. That is the whole negative: the
    // kernel already sits on the floor and no relayout can move it.
    //
    // This is `spec/decode-3bpw-amplification/tasks.md` T1, which reports 37
    // from three directions (bit-flip enumeration, symbolic window span, union
    // over all 32 lanes). Derived here rather than quoted because the number is
    // load-bearing for every load-width conclusion in this file.
    let slice_width = 16u32;
    let span = slice_width + (CODES_PER_LANE - 1) * KA;
    assert_eq!(span, 37, "37 bits per lane");
    assert!(span > 32, "one word is provably insufficient");
    let words = span.div_ceil(32);
    assert_eq!(words, 2, "2 words per lane is a floor");

    // And the amplification that follows, which is the quantity T1 shows is set
    // by the bit rate and not by the layout:
    assert_eq!(LANES * 2, 64, "64 lane-word requests per tile");
    assert_eq!(WORDS, 24, "24 words of tile");
    let amp = 64.0 / (CODES_PER_LANE * KA) as f64;
    assert!((amp - 2.666_666_7).abs() < 1e-6, "amp = 64/(8*3) = {amp}");

    // The multiply-before-mask detail that makes this structural rather than a
    // layout choice. `decode_state_mcg` is
    //   value = value * 0xCBAC1FED;  value = 0x3b603b60 ^ (value & 0x8fff8fff);
    //
    // A plausible-looking shortcut here is to argue from the mask: it clears 6
    // bits, so the slice is 10 effective bits and the span is 10 + 7*3 = 31,
// which is <= 32 and would NOT prove the floor. That argument is wrong, and it
    // is worth asserting that it is wrong rather than leaving the trap in place:
    // the mask is applied to the *product*, after a multiply that has already
    // consumed all 16 slice bits. So the mask buys nothing and the span is the
    // full 16-bit slice width, i.e. the 37 above.
    let f = |value: u32| {
        let value = value.wrapping_mul(0xCBAC1FEDu32);
        0x3b603b60u32 ^ (value & 0x8fff8fffu32)
    };
    let mask = 0x8fff8fffu32;
    let cleared = 32 - mask.count_ones();
    assert_eq!(cleared, 6, "0x8fff8fff clears 6 of 32 product bits");
    let mask_only_width = 16u32 - cleared;
    assert_eq!(mask_only_width, 10);
    assert!(
        mask_only_width + (CODES_PER_LANE - 1) * KA <= 32,
        "the mask-based argument would give {} bits -- under one word, so it \
         does NOT establish the floor and must not be used",
        mask_only_width + (CODES_PER_LANE - 1) * KA
    );
    // The argument that does hold: all 16 slice bits reach the output.
    assert_ne!(mask, u32::MAX);
    let moving = (0..16u32).filter(|bit| f(1u32 << bit) != f(0)).count();
    assert_eq!(moving, 16, "all 16 slice bits reach the output, mask or no mask");
    // Deterministic.
    assert_eq!(f(0), f(0));
}

/// **Prior art: both vector-load families are already measured and both LOST.**
///
/// Not a proposal — a record, so the next person does not re-run them:
///
/// | variant | result |
/// |---|---|
/// | `simd_shuffle` cooperative loads (share one tile's 24 words across lanes) | **180.5 → 146.5 GB/s** |
/// | `_u4` `uint4`, threadgroup-staged | **180.5 → 173.7 GB/s** |
/// | `_u8` deeper unroll | 1.01–1.03x |
/// | relayout to raise amp | impossible (T1, above) |
///
/// `spec/decode-19tok-repacked-250gb/tasks.md:56,74` and
/// `spec/decode-3bpw-amplification/implementation_notes.md:148-153`.
///
/// The stated cause of the `_simd` loss is the right one and worth repeating:
/// *dynamic* per-lane shuffles serialise through the SIMD crossbar. That is the
/// generic failure mode of any scheme that turns a coalesced load into a
/// scatter-plus-shuffle.
#[test]
fn both_vector_load_families_are_present_and_closed() {
    // The `_u4` kernel still exists in the tree. It is threadgroup-STAGED
    // (3072 B/tg, 2 barriers per 4 tiles) which is *why* it lost -- so it is
    // not evidence about load width alone, and a clean test of direct `uint4`
    // has genuinely never been run. Recorded here so that is not mistaken for
    // "uint4 was tested and lost".
    let u4_at = line_of(REPACKED, "kernel void linear_gemv_ishizuki_k3_colmajor_u4(");
    assert!(u4_at > 0, "the _u4 kernel must still be findable by name");
    let u4 = kernel_body(REPACKED, "linear_gemv_ishizuki_k3_colmajor_u4");
    assert!(
        u4.contains("threadgroup"),
        "_u4 stages through threadgroup memory; that staging is the confound"
    );
    assert!(u4.contains("uint4"), "_u4 is the 128-bit-load experiment");

    // The reverted cooperative-load kernel is gone from the decode path, and the
    // source says so in a comment. Assert the comment survives, because the
    // number 146.5 only means something if the kernel is known to be dead.
    assert!(
        REPACKED.contains("146.5"),
        "the reverted simd_shuffle experiment's measured number must stay in \
         the source comment so it is not re-proposed"
    );
}

// ---------------------------------------------------------------------------
// 4b. Instruction accounting, so the "1.8%" below is derived not guessed.
// ---------------------------------------------------------------------------

/// **The one concrete change: halve the activation loads in `_2col`'s inner
/// loop. 4 of 240 instructions per iteration, and it is bit-exact by
/// construction.**
///
/// **NOT NEW, and it is credited.** Sibling probe
/// `probe_actvec_read_amplification.rs:635` (`x_loads_are_two_aligned_uint_pairs`)
/// proposes exactly this change and independently proves the alignment
/// precondition. This test is therefore not a second source for the idea, and it
/// must not be counted twice. What is added here:
///
/// 1. **A refinement: `half2`, not `uint` + unpack.** The sibling writes it as
///    "2 uint loads + 2 unpacks". `as_type<half2>` reads `.x` and `.y` as
///    register subwords with no instruction at all, so the `uint` form pays two
///    extra unpacks for nothing:
///    ```
///    const device half2 *x2 = (const device half2 *)x + (tk + 0u) * 16u + r0;
///    const half2 q0 = x2[0];   // = (half r0,   half r0+1)
///    const half2 q1 = x2[4];   // = (half r0+8, half r0+9)
///    ```
///    4 loads + 0 ALU, against 4 loads + 4 ALU. Same 2-vs-4 load saving, strictly
///    less of everything else.
/// 2. The instruction budget (below), so the saving is a number and not a hope.
///
/// Current, `linear_repacked.metal` `_2col` unrolled body, per iteration
/// (2 K-substeps x 2 tile columns = 4 tiles = 32 codes):
///
/// ```
/// const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
/// const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
/// ```
///
/// Eight `half` loads, **all of which resolve to two distinct addresses per
/// simdgroup** (`r0 = (lane % 4) * 2`, so lanes 0-3 share `r0`=0,2,4,6 and
/// lanes 4-31 repeat them: a broadcast, not a gather).
///
/// **Legality.** A `half2` load needs 4-byte alignment. The byte address is
/// `2 * (tk * 16 + r0) = 32*tk + 2*r0`, and `r0 in {0,2,4,6}` so `2*r0 in
/// {0,4,8,12}` — divisible by 4 for every lane. `x` is `transformed` at
/// buffer offset 0 (`linear.rs:2147` `TensorRef::new(transformed, 0)`), and a
/// Metal buffer is page-aligned. So the load is aligned for all 32 lanes, with
/// no predicate and no select. (The sibling proves the same thing at
/// `probe_actvec_read_amplification.rs:642-644`.)
///
/// **Bit-exactness.** Same bits, moved. `x2[0].x` is the same `half` that
/// `xr0[0]` read; nothing is summed in a different order and no arithmetic
/// changes. The 16 accumulators and every `+=` stay exactly as they are — which
/// matters here because `spec/decode-19tok-repacked-250gb` records two separate
/// FMA-reassociation bit-exactness failures on this exact loop.
///
/// **Why it is worth doing at all.** It is small: 4 loads out of ~240
/// instructions per iteration (~1.7% of issue, ~0.4% of the step). It is on
/// this list anyway because it is the only load-side change in the whole angle
/// that survives all three negatives above, it needs no repack, no barrier, and
/// no quality decision.
#[test]
fn activation_loads_are_4_byte_alignable_and_collapse_2_to_1() {
    // The current form: four indexed `half` reads per activation tile.
    let k2 = kernel_body(REPACKED, "linear_gemv_ishizuki_k3_colmajor_2col");
    let anchored = k2
        .lines()
        .filter(|line| line.contains("float(xr0[0])"))
        .count()
        + k2
            .lines()
            .filter(|line| line.contains("float(xr1[0])"))
            .count();
    // The exact change sites, by absolute line number in `linear_repacked.metal`.
// Three of them: `xr0`/`xr1` once each in the unrolled 4-tile body and `xr0`
    // once in the `SG`-stride remainder loop. The vectorisation applies to all
    // three identically. (There are 25 such sites across the whole family;
    // these three are the ones in the live kernel.)
    assert_eq!(
        anchored, 3,
        "xr0 and xr1 in the unrolled body, xr0 in the remainder loop"
    );
    for (line, text) in [
        (2109u32, "const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);"),
        (2111, "const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);"),
        (2141, "const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);"),
    ] {
        let actual = REPACKED.lines().nth(line as usize - 1).unwrap_or("");
        assert_eq!(
            actual.trim(),
            text,
            "the change site at linear_repacked.metal:{line} moved"
        );
    }
    // And `r0` itself, the sole determinant of the alignment.
    assert_eq!(
        REPACKED.lines().nth(2075 - 1).unwrap_or("").trim(),
        "const uint r0 = (lane % 4u) * 2u;",
        "r0 moved; the 4-byte alignment proof depends on it"
    );

    // `r0` is the only thing that decides alignment, and it is even by
    // construction. This is the load-bearing check: if `r0` could be odd the
    // `half2` load would be ill-formed.
    let r0_at = line_of(REPACKED, "const uint r0 = (lane % 4u) * 2u;");
    assert!(r0_at > 0);
    for lane in 0..LANES {
        let r0 = (lane % 4) * 2;
        assert!(r0 % 2 == 0, "lane {lane}: r0 {r0} is odd -> half2 misaligned");
        assert_eq!(
            (32u32 * 7 + 2 * r0) % 4,
            0,
            "lane {lane}: byte address 32*tk + 2*r0 must be 4-byte aligned"
        );
    }

    // Instruction accounting for one unrolled iteration of `_2col`.
    let tiles = 4u32;
    let codes = tiles * CODES_PER_LANE;
    let weight_loads_now = tiles * 2; // two uint words per tile
    let act_loads_now = 2 * 4; // xr0[0,1,8,9] and xr1[0,1,8,9]
    let act_loads_after = 2 * 2; // half2 at +0 and +4 per activation tile
    let decodes = codes; // one decode_state_mcg per code
    let fmas = codes; // one fma per code
    // decode_state_mcg: imul, and, xor, as_type, add.f16, cvt.f32.f16.
    let alu_per_decode = 6u32;
    let total_before = weight_loads_now + act_loads_now + decodes * alu_per_decode + fmas;
    let total_after = weight_loads_now + act_loads_after + decodes * alu_per_decode + fmas;

    assert_eq!(codes, 32);
    assert_eq!(weight_loads_now, 8);
    assert_eq!(act_loads_now, 8);
    assert_eq!(total_before, 8 + 8 + 32 * 6 + 32);
    assert_eq!(total_before, 240);
    assert_eq!(total_before - total_after, 4);
    let share = (total_before - total_after) as f64 / total_before as f64;
    assert!(
        (share - 0.016_666_6).abs() < 1e-6,
        "4/240 = {share:.4} of the unrolled body's instructions"
    );

    // And the honest framing: even a *free* activation load is worth ~0.4% of a
    // 62.3 ms step, because the GEMV is not load-issue bound. ~515 Gcodes/s at
    // 7 instructions per code is ~3.6 Top/s of mixed issue against a ~2.6 T/s
    // fp32 issue roofline on 20 cores -- i.e. already over it.
    let codes_per_s = (BYTES_PER_TOKEN as f64 * 8.0 / 3.0) / (STEP_MS / 1000.0);
    let gcodes_per_s = codes_per_s / 1.0e9;
    assert!(
        (gcodes_per_s - 515.1).abs() < 3.0,
        "code rate {gcodes_per_s:.1} Gcodes/s"
    );
    assert!(
        codes_per_s * 7.0 > 2.0e12,
        "the inner loop is issue-bound, not load-bound -- which is why the \
         load-angle wins above are all ~1%"
    );
}

// ---------------------------------------------------------------------------
// 7. Sabotage checks. AGENTS.md §3.4: a coverage claim is not a claim until
//    the code has been broken and the test watched to fail.
// ---------------------------------------------------------------------------

/// The predicates above are run against deliberately mutated copies of the
/// sources, and each must FIRE. Without this, "the scan found nothing" and "the
/// scan cannot find anything" are indistinguishable from the outside — which is
/// the exact failure this repo has shipped twice
/// (`crates/engine/tests/linear_metal.rs:900`, and the env-flag test that
/// disabled the path under test).
#[test]
fn the_cache_negative_predicates_can_actually_fail() {
    // (a) The forbidden-token scan is sensitive. Inject each hint keyword into
    // the live kernel and confirm the scan sees it.
    let forbidden = [
        "cache_action",
        "read_action",
        "write_action",
        "evict_first",
        "evict_last",
        "non_temporal",
        "cache_hint",
        "texture2d",
        "sampler",
    ];
    let anchor = "kernel void linear_gemv_ishizuki_k3_colmajor_2col(";
    assert!(REPACKED.contains(anchor));
    for token in forbidden {
        let mutated =
            REPACKED.replacen(anchor, &format!("{anchor} // {token}\n    // sabotage"), 1);
        assert!(mutated.contains(token), "mutation for {token} did not land");
        assert!(
            strip_comments(&mutated).contains(token) == false,
            "{token}: the scan strips comments, so a comment-only mutation must \
             NOT fire -- this is the property that makes the real negative valid"
        );
        // And a mutation inside *code* (not a comment) must fire.
        let coded = REPACKED.replacen(
            "const device uint *col0 = (const device uint *)packed",
            &format!("const device uint *col0 = (const device uint *)packed; const int {token} = 0;"),
            1,
        );
        assert!(strip_comments(&coded).contains(token), "code mutation for {token} failed to land");
    }

    // (b) The strip is load-bearing, not decorative: the *raw* source really
    // does contain the word the test says is absent from the code.
    assert!(
        REPACKED.contains("sampler"),
        "the comment that makes the strip necessary has gone; re-check the strip"
    );
    assert!(!strip_comments(REPACKED).contains("sampler"));

    // (c) The counter scan is sensitive. Injected into *code*, because the scan
    // strips comments and a counter named only in prose is not an API call --
    // the same rule the hint scan uses.
    let with_counter = REPACKED.replacen(
        "const device uint *col0 = (const device uint *)packed",
        "const device uint *col0 = (const device uint *)packed; const int MTLCounterSet = 0;",
        1,
    );
    assert!(strip_comments(&with_counter).contains("MTLCounterSet"));
    // And a comment-only mention must NOT fire, which is the property that makes
    // the real negative mean "no API is called" rather than "the word is absent".
    let commented = REPACKED.replacen(anchor, &format!("{anchor} // MTLCounterSet"), 1);
    assert!(!strip_comments(&commented).contains("MTLCounterSet"));

    // (d) The `mem_flags` member scan is sensitive: a new member must change the
    // set, not be silently absorbed. Injected into code, because the scanner
    // strips comments -- and `mem_flags::mem_device` in a comment is prose.
    let clean = strip_comments(REPACKED);
    assert_eq!(
        mem_flags_members(&clean),
        ["mem_threadgroup".to_string()].into_iter().collect(),
        "the mutation test's baseline must match the real scan"
    );
    let anchor_barrier = "threadgroup_barrier(mem_flags::mem_threadgroup)";
    assert!(
        REPACKED.contains(anchor_barrier),
        "the barrier anchor moved; the mutation would be a silent no-op"
    );
    let mutated = REPACKED.replacen(
        anchor_barrier,
        "threadgroup_barrier(mem_flags::mem_device)",
        1,
    );
    let mutated_members = mem_flags_members(&strip_comments(&mutated));
    assert!(
        mutated_members.contains("mem_device"),
        "the member scanner missed an injected mem_flags::mem_device"
    );
    assert!(
        !mem_flags_members(&strip_comments(REPACKED)).contains("mem_device"),
        "the mutation leaked: the real source now claims a member it does not have"
    );

    // (e) The lane arithmetic is sensitive: perturb KA and the alignment verdicts
    // must move. If `even == 16` held for every KA the claim that `uint2` is
    // illegal for half the lanes would be vacuous.
    let even_for = |ka: u32| {
        let ring = 256 * ka;
        (0..32u32)
            .filter(|lane| {
                let lo = (lane * CODES_PER_LANE + 1) * ka + ring - 16;
                let hi = lo / 32;
                let words = ring / 32;
                let j0 = if hi >= words { hi - words } else { hi };
                j0 % 2 == 0
            })
            .count()
    };
    assert_eq!(even_for(KA), 16, "3 bpw: 16 of 32 lanes are 8-byte aligned");
    // Recorded because it is a stronger statement than "3 bpw is unlucky":
    // **exactly half** the lanes are 8-byte aligned at *every* integer rate
    // 1..=7, and at 8 bpw none are (the window then fits one word and the
    // offsets all go odd). So a `uint2` weight load is never legal for more
    // than half the lanes at any rate this engine supports.
    for ka in 1..=7u32 {
        assert_eq!(even_for(ka), 16, "rate {ka} bpw: half the lanes, exactly");
    }
    assert_eq!(even_for(8), 0, "8 bpw: no lane is 8-byte aligned");
    assert_ne!(even_for(KA), even_for(8), "the alignment verdict must track KA");

    // (f) The instruction accounting is sensitive: the saving must scale with the
// number of activation tiles, not sit at a hardcoded 4.
    // One activation tile = 16 codes and currently costs 4 `half` loads; it
    // costs 2 `half2` loads after. So the saving is exactly half, per tile.
    let act_tiles = |codes: u32| codes / 16;
    let saved = |codes: u32| act_tiles(codes) * (4 - 2);
    assert_eq!(act_tiles(32), 2);
    assert_eq!(saved(32), 4, "_2col's unrolled body: 2 activation tiles, 4 saved");
    assert_eq!(saved(64), 8, "twice the tiles, twice the saving");
    assert_ne!(saved(32), saved(64));
    assert_eq!(saved(32) * 2, saved(64));
}

const LAYERS: u64 = 60;
const SLIDING_LAYERS: u64 = 50;
const FULL_LAYERS: u64 = 10;
/// Asserted only to pin that the split above is the real one -- 50 + 10 = 60.
const _LAYER_SPLIT_CHECK: () = assert!(SLIDING_LAYERS + FULL_LAYERS == LAYERS);
const SLIDING_KV_HEADS: u64 = 16;
const SLIDING_HEAD_DIM: u64 = 256;
const SLIDING_WINDOW: u64 = 1024;
const FULL_KV_HEADS: u64 = 4;
const FULL_HEAD_DIM: u64 = 512;
const INITIAL_FULL_CACHE: u64 = 512;
const FP16: u64 = 2;
/// `key_positions` is `u32` per position (`gpu_cache.rs`), read once per
/// attention pass alongside K and V.
const POS_U32: u64 = 4;

/// Bytes one sliding layer's cache holds at capacity: `stream_len * 2 (K,V) *
/// fp16 + u32 positions`.
fn sliding_resident() -> u64 {
    SLIDING_KV_HEADS * SLIDING_HEAD_DIM * FP16 * 2 + POS_U32
}

fn full_resident() -> u64 {
    FULL_KV_HEADS * FULL_HEAD_DIM * FP16 * 2 + POS_U32
}

/// **881 MB is the allocation, 31.8 MiB is the live set, and only the live set
/// can be said to be evicted.**
#[test]
fn the_881_mb_figure_is_the_allocation_and_3_8_of_it_is_never_written() {
    const _: () = _LAYER_SPLIT_CHECK;
    let sliding = SLIDING_LAYERS * SLIDING_WINDOW * sliding_resident();
    let full = FULL_LAYERS * INITIAL_FULL_CACHE * full_resident();
    let allocated = sliding + full;

    // 880,803,840 B = 880.8 MB decimal = the brief's "881 MB".
    assert!(
        (allocated as f64 / 1.0e6 - 880.8).abs() < 0.5,
        "allocation is {allocated} B = {:.1} MB",
        allocated as f64 / 1.0e6
    );

    // The live set at the 37-token benchmark context: sliding layers are capped
    // at the window, full layers at `next_position`.
    let live = SLIDING_LAYERS * 37 * sliding_resident()
        + FULL_LAYERS * 37 * full_resident();
    // 33,350,320 B = 31.80 MiB, matching `perf_probe_kv_scaling.rs`'s header.
    // Note the convention: `perf_probe_kv_scaling.rs`'s `read_bytes_at` includes
    // the `u32` `positions` per entry (`gpu_cache.rs`), so does this.
    assert_eq!(live, 33_350_320, "31.80 MiB of live KV at 37 tokens");
    // The other convention, excluding `positions`, is the one
    // `perf_probe_bytes_per_token.rs`'s `kv_bytes_per_token` uses. Both are
    // asserted so neither can silently drift into the other.
    let live_no_pos = SLIDING_LAYERS * 37 * (sliding_resident() - POS_U32)
        + FULL_LAYERS * 37 * (full_resident() - POS_U32);
    assert_eq!(live_no_pos, 33_341_440, "without the u32 positions");
    let share = live as f64 / allocated as f64;
    assert!(
        (0.037..0.039).contains(&share),
        "only {:.2}% of the allocated KV is live at 37 tokens", share * 100.0
    );
    // The other 96% was never written. "Evicting" a line that has never been
    // allocated is not an event.
    // Dead:live is ~25:1, so ~96% of the allocation was never written and cannot
    // be "evicted". Evicting a line that was never allocated is not an event.
    assert!(
        allocated > 25 * live,
        "allocated {} vs live {} is only {:.1}x, not the ~26x claimed",
        allocated,
        live,
        allocated as f64 / live as f64
    );
}

/// **Yes, the weight stream evicts the live KV, and it is already evicted before
/// any hint could help — so there is nothing for a hint to recover.**
///
/// The KV's only reuse in this workload is *across tokens*: layer L's position
/// `t-1` is written at step `t-1` and re-read at step `t`, `t+1`, … That reuse
/// survives only if a line written during step `t-1` is still present at step
/// `t`. Between those two steps the engine streams **12.04 GB** of weights.
///
/// Against any plausible AGX L2 that is 380–1500 complete turnovers of the
/// cache between consecutive uses of a KV line. The reuse is therefore already
/// dead, and it is dead because of *volume*, not because of cache policy — which
/// is precisely the class of problem a non-temporal hint cannot address.
///
/// The measured cost of attention (0.842 ms for all 60 layers) is consistent
/// with this and does not need the reuse: each KV byte is read exactly **once**
/// per token, the same 1x property the weights have.
#[test]
fn kv_cross_token_reuse_is_already_destroyed_by_stream_volume_not_by_policy() {
    // L2 turnovers per token. Range, because Apple publishes no L2 size for this
    // part (AGENTS.md §10.3's rule: do not present a guessed constant as a
    // measured one -- state the bracket).
    for l2_mib in [8u64, 16, 32, 48] {
        let turnovers = BYTES_PER_TOKEN as f64 / (l2_mib as f64 * 1024.0 * 1024.0);
        assert!(
            turnovers > 100.0,
            "{l2_mib} MiB L2 -> {turnovers:.0} turnovers/token"
        );
    }
    // Even the most generous plausible L2 cannot hold the live KV at long context,
    // so there is no configuration in which the cross-token reuse survives
    // without also shrinking the weight stream. At 37 tokens it *could* fit a
    // 48 MiB L2 -- and that is exactly why the 37-token number is the wrong
    // place to look for this effect: at that context the reuse would survive
    // if the stream did not bury it.
    let live_37 = 33_350_320u64;
    // Each KV byte is read once per token -- the same 1x the weights have, so
    // there is no intra-token reuse a prefetcher could be missing either. The
    // cross-check is against `perf_probe_bytes_per_token.rs`'s asserted
    // `at_8192 = 1_509_949_440`, which uses the *other* convention (no `u32`
    // positions), so both are computed here from the shapes.
    let live_8192_no_pos = SLIDING_LAYERS * SLIDING_WINDOW * (sliding_resident() - POS_U32)
        + FULL_LAYERS * 8192 * (full_resident() - POS_U32);
    assert_eq!(
        live_8192_no_pos, 1_509_949_440,
        "matches perf_probe_bytes_per_token.rs's asserted at_8192"
    );
    let live_8192 = live_8192_no_pos
        + (SLIDING_LAYERS * SLIDING_WINDOW + FULL_LAYERS * 8192) * POS_U32;
    assert!(live_8192 > 48 * 1024 * 1024, "live KV at 8192 exceeds any plausible L2");
    assert!(
        live_8192 - live_8192_no_pos < 1024 * 1024,
        "the positions u32 add under 1 MiB; they are not what makes KV large"
    );
    assert_eq!(
        SLIDING_LAYERS * SLIDING_WINDOW * (sliding_resident() - POS_U32),
        838_860_800,
        "sliding layers are already capped at the window"
    );

    // The decisive ratio is not KV-vs-L2, it is weights-vs-KV: 12.04 GB of
    // weight stream against 31.8 MiB of live KV means the KV is a rounding error
    // in the pollution argument at any cache level, whatever the policy.
    let weights_over_kv = BYTES_PER_TOKEN as f64 / live_37 as f64;
    assert!(
        weights_over_kv > 300.0,
        "weights are {weights_over_kv:.0}x the live KV"
    );
    let share = live_37 as f64 / (BYTES_PER_TOKEN + live_37) as f64;
    assert!(
        (0.0027..0.0029).contains(&share),
        "live KV is {:.3}% of per-token bytes at 37 tokens", share * 100.0
    );
    // And at 8192 context, where the KV does become large, it is 11.1% of the
    // bytes -- still short of the point where protecting it could matter.
    let share_8192 = live_8192 as f64 / (BYTES_PER_TOKEN + live_8192) as f64;
    assert!(
        (0.110..0.113).contains(&share_8192),
        "live KV is {:.2}% of per-token bytes at 8192", share_8192 * 100.0
    );
}

// ---------------------------------------------------------------------------
// 6. Does the 512 MB arena segment affect cache or TLB behaviour?
// ---------------------------------------------------------------------------

/// **No. The segment size is a *placement* constraint, and by construction there
/// is no segment boundary inside any tensor's read range — so the k-loop sees one
/// flat contiguous array and the cache/TLB behaviour is byte-identical to a
/// single 15.5 GB buffer.**
///
/// `Arena::allocate` (`weights.rs:160-198`) splits at most into several spans,
/// but `WeightStore::tensor_buffer` (`weights.rs:806`) rejects anything that is
/// not **exactly one span at offset 0**:
///
/// ```
/// if tensor.spans.len() != 1 || tensor.spans[0].offset != 0 {
///     return Err(WeightError::InvalidSpan { .. });
/// }
/// ```
///
/// So a decode-time packed tensor never straddles 512 MB. `DEFAULT_SEGMENT_BYTES`
/// is `512 * 1024 * 1024` (`metal.rs:33`) purely because that is a safe
/// `newBufferWithLength` size.
///
/// `lm_head` is the interesting case and it does not go through the arena at
/// all: at 1.057 GB packed it exceeds one segment, so `weights.rs:585-587`
/// `offload_head_trellis` puts it on the **zero-copy mmap** path with empty spans.
/// Which is also why `lm_head` has always measured at the ceiling (253-257
/// GB/s) while the arena-resident layers do not — see
/// `spec/decode-ceiling/findings/M01-tlb-page-size.md:130-135`.
#[test]
fn segment_size_is_a_placement_constraint_not_a_cache_or_tlb_factor() {
    // The constant itself.
    assert!(
        METAL_RS.contains("pub const DEFAULT_SEGMENT_BYTES: u64 = 512 * 1024 * 1024;"),
        "DEFAULT_SEGMENT_BYTES moved; this test's arithmetic needs updating"
    );

    // The single-span invariant, quoted from the source rather than assumed.
    let at = line_of(WEIGHTS_RS, "if tensor.spans.len() != 1 || tensor.spans[0].offset != 0 {");
    assert!(at > 0, "tensor_buffer must still reject multi-span tensors");
    let neighbourhood: String = WEIGHTS_RS.lines().skip(at - 3).take(6).collect::<Vec<_>>().join("\n");
    assert!(neighbourhood.contains("InvalidSpan"));

    // lm_head is larger than one segment, so it cannot be arena-resident.
    let lm_head_bytes: u64 = 262_144 * 5_376 * 6 / 8;
    assert_eq!(lm_head_bytes, 1_056_964_608, "262144 x 5376 at 6 bpw");
    let segment = 512 * 1024 * 1024;
    assert!(
        lm_head_bytes > segment,
        "lm_head at {lm_head_bytes} B exceeds one {segment} B segment, which is \
         why it takes the zero-copy path instead"
    );
    assert!(
        WEIGHTS_RS.contains("offload_head_trellis"),
        "the head's dedicated zero-copy flag must still exist"
    );

    // Consequence for the k-loop: with no straddling tensor, the per-tile stride
    // is uniform across the whole read range, so there is no boundary at which
    // the access pattern changes. One flat array, 12.04 GB, 16 KB pages.
    let pages_per_token = BYTES_PER_TOKEN / 16384;
    // 734,832 from the exact byte count. M01 quotes 734,864 because it used the
    // rounded 12.04e9; the two are the same figure to four digits and neither
    // conclusion depends on the difference.
    assert_eq!(pages_per_token, 734_832);
    let m01_rounded = 12_040_000_000u64 / 16384;
    assert_eq!(m01_rounded, 734_863);
    assert!((pages_per_token as i64 - m01_rounded as i64).abs() < 64);
    // And M01 already measured the TLB cost at ~0.3% and closed it as "cannot
    // be made into a limiter in either direction"
    // (`spec/decode-ceiling/findings/M01-tlb-page-size.md:144,246`).
}

/// The GEMV's own concurrency, which is the one place a cache-adjacent
/// resource *is* close to a limit: `linear.rs:2142` launches
/// `min(out_features / 32, gemv_max_groups())` threadgroups, and `gemv_max_groups`
/// defaults to `u64::MAX` (`linear.rs:94-102`), so for the 21504-wide MLP
/// projections that is **672 threadgroups each sweeping two 129 KB column
/// streams**. Worth recording, and worth noting nobody has measured whether a
/// smaller cap helps — `KAPLUMBA_GEMV_MAX_GROUPS` exists and is unused in the
/// decode path's tuning.
#[test]
fn the_gemv_launches_1344_concurrent_weight_streams_and_no_one_has_measured_a_cap() {
    // MLP projections: intermediate_size 21504.
    let pairs = 21_504 / 32;
    assert_eq!(pairs, 672);
    let streams = pairs * 2; // `_2col` holds two column bases.
    assert_eq!(streams, 1344);

    // Each stream is one 96 B tile stride advancing 2 tiles per iteration, i.e.
    // 192 B contiguous per step over a 129 KB column: sequential per stream.
    let tiles_k = 21_504 / 16;
    assert_eq!(tiles_k, 1_344);
    let column_bytes = tiles_k * TILE_BYTES as u64;
    assert_eq!(column_bytes, 129_024);

    // The cap is real and settable, and nothing in the production decode path
    // sets it. Recorded as an open question, not a proposal -- it changes the
    // launch geometry and so must be A/B'd, and this file cannot measure it.
    assert!(LINEAR_RS.contains("KAPLUMBA_GEMV_MAX_GROUPS"));
    assert!(LINEAR_RS.contains(".unwrap_or(u64::MAX)"));
}