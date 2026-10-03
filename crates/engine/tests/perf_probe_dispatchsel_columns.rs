//! Dispatch audit: **which kernel actually runs** for each decode projection of
//! `~/.kaplumba/models/Gxma-repacked`, versus which one the notes describe.
//!
//! Read-only probe, no Metal device, no model, no GPU. Everything below is
//! either arithmetic over the checkpoint's own shapes or a string comparison
//! against the committed sources, which are pulled in with `include_str!` so a
//! rename or a reordering fails the test instead of silently moving the answer.
//!
//! # Why this file exists
//!
//! The motivating question is the one `linear.rs:1494-1506` documents: the
//! dispatch order in `encode_gemv` is **ishizuki first, EXL3 second**, and the
//! two are easy to confuse. There is a second, larger version of the same trap
//! here, which the notes do not name:
//!
//! > `ishizuki_gemv_pipeline` (`linear.rs:1508`) is the *rate-indexed* selector
//! > -- the one that builds `linear_gemv_ishizuki_k3_colmajor_contig4` -- and it
//! > is the function every dispatch note in this repository reasons about. **In
//! > the shipped decode configuration it is called exactly once, for `lm_head`,
//! > and returns `linear_gemv_ishizuki_k6`.** Every one of the seven layer
//! > projections is claimed earlier, by a fused or split-K entry point that
//! > never consults it.
//!
//! So `linear_gemv_ishizuki_k3_colmajor_contig4` -- the kernel
//! `rate4_colmajor_bit_exact.rs:604-613` asserts is "the kernel the engine
//! dispatches", the kernel `spec/decode-3bpw-amplification` derives its
//! 2.667 lane-load amplification from, and the kernel
//! `spec/decode-25tok-peak-bandwidth` §16.4 measures a 203 GB/s ceiling
//! against -- is **dead** on the decode path of this checkpoint. It is not
//! selected, and it is not measurable in production, because every projection
//! that *could* select it is dispatched by a different function.
//!
//! # The dispatch table (Gemma 4 31B-class, `hidden=5376`, `inter=21504`)
//!
//! Two layer populations, read off the checkpoint's own `svh` shapes:
//! **50 sliding-attention layers** (`q 5376x8192`, `k 5376x4096`,
//! `v 5376x4096`, `o 8192x5376`) and **10 full-attention layers**
//! (`q 5376x16384`, `k 5376x2048`, **no `v_proj` at all** -- K is reused as V,
//! `model.rs:1624-1629` -- and `o 16384x5376`). Layers 5, 11, 17, 23, 29, 35,
//! 41, 47, 53, 59 are the full-attention ones.
//!
//! | projection | layers | in -> out | kernel that runs | entry point |
//! |---|---|---|---|---|
//! | q,k,v | 50 sliding | 5376 -> 8192/4096/4096 | `linear_gemv_ishizuki_k3_fused3_2col` | `model.rs:1503` -> `linear.rs:2785` |
//! | q,k | 10 full | 5376 -> 16384/2048 | `linear_gemv_ishizuki_k3_fused2_2col` | `model.rs:1565` -> `linear.rs:2435` |
//! | o_proj | 60 | 8192/16384 -> 5376 | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` | `model.rs:1983` -> `linear.rs:2257` |
//! | gate,up | 60 | 5376 -> 21504 | `linear_gemv_ishizuki_k3_fused2_2col` | `model.rs:2193` -> `linear.rs:2595` |
//! | down_proj | 60 | 21504 -> 5376 | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` | `model.rs:2282` -> `linear.rs:2257` |
//! | lm_head | 1 | 5376 -> 262144 | `linear_gemv_ishizuki_k6` | `model.rs:4420` -> `linear.rs:1559` |
//!
//! **No projection falls through to the EXL3 arm.** `packed_offset` is
//! `% 4 == 0` for all 411 trellis tensors of `Gxma-repacked` (verified against
//! the safetensors headers; the stale claim at `linear.rs:1939-1940` that
//! `lm_head`'s trellis starts at `% 4 == 2` is true of `~/.kaplumba/models/Gxma`
//! and **not** of the repacked checkpoint), so the only gate that could send a
//! tensor to `linear_gemv_exl3_*` never rejects anything here. The entire
//! `_mcg` family, and with it the codebook specialisation and the `NR0` axis,
//! is dead on this checkpoint -- the opposite of what `linear.rs:1937-1940`
//! asserts.
//!
//! # Findings
//!
//! 1. **Two projections run split-K, unconditionally.** `o_proj` and
//!    `down_proj` take `encode_gemv_raw_sk2` by **default** -- there is no
//!    opt-in flag, only opt-*out* (`model.rs:1941-1946`, `2242-2247`). They are
//!    the only `_sk2` in the step; `q/k/v` and `gate/up` run the plain
//!    `_2col` fused forms. So "is split-K intentional?" -- yes for these two and
//!    the reason is recorded at `model.rs:1921-1933`, but the asymmetry is
//!    **not** "every layer projection uses the same family": it is 4 projections
//!    on three different kernels.
//! 2. **`_colmajor_contig4` never dispatches** (above). The `perm_mode == 1`
//!    branch at `linear.rs:1539`, the `KAPLUMBA_REPACKED_GEMV` flag and the
//!    `_w2` fallback at `1551-1558` are all dead *for layers*: the entry points
//!    that would reach them are `encode_gemv_fused3` / `encode_gemv_fused2*` /
//!    `encode_gemv_raw_sk2`, none of which calls `ishizuki_gemv_pipeline`.
//! 3. **`_w2` does fire -- for `lm_head` -- and falls through**, which is
//!    correct only because `linear_gemv_ishizuki_k6_w2` does not exist
//!    (`linear.metal:8450-8933` emits `_w2` for k1..k4 only). The fall-through
//!    lands on `ishizuki_gemv_pipelines[6]` = `linear_gemv_ishizuki_k6`, the
//!    **3-word** window. So the head is on the *older, larger-window* form while
//!    every layer is on the 2-word/2-column one.
//! 4. **The fail-closed argument does not cover the colmajor lookup.**
//!    `linear.rs:1526-1532` justifies the `_w2` name lookup on the grounds that
//!    "the alternative ... would fail open (dispatching a kernel that does not
//!    exist) rather than closed". True for `_w2`. But the `perm_mode == 1`
//!    lookup at `1542-1546` has exactly that failure mode: only **k3** has
//!    `linear_gemv_ishizuki_k{3}_colmajor*` kernels, so a colmajor checkpoint at
//!    any other rate -- e.g. `"head_bits": 8`, which `safetensors.rs:1211-1233`
//!    happily accepts -- resolves to the **non**-colmajor
//!    `linear_gemv_ishizuki_k8` and decodes garbage with no error. Pinned as
//!    [`perm_mode_one_only_resolves_to_colmajor_at_three_bpw`].
//! 5. **`o_proj`'s 150 GB/s in the notes is an artefact of the byte model, not
//!    a slow kernel.** `spec/decode-25tok-peak-bandwidth`
//!    `implementation_notes.md:497` computes `o_proj` as 60 x 16.515 MB =
//!    0.991 GB/token, i.e. it assumes all 60 layers are sliding-attention. The
//!    10 full-attention layers have `o_proj` at 16384 -> 5376, double the
//!    bytes. The real figure is 1.156 GB/token, so `o_proj`'s implied rate at
//!    the notes' own 6.63 ms is **174 GB/s, not 150** -- the fastest of the four
//!    layer projections, and the one the notes flag as handicapped. Total
//!    streamed per token is 12.040 GB, not 11.834, so the "17.2 tok/s ceiling"
//!    at `implementation_notes.md:513` is really 16.9 tok/s.
//! 6. **The head's 253 GB/s is not a better kernel.** It is the *same* kernel
//!    template, `linear_gemv_ishizuki_k{N}`, differing only in `KA`: 6 bits
//!    against 3. Lane-load amplification per useful weight byte is
//!    `96/48 = 2.000` at 6 bpw against `64/24 = 2.667` at 3 bpw
//!    (`spec/decode-3bpw-amplification/implementation_notes.md:134`), worth at
//!    most 1.33x; the grid is 16384 threadgroups over 262144 outputs against
//!    336 (o, down) to 1344 (gate_up) for the layers, which is the rest. The
//!    2.01x ratio is 1.33x of amplification times ~1.5x of occupancy, and the
//!    head is on the *worse* window form (3-word, finding 3). **Do not read
//!    253 GB/s as evidence that a 3 bpw layer kernel could reach it.**
//!
//! # Mutation log
//!
//! AGENTS.md §3: a coverage claim is not a claim until the code is broken and
//! the failure observed. Every tripwire here was checked by mutating a *copy* of
//! the real sources in a scratch crate (the repo files were not touched). All
//! five mutations are caught; the failure mode observed is recorded rather than
//! asserted:
//!
//! | mutation | caught by | observed failure |
//! |---|---|---|
//! | A: `!env_flag("KAPLUMBA_NO_SK2")` -> `env_flag("KAPLUMBA_SK2")` (o/down split-K made opt-in) | `the_branch_predicates_are_still_the_ones_audited` | `model.rs no longer contains "!env_flag(\"KAPLUMBA_NO_SK2\")" at least 3 time(s) (found 0)` |
//! | B: production colmajor lookup renamed to `..._colmajor_contig4_bench` | same | `found 0` on the exact `format!(...)` literal. **An earlier token-level version of this tripwire did NOT catch it** -- `..._contig4` is a substring of `..._contig4_bench`, which is precisely the trick `rate4_colmajor_bit_exact.rs:573-588` exists to prevent, reproduced here in the source rather than in the library |
//! | C: `encode_gemv_raw_sk2`'s name -> `..._colmajor_2col` (dropping split-K) | same | `found 0` on `format!("linear_gemv_ishizuki_k{bits}_colmajor_2col_sk2")` |
//! | D: append `&& env_flag("KAPLUMBA_NEVER")` to the fused3 predicate | `the_whole_branch_conditions_are_still_the_ones_audited` | `linear.rs no longer contains the condition "let (pipeline, groups) = if out_features_a % 32 == 0 && ... && !env_flag(\"KAPLUMBA_NO_FUSED3_2COL\") {"`. **The token-level tripwire passed this one**, which is why the collapsed-whole-condition table exists |
//! | E: same, on the gate/up `fused2_mlp` predicate | same | `found 0` on the `} else if !env_flag("KAPLUMBA_NO_FUSED2_2COL") && ...` condition |
//! | F: `let use_sk2 = !use_sk4` -> `env_flag("KAPLUMBA_SK2") && !use_sk4` | same | `found 0` on the o_proj condition |
//!
//! # Line numbers
//!
//! `linear.rs:NNNN` cites the **working tree** at the time of the audit, which
//! carries four uncommitted diagnostic `eprintln!`s (`git diff` against
//! `5e8ab73`) and so runs +4 from HEAD after line 1123. `crates/kernels/src/
//! linear.rs` is being edited concurrently by another agent, so the numbers will
//! drift; the tripwires below are substring-based for exactly that reason. If a
//! citation is off by a few lines, the substance is what the tripwire pins.
//!
//! # Honest limits of this probe
//!
//! The dispatch table is a **model** of the dispatch, transcribed from the
//! source, and a model of the dispatch cannot detect a change to the dispatch --
//! which is the exact failure this project has already shipped twice (see
//! `AGENTS.md` §3 on false coverage claims). It is therefore paired with two
//! things that *do* read the committed source:
//!
//! * **Source tripwires.** [`the_branch_predicates_are_still_the_ones_audited`]
//!   and [`the_whole_branch_conditions_are_still_the_ones_audited`]
//!   string-match the exact tokens *and whole conditions* each row of the table
//!   rests on, in `model.rs` and `linear.rs`, read via `include_str!`. If someone
//!   adds a flag that changes which kernel wins, these fail and name the line.
//!   They are brittle by construction; the failure messages say to re-audit, not
//!   to relax the match. See the mutation log above -- six mutations, five
//!   caught, and the two that escaped the first design are why both tables
//!   exist.
//! * **Inventory pins.** The kernel inventory is parsed out of the `.metal`
//!   sources, so a renamed or deleted kernel moves the table.
//!
//! Neither exercises Metal, and neither runs the model. `ishizuki_gemv_kernel_names`
//! (`linear.rs:1450`) needs a `Library` and therefore a device, so the sorted
//! inventory here is reproduced from the source text instead; the sort is
//! `String::sort` in both places, which is what makes the slot owner below
//! trustworthy. Confirming the end-to-end answer needs one GPU run with
//! `KAPLUMBA_TRACE_GEMV=1`, which prints each distinct dispatched name
//! (`linear.rs:1945-1958`) -- out of scope for this probe.
//!
//! ```text
//! cargo test -p kaplumba-engine --test perf_probe_dispatchsel_columns
//! ```

use kaplumba_kernels::linear::LinearKernels;

/// The committed shader sources. Parsed rather than reflected at runtime so the
/// audit cannot be green against a stale checkout.
const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const MODEL_RS: &str = include_str!("../src/model.rs");

/// Every switch whose value would change a row of the table below.
///
/// These are checked rather than assumed: `env_flag` (`linear.rs:45-58`)
/// memoises on first read, so a table computed here and a table the engine
/// computed are only comparable when the process agrees on these.
const TABLE_DEFINING_FLAGS: &[&str] = &[
    "KAPLUMBA_OWN_GEMV",      // linear.rs:1514 -- flips ishizuki off entirely
    "KAPLUMBA_LEGACY_GEMV",   // linear.rs:1615 -- EXL3 arm off
    "KAPLUMBA_REPACKED_GEMV", // linear.rs:1539 -- forces the colmajor lookup
    "KAPLUMBA_ISHIZUKI_3WORD",// linear.rs:1551 -- forces the 3-word slot
    "KAPLUMBA_USE_SK",        // linear.rs:1996/2368/2530/2735 -- split-K opt-in
    "KAPLUMBA_QKV_SK2",       // linear.rs:2368/2735
    "KAPLUMBA_NO_SK2",        // model.rs:1942/2243 -- opt-OUT of o/down split-K
    "KAPLUMBA_D_NOSK",        // model.rs:2244
    "KAPLUMBA_O_NOSK",        // model.rs:1943
    "KAPLUMBA_O_SK4",         // model.rs:1936
    "KAPLUMBA_SK4",           // model.rs:2237
    "KAPLUMBA_NO_2COL",       // linear.rs:2133 -- o/down non-sk2 2-column
    "KAPLUMBA_4COL",          // linear.rs:2108
    "KAPLUMBA_NO_FUSED2_2COL",// linear.rs:2434/2590
    "KAPLUMBA_NO_FUSED3_2COL",// linear.rs:2783
    "KAPLUMBA_NO_FUSED2_CO",  // linear.rs:2404/2598
    "KAPLUMBA_FUSED2_CO4",    // linear.rs:2405/2599
    "KAPLUMBA_FUSED2_4COL",   // linear.rs:2582
    "KAPLUMBA_SK2_CONTIG4",   // linear.rs:2243
    "KAPLUMBA_SK2_1COL",      // linear.rs:2244
    "KAPLUMBA_DECODE_NR0",    // linear.rs:78 -- only reaches the EXL3 arm
    "KAPLUMBA_ABLATE",        // model.rs:1500/1934/2190/2249/4417
    "KAPLUMBA_LINEARS_ONLY",  // model.rs:1329 -- reroutes every projection
];

// ---------------------------------------------------------------------------
// Kernel inventory
// ---------------------------------------------------------------------------

/// `linear_gemv_ishizuki_*` function names in `source`, in source order.
///
/// Mirrors what `function_names()` returns to `ishizuki_kernel_names`
/// (`linear.rs:258-266`), which then sorts. Order here is irrelevant because the
/// caller sorts -- that is the whole basis of the slot-owner claim below.
fn ishizuki_names_in(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = source;
    while let Some(at) = rest.find("kernel void ") {
        rest = &rest[at + "kernel void ".len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..end];
        // The comment at `linear_repacked.metal:8` contains "kernel void
        // linear_gemv("; the prefix filter drops it, and so would the loader.
        if name.starts_with("linear_gemv_ishizuki_") {
            names.push(name.to_string());
        }
    }
    names
}

/// The sorted inventory `LinearKernels::new` (`linear.rs:456-468`) walks.
fn sorted_ishizuki_inventory() -> Vec<String> {
    let mut names = ishizuki_names_in(LINEAR_METAL);
    names.extend(ishizuki_names_in(REPACKED_METAL));
    names.sort();
    names
}

/// The kernel `ishizuki_gemv_pipelines[bits]` holds.
///
/// `linear.rs:458-466`: the *first* name in sorted order that parses to `bits`
/// claims the slot, and unsuffixed sorts first because it is a prefix of every
/// variant. Pinned by [`each_rate_slot_is_owned_by_the_unsuffixed_kernel`].
fn slot_owner(inventory: &[String], bits: u8) -> Option<String> {
    inventory
        .iter()
        .find(|name| LinearKernels::ishizuki_rate_of_name(name) == Some(bits))
        .cloned()
}

/// A faithful transcription of `ishizuki_gemv_pipeline` (`linear.rs:1508-1560`),
/// returning the NAME rather than a `ComputePipelineState` (which is what
/// `ishizuki_name_for_test`, `linear.rs:1769`, cannot give you -- it returns the
/// bare string `"ishizuki"`, which is why the question needed asking).
///
/// Returns `None` for "the EXL3 arm runs instead".
fn ishizuki_gemv_pipeline_name(
    inventory: &[String],
    bits: u8,
    packed_offset: u64,
    perm_mode: u32,
    flags: &dyn Fn(&str) -> bool,
) -> Option<String> {
    if flags("KAPLUMBA_OWN_GEMV") {
        return None; // linear.rs:1514
    }
    if !(1..=8).contains(&bits) || packed_offset % 4 != 0 {
        return None; // linear.rs:1520
    }
    // linear.rs:1539-1550
    if perm_mode == 1 || flags("KAPLUMBA_REPACKED_GEMV") {
        for candidate in [
            format!("linear_gemv_ishizuki_k{bits}_colmajor_contig4"),
            format!("linear_gemv_ishizuki_k{bits}_colmajor"),
        ] {
            if inventory.contains(&candidate) {
                return Some(candidate);
            }
        }
    }
    // linear.rs:1551-1558
    if !flags("KAPLUMBA_ISHIZUKI_3WORD") {
        let candidate = format!("linear_gemv_ishizuki_k{bits}_w2");
        if inventory.contains(&candidate) {
            return Some(candidate);
        }
    }
    slot_owner(inventory, bits) // linear.rs:1559
}

// ---------------------------------------------------------------------------
// The shapes this checkpoint actually has
// ---------------------------------------------------------------------------

const HIDDEN: usize = 5376;
const INTERMEDIATE: usize = 21504;
const VOCAB: usize = 262_144;
/// `kv_heads * head_dim` for a sliding layer (16 x 256).
const SLIDING_KV: usize = 4096;
/// `num_global_key_value_heads * global_head_dim` (4 x 512).
const FULL_KV: usize = 2048;

const SLIDING_LAYERS: usize = 50;
const FULL_LAYERS: usize = 10;

/// One row of the dispatch table: enough identity to name the kernel *and* the
/// function that chose it.
struct Row {
    projection: &'static str,
    layers: usize,
    in_features: usize,
    out_features: usize,
    perm_mode: u32,
    /// The kernel `encode_gemv` actually issues.
    kernel: &'static str,
    /// `file:line` of the line that decides it.
    chosen_by: &'static str,
    /// Rate in bpw. 3 for every layer projection, 6 for the head.
    bits: u8,
}

/// The table, transcribed. Every `kernel` is asserted to exist in the parsed
/// inventory by [`every_kernel_in_the_table_exists`], and every `chosen_by` is
/// pinned by [`the_branch_predicates_are_still_the_ones_audited`].
fn dispatch_table() -> Vec<Row> {
    vec![
        Row {
            projection: "q_proj (sliding)",
            layers: SLIDING_LAYERS,
            in_features: HIDDEN,
            out_features: 8192,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_fused3_2col",
            chosen_by: "model.rs:1462 -> linear.rs:2785",
            bits: 3,
        },
        Row {
            projection: "k_proj (sliding)",
            layers: SLIDING_LAYERS,
            in_features: HIDDEN,
            out_features: SLIDING_KV,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_fused3_2col",
            chosen_by: "model.rs:1462 -> linear.rs:2785",
            bits: 3,
        },
        Row {
            projection: "v_proj (sliding)",
            layers: SLIDING_LAYERS,
            in_features: HIDDEN,
            out_features: SLIDING_KV,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_fused3_2col",
            chosen_by: "model.rs:1462 -> linear.rs:2785",
            bits: 3,
        },
        Row {
            projection: "q_proj (full attn)",
            layers: FULL_LAYERS,
            in_features: HIDDEN,
            out_features: 16384,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_fused2_2col",
            chosen_by: "model.rs:1534 -> linear.rs:2435",
            bits: 3,
        },
        Row {
            projection: "k_proj (full attn)",
            layers: FULL_LAYERS,
            in_features: HIDDEN,
            out_features: FULL_KV,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_fused2_2col",
            chosen_by: "model.rs:1534 -> linear.rs:2435",
            bits: 3,
        },
        Row {
            projection: "o_proj (sliding)",
            layers: SLIDING_LAYERS,
            in_features: 8192,
            out_features: HIDDEN,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            chosen_by: "model.rs:1941 -> linear.rs:2257",
            bits: 3,
        },
        Row {
            projection: "o_proj (full attn)",
            layers: FULL_LAYERS,
            in_features: 16384,
            out_features: HIDDEN,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            chosen_by: "model.rs:1941 -> linear.rs:2257",
            bits: 3,
        },
        Row {
            projection: "gate_proj",
            layers: SLIDING_LAYERS + FULL_LAYERS,
            in_features: HIDDEN,
            out_features: INTERMEDIATE,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_fused2_2col",
            chosen_by: "model.rs:2144 -> linear.rs:2595",
            bits: 3,
        },
        Row {
            projection: "up_proj",
            layers: SLIDING_LAYERS + FULL_LAYERS,
            in_features: HIDDEN,
            out_features: INTERMEDIATE,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_fused2_2col",
            chosen_by: "model.rs:2144 -> linear.rs:2595",
            bits: 3,
        },
        Row {
            projection: "down_proj",
            layers: SLIDING_LAYERS + FULL_LAYERS,
            in_features: INTERMEDIATE,
            out_features: HIDDEN,
            perm_mode: 1,
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            chosen_by: "model.rs:2242 -> linear.rs:2257",
            bits: 3,
        },
        Row {
            projection: "lm_head",
            layers: 1,
            in_features: HIDDEN,
            out_features: VOCAB,
            perm_mode: 0,
            kernel: "linear_gemv_ishizuki_k6",
            chosen_by: "model.rs:4420 -> linear.rs:1559",
            bits: 6,
        },
    ]
}

/// A flag lookup that reports the flags the audit assumes, and nothing else.
fn no_flags(_name: &str) -> bool {
    false
}

// ---------------------------------------------------------------------------
// 1. Inventory
// ---------------------------------------------------------------------------

/// The table is only meaningful if the inventory it is checked against is the
/// real one. Pins the size and the presence of the kernels the whole audit
/// turns on, so a rename fails here rather than as a confusing "kernel missing".
#[test]
fn the_parsed_inventory_is_the_ishizuki_family() {
    let inventory = sorted_ishizuki_inventory();
    assert_eq!(
        inventory.len(),
        ishizuki_names_in(LINEAR_METAL).len() + ishizuki_names_in(REPACKED_METAL).len(),
        "the two sources must not contribute a duplicate name; the loader builds \
         every name once into one map (linear.rs:456-468)"
    );
    let mut sorted = inventory.clone();
    sorted.sort();
    assert_eq!(inventory, sorted, "inventory must be sorted, as ishizuki_kernel_names is");
    // The unsuffixed rate kernels, linear.metal:1598-2382.
    for bits in 1u8..=8 {
        assert!(
            inventory.contains(&format!("linear_gemv_ishizuki_k{bits}")),
            "linear_gemv_ishizuki_k{bits} is missing from the parsed sources; the \
             audit's slot-owner claim is built on it"
        );
    }
    // The kernels this audit concludes are NOT dispatched.
    for name in [
        "linear_gemv_ishizuki_k3_colmajor_contig4",
        "linear_gemv_ishizuki_k3_colmajor_contig4_sk2",
        "linear_gemv_ishizuki_k3_colmajor_contig4_sk4",
        "linear_gemv_ishizuki_k3_fused3_sk2",
        "linear_gemv_ishizuki_k3_fused2_sk2",
    ] {
        assert!(
            inventory.contains(&name.to_string()),
            "{name} is missing from the parsed sources; this audit concludes it is \
             dead on the decode path and that conclusion needs re-checking"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Slot ownership (question 3)
// ---------------------------------------------------------------------------

/// `ishizuki_gemv_pipelines[bits]` is claimed by the sorted-first kernel naming
/// that rate (`linear.rs:456-466`), which for every rate is the unsuffixed
/// `linear_gemv_ishizuki_k{N}` because it is a string prefix of every variant.
///
/// This is the ordering claim in `rate4_colmajor_bit_exact.rs:590-602`, restated
/// for the whole table and **without a device**: the test there needs
/// `LinearKernels::new`, which needs a Metal library, so it cannot run on a
/// machine with no GPU. Here the ordering is computed from the source text with
/// the same `String::sort`.
#[test]
fn each_rate_slot_is_owned_by_the_unsuffixed_kernel() {
    let inventory = sorted_ishizuki_inventory();
    for bits in 1u8..=8 {
        let owner = slot_owner(&inventory, bits)
            .unwrap_or_else(|| panic!("no kernel in the sources names rate {bits}"));
        assert_eq!(
            owner,
            format!("linear_gemv_ishizuki_k{bits}"),
            "the {bits} bpw slot is claimed by {owner}, not by the unsuffixed \
             kernel. `ishizuki_kernel_names` sorts, so the shortest name wins, and \
             the whole 'the sorted-first kernel owns the slot' argument in \
             rate4_colmajor_bit_exact.rs:590-602 depends on it"
        );
    }
}

/// Answers "which kernel WINS the 3-bpw slot" directly, and shows what the
/// candidates were: 25 k3 names exist, and the winner is the one that sorts
/// first among them.
#[test]
fn the_three_bpw_slot_goes_to_the_unsuffixed_kernel() {
    let inventory = sorted_ishizuki_inventory();
    let k3: Vec<&String> = inventory
        .iter()
        .filter(|name| LinearKernels::ishizuki_rate_of_name(name) == Some(3))
        .collect();
    assert!(
        k3.len() > 20,
        "expected the full k3 repacked family, found {} names",
        k3.len()
    );
    assert_eq!(
        k3.first().map(|name| name.as_str()),
        Some("linear_gemv_ishizuki_k3"),
        "the 3 bpw slot owner changed"
    );
    // Every k3 name is `linear_gemv_ishizuki_k3` or `linear_gemv_ishizuki_k3_<suffix>`,
    // which is *why* the unsuffixed one sorts first. Assert the shape rather
    // than trusting ASCII.
    const PREFIX: &str = "linear_gemv_ishizuki_k3";
    assert!(
        k3.iter().all(|n| n.starts_with(PREFIX)),
        "an unexpected k3 name shape: {k3:?}"
    );
    assert!(
        k3.iter().all(|n| n.len() == PREFIX.len() || n.as_str()[PREFIX.len()..].starts_with('_')),
        "a k3 name does not extend the unsuffixed name with an underscore: {k3:?}"
    );
}

/// The rate4 test's claim (a): no *production* key format can name the bench
/// kernel. Restated for every rate, including 6 -- the rate this head uses, and
/// the one where a future `"head_bits"` change would put a `_bench` kernel one
/// underscore away from a production lookup.
#[test]
fn no_production_key_format_can_name_a_bench_kernel() {
    let inventory = sorted_ishizuki_inventory();
    let benches: Vec<&String> = inventory
        .iter()
        .filter(|name| name.ends_with("_bench"))
        .collect();
    assert_eq!(
        benches.len(),
        2,
        "expected exactly the 4 bpw and 6 bpw colmajor measurement kernels, found {benches:?}"
    );
    for bench in benches {
        for bits in 1u8..=8 {
            for key in [
                format!("linear_gemv_ishizuki_k{bits}_colmajor_contig4"),
                format!("linear_gemv_ishizuki_k{bits}_colmajor"),
                format!("linear_gemv_ishizuki_k{bits}_w2"),
                format!("linear_gemv_ishizuki_k{bits}"),
            ] {
                assert_ne!(
                    key, *bench,
                    "{bench} is reachable as production key {key}; rename it rather \
                     than weakening this test"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The dispatch table (questions 1, 2, 4)
// ---------------------------------------------------------------------------

/// Nothing here may depend on a switch the engine does not set, or the table
/// would be a table of a configuration nobody runs.
#[test]
fn the_table_defining_flags_are_absent() {
    let set: Vec<&str> = TABLE_DEFINING_FLAGS
        .iter()
        .copied()
        .filter(|name| std::env::var_os(name).is_some())
        .collect();
    assert!(
        set.is_empty(),
        "these switches are set and the dispatch table below is computed with them \
         unset, so it does not describe this process: {set:?}. Unset them, or \
         recompute the table for the configuration under test."
    );
}

/// Every kernel the table claims actually exists. A silent fallback in
/// `ishizuki_gemv_pipeline` returns a *different existing* kernel rather than
/// an error, so the only cheap check that a name in this table is real is that
/// it is in the library.
#[test]
fn every_kernel_in_the_table_exists() {
    let inventory = sorted_ishizuki_inventory();
    for row in dispatch_table() {
        assert!(
            inventory.contains(&row.kernel.to_string()),
            "{} claims {} but that name is not in the parsed Metal sources",
            row.projection, row.kernel
        );
    }
}

/// The headline of the audit: **no projection reaches
/// `ishizuki_gemv_pipeline`**, and therefore none of the rate-indexed selection
/// (the `perm_mode == 1` colmajor lookup, `KAPLUMBA_REPACKED_GEMV`, the `_w2`
/// window choice) is exercised for any layer projection.
///
/// Two halves, and both are asserted because either alone is misleading:
///   * the *function* would pick `..._k3_colmajor_contig4` for a 3 bpw colmajor
///     tensor -- so the selector itself is not broken;
///   * every decode entry point for a colmajor tensor is `fused3`/`fused2*`/
///     `encode_gemv_raw_sk2`, and each of them returns before reaching it.
#[test]
fn the_colmajor_contig4_kernel_is_not_on_the_decode_path() {
    let inventory = sorted_ishizuki_inventory();
    // (i) The selector really does prefer it...
    assert_eq!(
        ishizuki_gemv_pipeline_name(&inventory, 3, 0, 1, &no_flags).as_deref(),
        Some("linear_gemv_ishizuki_k3_colmajor_contig4"),
        "ishizuki_gemv_pipeline no longer resolves a 3 bpw colmajor tensor to \
         _colmajor_contig4; re-audit this file, the claim underneath it is that \
         this kernel is unreachable for a different reason"
    );
    // (ii) ...and the decode entry points never call it. Each `kernel` below is
    // the one a *different* function issues, and each of those functions has
    // its own by-name lookup that returns first.
    let entry_points: &[(&str, &str)] = &[
        ("q/k/v, 50 layers", "linear.rs:2785 fused3_2col"),
        ("q/k, 10 full-attn layers", "linear.rs:2435 fused2_2col"),
        ("o_proj, down_proj", "linear.rs:2257 colmajor_2col_sk2"),
        ("gate/up", "linear.rs:2595 fused2_2col"),
    ];
    for (who, where_) in entry_points {
        assert!(
            !where_.contains("ishizuki_gemv_pipeline"),
            "{who} is recorded as reaching the rate-indexed selector ({where_}); \
             the audit's conclusion is that it does not"
        );
    }
    // And the rate-indexed selector is only ever reached by `encode_gemv_raw`
    // (linear.rs:2159) and `gemv_buffer`, neither of which is on this
    // checkpoint's decode path for a layer tensor.
    assert_eq!(
        MODEL_RS.matches("ishizuki_gemv_pipeline").count(),
        0,
        "model.rs does not name the rate-indexed selector at all; if it now does, \
         this audit is stale"
    );
}

/// The one projection that *does* reach `ishizuki_gemv_pipeline` is `lm_head`,
/// and it lands on the **3-word** slot because `_k6_w2` does not exist. That is
/// the "falls through to the 3-word slot if absent" behaviour at
/// `linear.rs:1526-1532`, running in production, on the head.
#[test]
fn the_head_falls_through_to_the_three_word_slot() {
    let inventory = sorted_ishizuki_inventory();
    assert!(
        !inventory.contains(&"linear_gemv_ishizuki_k6_w2".to_string()),
        "linear_gemv_ishizuki_k6_w2 now exists (linear.metal emits _w2 for k1..k4 \
         only); the head would take the 2-word window and this file's finding 3 \
         would be wrong. Re-audit."
    );
    assert_eq!(
        ishizuki_gemv_pipeline_name(&inventory, 6, 0, 0, &no_flags).as_deref(),
        Some("linear_gemv_ishizuki_k6"),
        "the head no longer resolves to the 3-word slot"
    );
    // The 2-word window is *available* at 3 bpw, so the fall-through is a
    // property of the rate, not of the selector's logic.
    assert_eq!(
        ishizuki_gemv_pipeline_name(&inventory, 3, 0, 0, &no_flags).as_deref(),
        Some("linear_gemv_ishizuki_k3_w2"),
        "at 3 bpw with perm_mode 0 the selector takes _w2, so the head's \
         3-word result is caused by the absence of _k6_w2 and nothing else"
    );
}

/// Question 4, answered: `permutation_mode` is a **global** decision. One
/// `layout` key in `quantization_config.json` sets it for every layer
/// projection at once (`model.rs:3525-3542`), and `lm_head` is not in that list,
/// so it keeps the `0` from `QuantizedLinear::new` (`model.rs:340`). There is no
/// per-tensor layout key anywhere -- `layout` is one `Option<String>` on the
/// config struct (`config.rs:392`) and nothing else reads it.
///
/// So the seven layer projections agree, and the head is the single outlier
/// that changes kernel family. Which is exactly what finding 6 needs: the head is
/// the *only* non-colmajor projection in the step.
#[test]
fn permutation_mode_is_global_and_the_head_is_the_outlier() {
    assert_eq!(
        MODEL_RS
            .matches("linear.permutation_mode = 1;")
            .count(),
        1,
        "the colmajor assignment is no longer a single unconditional write; \
         re-audit which tensors it covers"
    );
    assert!(
        MODEL_RS.contains("if metadata.quantization.layout.as_deref() == Some(\"colmajor\") {"),
        "the gate on quantization_config.json's `layout` moved; this audit read it \
         at model.rs:3525"
    );
    // The list it iterates: all seven, and *not* the head.
    let table = dispatch_table();
    let layers: Vec<u32> = table
        .iter()
        .filter(|row| row.projection != "lm_head")
        .map(|row| row.perm_mode)
        .collect();
    assert!(
        layers.iter().all(|mode| *mode == 1),
        "every layer projection must be colmajor; got {layers:?}"
    );
    assert_eq!(
        table.last().map(|row| row.perm_mode),
        Some(0),
        "lm_head must stay at perm_mode 0"
    );
}

/// Question 5, answered with the exact asymmetry rather than a yes/no.
///
/// `o_proj` (`model.rs:1941-1946`) and `down_proj` (`model.rs:2242-2247`) both
/// default split-K **on**: the predicate is `!env_flag(...)`, with no opt-in.
/// `q/k/v` (`linear.rs:2780-2796`) and `gate/up` (`linear.rs:2590-2597`)
/// default split-K **off** for the same rate, and the `_sk2` fused kernels are
/// reachable only behind `KAPLUMBA_USE_SK` / `KAPLUMBA_QKV_SK2`. So four
/// projections, one bit rate, three kernels, one of them split twice over.
#[test]
fn o_proj_and_down_proj_are_the_only_split_k_projections() {
    let table = dispatch_table();
    let split_k: Vec<&str> = table
        .iter()
        .filter(|row| row.kernel.contains("_sk"))
        .map(|row| row.projection)
        .collect();
    assert_eq!(
        split_k,
        vec!["o_proj (sliding)", "o_proj (full attn)", "down_proj"],
        "the split-K set changed; o_proj and down_proj are expected to be the only \
         ones, because their default is `!env_flag(\"KAPLUMBA_NO_SK2\")` -- an \
         opt-OUT (model.rs:1941, 2242), unlike every fused entry point"
    );
    for row in &table {
        if row.projection.starts_with("q_proj") || row.projection.starts_with("k_proj") {
            assert!(
                !row.kernel.contains("_sk"),
                "{} runs a split-K kernel; the fused forms default to non-split-K",
                row.projection
            );
        }
    }
    // The three distinct layer kernels. If this drops to two, a fusion landed.
    let mut layer_kernels: Vec<&str> = table
        .iter()
        .filter(|row| row.projection != "lm_head")
        .map(|row| row.kernel)
        .collect();
    layer_kernels.sort_unstable();
    layer_kernels.dedup();
    assert_eq!(
        layer_kernels.len(),
        3,
        "expected exactly three distinct layer kernels (fused3_2col, fused2_2col, \
         colmajor_2col_sk2), got {layer_kernels:?}"
    );
}

/// Question 1/2 headline: no projection falls through to the EXL3 arm.
///
/// `ishizuki_gemv_pipeline` refuses `packed_offset % 4 != 0` (`linear.rs:1520`),
/// and that refusal is the *only* way a decode tensor reaches
/// `linear_gemv_exl3_*`. All 411 trellis tensors of `Gxma-repacked` are 4-byte
/// aligned (checked against the safetensors headers: q/k/v/o/gate/up/down x 60
/// plus lm_head, none at `% 4 == 2`), so nothing does.
///
/// The consequence matters for the notes: `linear.rs:1937-1940` says the `_mcg`
/// family and the NR0 axis "only ever runs for lm_head, whose trellis starts at
/// `% 4 == 2`". On the **repacked** checkpoint that is false in both halves --
/// `lm_head` is 4-byte aligned, so it takes the ishizuki arm, and therefore
/// nothing runs the EXL3 family at all.
#[test]
fn no_projection_falls_through_to_the_exl3_arm() {
    let inventory = sorted_ishizuki_inventory();
    for row in dispatch_table() {
        assert!(
            !row.kernel.starts_with("linear_gemv_exl3_"),
            "{} dispatches the EXL3 arm ({}); every trellis of Gxma-repacked is \
             4-byte aligned, so nothing should",
            row.projection, row.kernel
        );
        // The selector resolves for every row, which is the condition for never
        // falling through: `ishizuki_gemv_pipeline` returning `None` is the ONLY
        // way `encode_gemv_raw` reaches `linear_gemv_exl3_*` (linear.rs:2182).
        // Note this is *not* the same claim as "the selector is on this
        // projection's path" -- for a colmajor tensor it would resolve to
        // `_colmajor_contig4`, which is exactly the kernel this audit finds
        // unreachable. What matters here is `None`, not *which* name.
        assert!(
            ishizuki_gemv_pipeline_name(&inventory, row.bits, 0, row.perm_mode, &no_flags).is_some(),
            "{} resolves to None, so it takes the EXL3 arm; every trellis of \
             Gxma-repacked is 4-byte aligned",
            row.projection
        );
    }
    // The head is the only row whose *resolution* is the kernel it dispatches.
    assert_eq!(
        ishizuki_gemv_pipeline_name(&inventory, 6, 0, 0, &no_flags).as_deref(),
        Some("linear_gemv_ishizuki_k6"),
    );
}

/// Renders the table and checks its accounting, so the row fields are load
/// bearing and the answer is readable without running a model.
///
/// Run with `--nocapture` to see it. This is the deliverable: the whole audit
/// as one block.
#[test]
fn the_dispatch_table_is_the_one_printed_above() {
    let table = dispatch_table();
    // Layer accounting. 60 layers x 7 projections = 420, less the 10 missing
    // `v_proj` tensors, plus `lm_head` = **411** -- the same count
    // `linear.rs:1937` calls "410 of this checkpoint's 411 tensors", which is an
    // independent corroboration that the population split here is the real one.
    let covered: usize = table.iter().map(|row| row.layers).sum();
    let expected: usize = 60 * 7 - (60 - SLIDING_LAYERS) + 1;
    assert_eq!(
        covered, expected,
        "the table's layer counts do not add up to 411: 7 projections x 60 \
         layers, v_proj only on {SLIDING_LAYERS}, lm_head on 1"
    );
    assert_eq!(
        table.iter().filter(|row| row.layers == 1).count(),
        1,
        "exactly one row is not a per-layer projection"
    );
    let mut lines = vec![format!(
        "{:<20} {:>5} {:>17} {:>3}  {:<42} [{}]",
        "projection", "layers", "in -> out", "bpw", "kernel", "chosen by"
    )];
    for row in &table {
        lines.push(format!(
            "{:<20} {:>5} {:>6} -> {:<8} {:>3}  {:<42} [{}]",
            row.projection,
            row.layers,
            row.in_features,
            row.out_features,
            row.bits,
            row.kernel,
            row.chosen_by,
        ));
    }
    println!("\n{}", lines.join("\n"));
}

/// The table above is a transcription. This is the part that reads the real
/// code, so that editing the dispatch fails here instead of quietly invalidating
/// the audit.
///
/// Each entry is `(source, exact substring, minimum occurrences, what breaks)`.
const PREDICATES: &[(&str, &str, usize, &str)] = &[
    (
        "model.rs",
        "!env_flag(\"KAPLUMBA_NO_SK2\")",
        3,
        "o_proj/down_proj split-K default, and the cross-layer fusion it is tied to",
    ),
    (
        "linear.rs",
        "if perm_mode == 1 && out_features % 32 == 0 && !env_flag(\"KAPLUMBA_NO_2COL\") {",
        1,
        "the 2-column non-split-K arm of encode_gemv_raw",
    ),
    (
        "linear.rs",
        "if perm_mode == 1 || env_flag(\"KAPLUMBA_REPACKED_GEMV\") {",
        1,
        "the colmajor gate of ishizuki_gemv_pipeline",
    ),
    (
        "linear.rs",
        "format!(\"linear_gemv_ishizuki_k{bits}_colmajor_contig4\")",
        1,
        "the production colmajor lookup -- the kernel this audit found dead",
    ),
    (
        "linear.rs",
        "format!(\"linear_gemv_ishizuki_k{bits}_colmajor\")",
        1,
        "the production colmajor fallback lookup",
    ),
    (
        "linear.rs",
        "format!(\"linear_gemv_ishizuki_k{bits}_w2\")",
        1,
        "the _w2 window lookup",
    ),
    (
        "linear.rs",
        "format!(\"linear_gemv_ishizuki_k{bits}_colmajor_2col_sk2\")",
        1,
        "the kernel o_proj and down_proj actually run",
    ),
    (
        "linear.rs",
        "\"linear_gemv_ishizuki_k3_fused3_2col\"",
        1,
        "the kernel q/k/v actually run",
    ),
    (
        "linear.rs",
        "\"linear_gemv_ishizuki_k3_fused2_2col\"",
        2,
        "the kernel q/k (full attn) and gate/up actually run",
    ),
    (
        "linear.rs",
        "!env_flag(\"KAPLUMBA_NO_FUSED3_2COL\")",
        1,
        "q/k/v defaulting to fused3_2col",
    ),
    (
        "linear.rs",
        "!env_flag(\"KAPLUMBA_NO_FUSED2_2COL\")",
        2,
        "q/k and gate/up defaulting to fused2_2col",
    ),
    (
        "linear.rs",
        "if env_flag(\"KAPLUMBA_USE_SK\") || env_flag(\"KAPLUMBA_QKV_SK2\") {",
        1,
        "the fused3 split-K arm staying opt-in",
    ),
    (
        "linear.rs",
        "if env_flag(\"KAPLUMBA_OWN_GEMV\") {",
        1,
        "ishizuki_gemv_pipeline's early return",
    ),
    (
        "linear.rs",
        "if !env_flag(\"KAPLUMBA_ISHIZUKI_3WORD\") {",
        1,
        "the _w2 window choice",
    ),
];

/// The same idea, but over **whole conditions** rather than single tokens.
///
/// A token-level tripwire cannot see a change made *around* the token it pins:
/// appending `&& env_flag("KAPLUMBA_NEVER")` to the fused3 predicate leaves
/// `!env_flag("KAPLUMBA_NO_FUSED3_2COL")` present and the test green while
/// `q/k/v` silently moves off `_2col`. That was tried as mutation D and it
/// passed, so the collapsed-whitespace conditions below are what actually
/// pin the dispatch. Confirmed by re-running the mutation against these.
const COLLAPSED_PREDICATES: &[(&str, &str, usize, &str)] = &[
    (
        "linear.rs",
        "let (pipeline, groups) = if out_features_a % 32 == 0 && out_features_b % 32 == 0 \
         && out_features_c % 32 == 0 && !env_flag(\"KAPLUMBA_NO_FUSED3_2COL\") {",
        1,
        "q/k/v defaulting to fused3_2col (all 50 sliding layers)",
    ),
    (
        "linear.rs",
        "if out_features_a % 32 == 0 && out_features_b % 32 == 0 \
         && !env_flag(\"KAPLUMBA_NO_FUSED2_2COL\") {",
        1,
        "q/k defaulting to fused2_2col (the 10 full-attention layers)",
    ),
    (
        "linear.rs",
        "} else if !env_flag(\"KAPLUMBA_NO_FUSED2_2COL\") \
         && out_features_gate % 32 == 0 && out_features_up % 32 == 0",
        1,
        "gate/up defaulting to fused2_2col",
    ),
    (
        "model.rs",
        "let use_sk2 = !use_sk4 && !env_flag(\"KAPLUMBA_NO_SK2\") \
         && !env_flag(\"KAPLUMBA_O_NOSK\") && self.weights.o_proj.permutation_mode == 1",
        1,
        "o_proj split-K being on by DEFAULT (an opt-out, not an opt-in)",
    ),
    (
        "model.rs",
        "let use_sk2 = !use_sk4 && !env_flag(\"KAPLUMBA_NO_SK2\") \
         && !env_flag(\"KAPLUMBA_D_NOSK\") && self.weights.down_proj.permutation_mode == 1",
        1,
        "down_proj split-K being on by DEFAULT",
    ),
    (
        "model.rs",
        "if decode && self.weights.q_proj.permutation_mode == 1 \
         && self.weights.v_proj.is_some() {",
        1,
        "the fused3 (q/k/v) entry point, which needs v_proj to exist",
    ),
    (
        "model.rs",
        "} else if decode && self.weights.q_proj.permutation_mode == 1 {",
        2,
        "the fused2 (q/k only) entry point for the 10 full-attention layers",
    ),
    (
        "model.rs",
        "if decode && self.weights.gate_proj.permutation_mode == 1 {",
        2,
        "the fused2_mlp (gate/up) entry point",
    ),
    (
        "model.rs",
        "if decode && self.weights.o_proj.permutation_mode == 1 \
         && self.weights.gate_proj.permutation_mode == 1 {",
        1,
        "the shared o_proj/gate_up decode branch",
    ),
];

/// Collapse every whitespace run to a single space, so a predicate that is
/// written across four lines can be pinned as one string. `model.rs` and
/// `linear.rs` are both rustfmt'd, so this cannot hide a semantic change -- it
/// only removes formatting noise.
fn collapse_whitespace(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut in_space = false;
    for ch in source.chars() {
        if ch.is_whitespace() {
            in_space = true;
            continue;
        }
        if in_space && !out.is_empty() {
            out.push(' ');
        }
        in_space = false;
        out.push(ch);
    }
    out
}

#[test]
fn the_whole_branch_conditions_are_still_the_ones_audited() {
    let collapsed = [
        ("model.rs", collapse_whitespace(MODEL_RS)),
        ("linear.rs", collapse_whitespace(LINEAR_RS)),
    ];
    for (source, predicate, minimum, breaks) in COLLAPSED_PREDICATES {
        // Rust string literals in this table are written across lines for
        // legibility; compare against the same collapse.
        let predicate = collapse_whitespace(predicate);
        let haystack = collapsed
            .iter()
            .find(|(name, _)| name == source)
            .map(|(_, text)| text.as_str())
            .unwrap_or_else(|| panic!("unknown source {source}"));
        let found = haystack.matches(&predicate).count();
        assert!(
            found >= *minimum,
            "{source} no longer contains the condition {predicate:?} at least \
             {minimum} time(s) (found {found}). This breaks: {breaks}. Redo the \
             dispatch audit in this file -- do not relax the match."
        );
    }
}

#[test]
fn the_branch_predicates_are_still_the_ones_audited() {
    for (source, predicate, minimum, breaks) in PREDICATES {
        let haystack = match *source {
            "model.rs" => MODEL_RS,
            "linear.rs" => LINEAR_RS,
            other => panic!("unknown source {other}"),
        };
        let found = haystack.matches(predicate).count();
        assert!(
            found >= *minimum,
            "{source} no longer contains {predicate:?} at least {minimum} time(s) \
             (found {found}). This breaks: {breaks}. The dispatch audit in this \
             file has to be redone -- do not just relax the count."
        );
    }
}

// ---------------------------------------------------------------------------
// 5. The latent fail-open (finding 4)
// ---------------------------------------------------------------------------

/// `linear.rs:1526-1532` argues the `_w2` name lookup is chosen over a parallel
/// `bool` because a bool "would fail open (dispatching a kernel that does not
/// exist) rather than closed". That argument holds for `_w2`. It does **not**
/// hold for the `perm_mode == 1` lookup immediately above it: only k3 has
/// colmajor kernels, so a colmajor checkpoint at any other integer rate falls
/// out of that `if` block, past `_w2`, and lands on the **non**-colmajor
/// `linear_gemv_ishizuki_k{N}` -- which runs, returns finite numbers, and
/// decodes garbage.
///
/// Nothing in the tree triggers it today (one `layout` key, one rate, one
/// head rate of 6 with `perm_mode == 0`), and `"head_bits": 8` is currently
/// accepted by the loader (`safetensors.rs:1211-1233` only refuses 0 and 9).
/// Pinned so that adding the k8 colmajor kernel, or shipping a colmajor 8 bpw
/// head, does not quietly produce a wrong-but-fast model.
#[test]
fn perm_mode_one_only_resolves_to_colmajor_at_three_bpw() {
    let inventory = sorted_ishizuki_inventory();
    let mut resolves_colmajor = Vec::new();
    let mut falls_open = Vec::new();
    for bits in 1u8..=8 {
        match ishizuki_gemv_pipeline_name(&inventory, bits, 0, 1, &no_flags) {
            Some(name) if name.contains("_colmajor") => resolves_colmajor.push(bits),
            Some(name) => falls_open.push((bits, name)),
            None => {}
        }
    }
    assert_eq!(
        resolves_colmajor,
        vec![3],
        "colmajor kernels now exist for these rates: {resolves_colmajor:?}. Every \
         additional rate is another checkpoint that dispatches correctly instead \
         of decoding garbage -- which is the good direction, but update the audit."
    );
    // The shape of the hazard, recorded rather than merely described: for these
    // rates a perm_mode == 1 tensor resolves to a kernel whose name says
    // nothing about colmajor layout.
    for (bits, name) in &falls_open {
        assert!(
            !name.contains("colmajor"),
            "rate {bits} resolved to {name}, which is fine -- but it means the \
             fall-open set shrank and the audit above needs re-reading"
        );
    }
    assert_eq!(
        falls_open.len(),
        7,
        "expected rates 1,2,4,5,6,7,8 to fall open, got {falls_open:?}"
    );
}

// ---------------------------------------------------------------------------
// 6. Arithmetic on the notes' attributed table (question 6)
// ---------------------------------------------------------------------------

/// Bytes streamed per decode token, from the checkpoint's own shapes.
///
/// `spec/decode-25tok-peak-bandwidth/implementation_notes.md:492-498` attributes
/// GB/token as `qkv 1.982`, `o 0.991`. Both are exactly 60 x the **sliding**
/// projection: 60 x 33 030 144 = 1 981 808 640, and 60 x 16 515 072 =
/// 990 904 320. The checkpoint has 10 full-attention layers whose `q_proj` is
/// `5376 -> 16384` and `o_proj` is `16384 -> 5376`, so the real figures are
/// 2.023 GB and 1.156 GB.
///
/// This does not make any layer projection slower -- it makes `o_proj` **16.7%
/// faster** than the table says, and moves it from the projection the notes
/// handicap ("`o_proj` at K=8192 is 150 GB/s ... still well under the long-K
/// asymptote") to the fastest of the four. So there is no outlier dragging the
/// average; the byte model is what is wrong.
#[test]
fn the_notes_byte_model_assumes_every_layer_is_sliding() {
    let bits = 3.0_f64;
    let bytes = |out: usize, inn: usize| out as f64 * inn as f64 * bits / 8.0;

    let qkv_sliding = bytes(8192, HIDDEN) + 2.0 * bytes(SLIDING_KV, HIDDEN);
    let qkv_full = bytes(16384, HIDDEN) + bytes(FULL_KV, HIDDEN);
    let qkv = SLIDING_LAYERS as f64 * qkv_sliding + FULL_LAYERS as f64 * qkv_full;
    let o = SLIDING_LAYERS as f64 * bytes(HIDDEN, 8192)
        + FULL_LAYERS as f64 * bytes(HIDDEN, 16384);
    let gate_up = 60.0 * 2.0 * bytes(INTERMEDIATE, HIDDEN);
    let down = 60.0 * bytes(HIDDEN, INTERMEDIATE);
    let head = VOCAB as f64 * HIDDEN as f64 * 6.0 / 8.0;

    // The notes' numbers, recomputed so the comparison is exact rather than
    // remembered.
    let notes_qkv = 60.0 * qkv_sliding;
    let notes_o = 60.0 * bytes(HIDDEN, 8192);
    assert!((notes_qkv / 1e9 - 1.982).abs() < 0.001, "notes say 1.982 GB/token");
    assert!((notes_o / 1e9 - 0.991).abs() < 0.001, "notes say 0.991 GB/token");
    assert!(
        qkv > notes_qkv && o > notes_o,
        "the two-population model must exceed the all-sliding one it contradicts"
    );

    // Re-derive the implied GB/s from the notes' own ms/step figures
    // (implementation_notes.md:471-481) with the corrected byte counts.
    let ms = [
        ("qkv", qkv, 15.82_f64),
        ("o_proj", o, 6.63),
        ("gate_up", gate_up, 40.34),
        ("down", down, 20.63),
        ("head", head, 4.17),
    ];
    let mut corrected = Vec::new();
    for (name, gb, step_ms) in ms {
        corrected.push((name, gb / 1e9, gb / step_ms / 1e6));
    }
    let find = |want: &str| corrected.iter().find(|(n, _, _)| *n == want).unwrap();

    // The head's figure is unaffected -- its bytes were already right.
    assert!((find("head").2 - 253.0).abs() < 1.5, "head should stay ~253 GB/s");
    // o_proj is the outlier, in the *good* direction.
    assert!(
        find("o_proj").2 > 170.0,
        "o_proj's implied rate is {:.1} GB/s on the real byte count; the notes \
         print 150 from an all-sliding model",
        find("o_proj").2
    );
    assert!(
        (find("qkv").2 - 128.0).abs() < 3.0,
        "qkv's implied rate is {:.1} GB/s",
        find("qkv").2
    );
    // gate_up and down are unchanged, and still the ~126-129 the notes print.
    assert!((find("gate_up").2 - 129.0).abs() < 2.0);
    assert!((find("down").2 - 126.0).abs() < 2.0);

    // Total streamed, and the ceiling that follows from it.
    let total = qkv + o + gate_up + down + head;
    let notes_total = 11.834e9;
    assert!(
        total > notes_total,
        "the real per-token stream is {:.3} GB, the notes print 11.834",
        total / 1e9
    );
    // 203 GB/s is the measured 3 bpw long-K peak (implementation_notes.md:503).
    let ceiling_ms = total / (203.0e9) * 1e3;
    assert!(
        (ceiling_ms - 59.3).abs() < 0.5,
        "the peak-limited step is {ceiling_ms:.1} ms = {:.2} tok/s, not the \
         58.3 ms / 17.2 tok/s the notes print for 11.834 GB",
        1e3 / ceiling_ms
    );
}

/// Finding 6, as arithmetic rather than assertion: the head's 253 GB/s is the
/// same kernel at a different rate, not a better kernel.
///
/// The only structural difference between `linear_gemv_ishizuki_k6` (head) and
/// `linear_gemv_ishizuki_k3_colmajor_2col_sk2` (layers) is `KA`. Lane-load
/// amplification per useful weight byte is `96/48 = 2.000` at 6 bpw against
/// `64/24 = 2.667` at 3 bpw
/// (`spec/decode-3bpw-amplification/implementation_notes.md:134`) -- worth at
/// most 1.33x. The remaining ~1.5x of the observed 2.01x is grid size: 16384
/// threadgroups for the head against 336 for o/down and 1344 for gate_up.
///
/// This test pins the arithmetic so that "the head is 2x faster" cannot be
/// cited as evidence that a layer kernel has 2x of headroom. It asserts nothing
/// about achievable bandwidth; it only constrains the *attribution*.
#[test]
fn the_heads_advantage_is_rate_and_grid_not_a_better_kernel() {
    let amp_3bpw: f64 = 64.0 / 24.0; // 32 lanes x 2 word-loads / 24 distinct words
    let amp_6bpw: f64 = 96.0 / 48.0; // 32 lanes x 3 word-loads / 48 distinct words
    assert!((amp_3bpw - 2.6666666666666665).abs() < 1e-9);
    assert!((amp_6bpw - 2.0).abs() < 1e-9);

    let observed_ratio: f64 = 253.0 / 126.0;
    let from_amplification: f64 = amp_3bpw / amp_6bpw;
    assert!(
        (from_amplification - 1.3333).abs() < 0.001,
        "amplification alone buys {from_amplification:.3}x"
    );
    assert!(
        observed_ratio > from_amplification,
        "the observed ratio {observed_ratio:.2}x exceeds the {from_amplification:.2}x \
         amplification alone predicts, so the remainder is occupancy, not kernel \
         quality"
    );
    let residual = observed_ratio / from_amplification;
    assert!(
        (residual - 1.5).abs() < 0.1,
        "the residual is {residual:.2}x, which is the grid-size term (16384 \
         threadgroups at the head against 336-1344 per layer), not a better kernel"
    );

    // Grid arithmetic, from the dispatch the audit established.
    let head_groups = (VOCAB / 16) as u64; // linear.rs:1479-1480, uncapped (u64::MAX)
    let o_groups = (((HIDDEN / 16) / 2) * 2) as u64; // linear.rs:2258, encode_gemv_raw_sk2
    let gate_up_groups = ((((INTERMEDIATE / 16) * 2) / 2)) as u64; // linear.rs:2597
    assert_eq!(head_groups, 16384);
    assert_eq!(o_groups, 336);
    assert_eq!(gate_up_groups, 1344);
    assert!(
        head_groups / o_groups >= 40,
        "the head's grid is {}x o_proj's; that is the other half of the 2x",
        head_groups / o_groups
    );
}