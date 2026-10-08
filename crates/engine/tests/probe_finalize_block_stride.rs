//! PERF PROBE (read-only, no GPU): the finalize/norm kernels' block-loop stride
//! must equal the number of simdgroups their dispatch creates.
//!
//! ## Why this file exists
//!
//! `linear.metal:1187` ran `for (uint b = sg; b < params.blocks; b += 8u)` in
//! `finalize_post_ffn_and_next_input_norm_transform`, a kernel dispatched with
//! **1 threadgroup of 1024 threads = 32 simdgroups** (`linear.rs`
//! `encode_cross_layer_finalize_and_norm` -> `dispatch_groups(.., 1, 1024)`,
//! and the comment there says so explicitly: "1024 threads, because the
//! layer-L half is the `_sk2` finalize"). A stride of 8 left simdgroups 8-31
//! idle and ran `ceil(42/8) = 6` rounds where its siblings run
//! `ceil(42/32) = 2`. It was carried over verbatim from the 256-thread
//! `input_norm_and_qkv_transform` (`linear.metal:1238`), where `8u` *is*
//! correct, during the T5.6 widen that bumped the other four loops.
//!
//! The kernel is one of two dispatched as a single threadgroup, so it cannot
//! reach more than one core, and it moves ~129 KB per layer per token through
//! that core. That makes the stride the difference between the core doing 12.5%
//! and 37.5% of its own block work per round.
//!
//! ## The invariant
//!
//! `sg` is `thread_position_in_threadgroup / 32` and takes every value in
//! `0 .. threads/32`. Every `for (uint b = sg; b < params.blocks; b += Nu)`
//! therefore **must** have `N == threads / 32`:
//!
//! * `N < threads/32` idles whole simdgroups (the bug above).
//! * `N > threads/32` skips blocks: `b` is reachable iff `b % N == sg % N`, so
//!   with `sg < threads/32 <= N` some residue classes are never taken and their
//!   elements are never written.
//!
//! The loop body is a pure elementwise map over `blocks` of 128 elements, each
//! block writing its own disjoint `normed[base..]` / `transformed[base..]`
//! slots with no cross-block reduction and no accumulation across iterations,
//! so changing the stride **cannot change a value** -- only which simdgroup
//! computes which block. That is why this is a safe correctness-preserving fix,
//! and why `fused_bit_exact` is the wrong place to test it: that suite does not
//! reach these kernels at all.
//!
//! ## Sabotage log (AGENTS.md §3.4 -- each broken, each failure observed)
//!
//! | sabotage | observed failure |
//! |---|---|
//! | revert `:1187` to `b += 8u` | `every_block_loop_stride_matches_its_dispatch` FAILS: "stride 8u but the kernel is dispatched with 1024 threads = 32 simdgroups" |
//! | bump `:1238` (`input_norm_and_qkv_transform`) to `32u` | same test FAILS the other way: 256 threads = 8 simdgroups |
//! | set `:1187` to `64u` | `no_block_is_skipped` FAILS: "covers 21 of 42 blocks" |

use std::fs;
use std::path::PathBuf;

const KERNELS_LINEAR: &str = "crates/kernels/src/linear.metal";
const LINEAR_RS: &str = "crates/kernels/src/linear.rs";

/// `(metal kernel, dispatch tail threads, literal that must appear in
/// linear.rs as this kernel's pipeline field)`.
///
/// The thread counts are read out of `linear.rs`'s `dispatch_groups(..., 1,
/// THREADS)` calls; the evidence literal is the pipeline *field* name, which
/// drops the `linear_` prefix the kernel name carries and is, for the
/// cross-layer kernel, `cross_layer_pipeline` rather than the kernel name.
const KERNELS: &[(&str, usize, &str)] = &[
    (
        "linear_finalize_and_post_ffn_norm_scale",
        256,
        "finalize_and_post_ffn_norm_scale_pipeline",
    ),
    (
        "post_attn_pre_ffn_transform",
        256,
        "post_attn_pre_ffn_transform_pipeline",
    ),
    (
        "linear_finalize_o_proj_and_pre_ffn_transform",
        256,
        "finalize_o_proj_and_pre_ffn_transform_pipeline",
    ),
    (
        "input_norm_and_qkv_transform",
        256,
        "input_norm_and_qkv_transform_pipeline",
    ),
    (
        "linear_finalize_and_post_ffn_norm_scale_sk2",
        1024,
        "finalize_and_post_ffn_norm_scale_sk2_pipeline",
    ),
    (
        "linear_finalize_and_post_ffn_norm_scale_sk4",
        1024,
        "finalize_and_post_ffn_norm_scale_sk4_pipeline",
    ),
    (
        "linear_finalize_o_proj_and_pre_ffn_transform_sk2",
        1024,
        "finalize_o_proj_and_pre_ffn_transform_sk2_pipeline",
    ),
    (
        "linear_finalize_o_proj_and_pre_ffn_transform_sk4",
        1024,
        "finalize_o_proj_and_pre_ffn_transform_sk4_pipeline",
    ),
    (
        "finalize_post_ffn_and_next_input_norm_transform",
        1024,
        "cross_layer_pipeline",
    ),
];

/// Production `width / 128`. `hidden_size` is 5376, so every one of these
/// kernels runs 42 blocks.
const BLOCKS: usize = 42;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("kaplumba-engine has a parent")
        .parent()
        .expect("kaplumba has a parent")
        .to_path_buf()
}

fn read(relative: &str) -> String {
    let path = workspace_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {relative} at {}: {error}", path.display()))
}

/// The body of `kernel`, from its `kernel void NAME(` line to the next
/// column-0 `}`.
fn kernel_body<'a>(source: &'a str, kernel: &str) -> &'a str {
    let start = source
        .find(&format!("kernel void {kernel}("))
        .unwrap_or_else(|| panic!("{kernel} is not defined in {KERNELS_LINEAR}"));
    let rest = &source[start..];
    let end = rest.find("\n}\n").map(|i| i + 3).unwrap_or(rest.len());
    &rest[..end]
}

/// Every `for (uint b = sg; b < params.blocks; b += Nu)` stride in `body`,
/// paired with its 1-based line number inside `body`.
fn block_strides(body: &str) -> Vec<(u32, usize)> {
    let mut found = Vec::new();
    for (index, line) in body.lines().enumerate() {
        let needle = "for (uint b = sg; b < params.blocks; b += ";
        let Some(offset) = line.find(needle) else {
            continue;
        };
        let tail = &line[offset + needle.len()..];
        let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
        assert!(
            !digits.is_empty() && tail.ends_with("u) {"),
            "unexpected block-loop spelling at body line {}: {line:?}",
            index + 1
        );
        let stride: u32 = digits
            .parse()
            .unwrap_or_else(|error| panic!("stride {digits:?} at body line {}: {error}", index + 1));
        found.push((stride, index + 1));
    }
    assert!(
        !found.is_empty(),
        "no block loop found; the probe's parsing or the kernel drifted"
    );
    found
}

#[test]
fn every_block_loop_stride_matches_its_dispatch() {
    let source = read(KERNELS_LINEAR);
    let rust = read(LINEAR_RS);
    let mut checked = 0_usize;
    for (kernel, threads, field) in KERNELS {
        assert!(
            rust.contains(field),
            "linear.rs never references {field}; the probe's dispatch mapping for \
             {kernel} drifted"
        );
        let body = kernel_body(&source, kernel);
        let expected = (threads / 32) as u32;
        for (stride, line) in block_strides(body) {
            assert_eq!(
                stride, expected,
                "{kernel} body line {line}: block-loop stride {stride}u but the kernel \
                 is dispatched with {threads} threads = {expected} simdgroups"
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 9,
        "only {checked} block loops checked; expected at least 9 across {} kernels",
        KERNELS.len()
    );
}

#[test]
fn no_block_is_skipped() {
    // Independent of the stride assertion above: `b` is reachable from `sg`
    // iff `b % stride == sg % stride`, and `sg` ranges over `0..threads/32`.
    // So the covered blocks are exactly `{b : b % stride < threads/32}`, and
    // that must be all `BLOCKS` of them or elements are never written.
    let source = read(KERNELS_LINEAR);
    for (kernel, threads, _) in KERNELS {
        let simdgroups = (threads / 32) as u32;
        for (stride, line) in block_strides(kernel_body(&source, kernel)) {
            let covered = (0..BLOCKS as u32)
                .filter(|b| b % stride < simdgroups)
                .count();
            assert_eq!(
                covered, BLOCKS,
                "{kernel} body line {line}: stride {stride}u with {simdgroups} simdgroups \
                 reaches only {covered} of {BLOCKS} blocks -- b is reachable iff \
                 b % {stride} < {simdgroups}"
            );
        }
    }
}

#[test]
fn the_cross_layer_finalize_is_not_under_parallelised() {
    // The specific regression, isolated so a future failure names it.
    let source = read(KERNELS_LINEAR);
    let strides: Vec<u32> = block_strides(kernel_body(
        &source,
        "finalize_post_ffn_and_next_input_norm_transform",
    ))
    .into_iter()
    .map(|(stride, _)| stride)
    .collect();
    assert_eq!(
        strides,
        vec![32, 32],
        "finalize_post_ffn_and_next_input_norm_transform is dispatched with 1024 threads \
         (32 simdgroups) and has two block loops; a stride of 8u idles 24 of its 32 \
         simdgroups and runs 6 rounds instead of 2. Observed {strides:?}."
    );
}

#[test]
fn a_256_thread_kernel_keeps_its_stride_of_eight() {
    // The other direction, so nobody "fixes" the 256-thread kernel into a bug.
    let source = read(KERNELS_LINEAR);
    for (kernel, threads, _) in KERNELS {
        if *threads != 256 {
            continue;
        }
        let strides: Vec<u32> = block_strides(kernel_body(&source, kernel))
            .into_iter()
            .map(|(stride, _)| stride)
            .collect();
        assert!(
            strides.iter().all(|stride| *stride == 8),
            "{kernel} is a 256-thread dispatch (8 simdgroups); every block-loop stride \
             should stay 8u. Observed {strides:?}."
        );
    }
}
