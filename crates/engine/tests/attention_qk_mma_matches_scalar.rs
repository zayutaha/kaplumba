//! T1 (spec/prefill-120tps): does the simdgroup-MMA QK scoring compute the
//! same attention output as the scalar per-lane dot product?
//!
//! # What is being claimed
//!
//! `AttentionKernels::encode_online_attention_qk` dispatches the *same* prefill
//! request to either `online_attention_gqa_*` (scalar QK, `attention_gqa_body`)
//! or `online_attention_gqa_mma_*` (MMA QK, `attention_gqa_mma_body`). Only the
//! QK step differs by construction: the online softmax, the double-buffered
//! weight tile, the PV fold and the epilogue are copied line-for-line in the
//! Metal source. This file runs both on identical inputs and compares the
//! stored fp16 outputs.
//!
//! # Why a tolerance and not `to_bits()`
//!
//! The MMA reassociates the dot product: scalar accumulates `q[d] * k[d]` in
//! `d` order (`attention.metal:846-849`), the tensor unit accumulates in its
//! own hardware order. Floating-point addition is not associative, so bit
//! equality is not even well-defined across the two. What *is* well-defined is
//! that both are correctly-rounded-enough evaluations of the same sum of 256
//! (or 512) fp16-exact products: the divergence is a few fp32 ulps of the dot,
//! which survives `scale`, `exp` and the normalisation as a much smaller fp16
//! output delta. The threshold below is set from the measured worst case (see
//! `spec/prefill-120tps/notes_T1_T2.md`) with headroom, and the sabotage check
//! (AGENTS.md 3.4) shows a real QK bug produces deltas orders of magnitude
//! above it, so the threshold discriminates rather than rubber-stamps.
//!
//! # Shapes and boundaries covered
//!
//! * Both model shapes: sliding 32q/16kv/d256 w1024 (NSG=2) and full
//!   32q/4kv/d512 (NSG=8), plus GQA ratio 4 (NSG=4) and 1 (NSG=1), because the
//!   extraction lane map (`pair = sg >> 1`, src lane `4m + pair`) is a
//!   per-NSG layout and each ratio lands simdgroups on different pairs.
//! * Key counts straddling the 8-key MMA tile and the 32-key softmax tile:
//!   `< 8` (guarded tail staging and zero fill), `31/32/33` (n < 32 tail and
//!   first full tile), `1000` (mid-range, full tiles only), `2000` (window
//!   kills the oldest keys while the MMA still scores them).
//! * Dead keys by *window* (sliding case) and by *causal position*
//!   (`query_abs < key_count`), because the MMA computes scores for dead keys
//!   and forces them to NEG afterwards, which the scalar path never does.
//! * `queries_count > 1`, so several `(query, kv head)` threadgroups exist and
//!   the `tgid` mapping is exercised, not just one row.
//! * A non-unit `scale` (1/sqrt(512)), applied after extraction rather than
//!   inside the dot.
//!
//! The default path is pinned separately: `encode_online_attention` is
//! shape-aware (T2), so with no env override it must be bit-identical to the
//! `AttentionQk::Mma` arm on the full-attention shape and to the
//! `AttentionQk::Scalar` arm on the sliding shape. `attention_single_pass_bit_exact`
//! pins the scalar arm explicitly so its bit-exact claim is not affected.

use half::f16;
use kaplumba_kernels::attention::{
    AttentionEncodeRequest, AttentionKernels, AttentionQk,
};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// 0 for full attention (no window test), else the sliding window.
    window: usize,
}

/// NSG = q_heads / kv_heads = 2, the 50 sliding layers.
const SLIDING: Shape = Shape {
    name: "sliding 32q/16kv/d256 w1024 (NSG=2)",
    q_heads: 32,
    kv_heads: 16,
    head_dim: 256,
    window: 1024,
};
/// NSG = 8, the 10 full layers.
const FULL: Shape = Shape {
    name: "full    32q/4kv/d512 (NSG=8)",
    q_heads: 32,
    kv_heads: 4,
    head_dim: 512,
    window: 0,
};
/// NSG = 4, the Llama-3-style ratio; exercises pair columns 0..1 per sg pair.
const GQA4: Shape = Shape {
    name: "gqa4    8q/2kv/d256 (NSG=4)",
    q_heads: 8,
    kv_heads: 2,
    head_dim: 256,
    window: 0,
};
/// NSG = 1: the whole threadgroup is one simdgroup, row 0 only.
const NSG1: Shape = Shape {
    name: "nsg1    4q/4kv/d256 (NSG=1)",
    q_heads: 4,
    kv_heads: 4,
    head_dim: 256,
    window: 0,
};

/// (shape, key_count, queries_count, query_abs, scale)
///
/// `query_abs` below `key_count` makes causal-future keys dead; `window` makes
/// the oldest keys dead. `scale` 1.0 mirrors the engine's decode-scale
/// convention, and 1/sqrt(512) proves the post-extraction multiply is where
/// the scale is applied.
fn cases() -> Vec<(Shape, usize, usize, u32, f32)> {
    vec![
        // Tail below one MMA tile: guarded kbuf staging, rows zero-filled.
        (SLIDING, 5, 2, 5, 1.0),
        (FULL, 7, 1, 7, 1.0),
        // 33 = one full 32-tile plus a single-key tile (n = 1 path).
        (SLIDING, 33, 1, 33, 1.0),
        (FULL, 31, 3, 31, 1.0),
        (FULL, 32, 1, 32, 1.0),
        // Mid-range full tiles; window kills nothing yet.
        (SLIDING, 64, 4, 64, 1.0),
        (GQA4, 1000, 2, 1000, 1.0),
        // Causal: half the keys are dead (k_abs > q_abs).
        (FULL, 1000, 2, 500, 1.0),
        // Window: keys 0..(2000-1024) dead, MMA still scores them.
        (SLIDING, 2000, 3, 1999, 1.0),
        // Non-unit scale on both model shapes.
        (FULL, 129, 2, 129, 0.044_194_174),
        (SLIDING, 257, 1, 257, 0.062_5),
        // NSG=1 ratio.
        (NSG1, 65, 4, 65, 1.0),
    ]
}

struct Fixture {
    device: Device,
    queue: CommandQueue,
    kernels: AttentionKernels,
}

struct Case {
    queries: Buffer,
    keys: Buffer,
    values: Buffer,
    query_positions: Buffer,
    key_positions: Buffer,
    output_a: Buffer,
    output_b: Buffer,
    shape: Shape,
    key_count: usize,
    queries_count: usize,
    scale: f32,
}

impl Fixture {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let kernels = AttentionKernels::new()
            .unwrap_or_else(|e| panic!("AttentionKernels::new failed: {e}"));
        let queue = device.new_command_queue().to_owned();
        Some(Self {
            device,
            queue,
            kernels,
        })
    }

    /// Deterministic, well-scaled fp16 (same generator as
    /// `attention_single_pass_bit_exact`), so fp32 accumulation is exercised
    /// rather than saturating.
    fn row(&self, n: usize, salt: f32) -> Buffer {
        let bits: Vec<u16> = (0..n)
            .map(|i| {
                let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                    + ((i as f32 + salt) * 0.0031).cos() * 0.25;
                f16::from_f32(x).to_bits()
            })
            .collect();
        self.u16(&bits)
    }

    fn u16(&self, v: &[u16]) -> Buffer {
        self.device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn u32(&self, v: &[u32]) -> Buffer {
        self.device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn case(&self, shape: Shape, key_count: usize, queries_count: usize, query_abs: u32, scale: f32) -> Case {
        let kv = key_count * shape.kv_heads * shape.head_dim;
        let queries = queries_count * shape.q_heads * shape.head_dim;
        let output_elements = queries_count * shape.q_heads * shape.head_dim;
        let key_positions: Vec<u32> = (0..key_count as u32).collect();
        // Consecutive query positions starting at `query_abs`: with
        // `query_abs < key_count` every query sees a different causal cut.
        let query_positions: Vec<u32> =
            (0..queries_count as u32).map(|i| query_abs + i).collect();
        Case {
            queries: self.row(queries, 3.0),
            keys: self.row(kv, 11.0),
            values: self.row(kv, 29.0),
            query_positions: self.u32(&query_positions),
            key_positions: self.u32(&key_positions),
            output_a: self.device.new_buffer(
                (output_elements * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            output_b: self.device.new_buffer(
                (output_elements * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            shape,
            key_count,
            queries_count,
            scale,
        }
    }
}

impl Case {
    fn request<'a>(&'a self, output: &'a Buffer) -> AttentionEncodeRequest<'a> {
        AttentionEncodeRequest {
            queries: &self.queries,
            keys: &self.keys,
            values: &self.values,
            query_positions: &self.query_positions,
            key_positions: &self.key_positions,
            output,
            queries_count: self.queries_count,
            q_heads: self.shape.q_heads,
            kv_heads: self.shape.kv_heads,
            head_dim: self.shape.head_dim,
            key_count: self.key_count,
            scale: self.scale,
            window: self.shape.window,
        }
    }

    fn label(&self) -> String {
        format!(
            "{} keys={} queries={} scale={}",
            self.shape.name, self.key_count, self.queries_count, self.scale
        )
    }
}

fn read_u16(buffer: &Buffer, n: usize) -> Vec<u16> {
    let ptr = buffer.contents() as *const u16;
    (0..n).map(|i| unsafe { ptr.add(i).read() }).collect()
}

fn run(
    fx: &Fixture,
    case: &Case,
    output: &Buffer,
    qk: AttentionQk,
) -> Vec<u16> {
    let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
    fx.kernels
        .encode_online_attention_qk(&mut enc, case.request(output), qk)
        .unwrap_or_else(|e| panic!("prefill ({qk:?}) on {}: {e}", case.label()));
    enc.commit_and_wait().expect("wait");
    let n = case.queries_count * case.shape.q_heads * case.shape.head_dim;
    read_u16(output, n)
}

/// Worst absolute difference in f32 plus a NaN count, over the stored fp16.
fn worst(a: &[u16], b: &[u16]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0f32;
    let mut nans = 0usize;
    for (x, y) in a.iter().zip(b) {
        let fx = f16::from_bits(*x).to_f32();
        let fy = f16::from_bits(*y).to_f32();
        if fx.is_nan() || fy.is_nan() {
            nans += 1;
            continue;
        }
        max = max.max((fx - fy).abs());
    }
    (max, nans)
}

/// The core claim: for every shape, key count and masking regime, the MMA QK
/// arm stays within tolerance of the scalar arm, with no NaN.
#[test]
fn attention_mma_matches_scalar() {
    let Some(fx) = Fixture::new() else {
        // No Metal device: the repo-wide convention is to no-op rather than
        // claim a pass about hardware that is not there.
        return;
    };
    // Set from the measured worst case (recorded in notes_T1_T2.md); a real
    // QK bug measured via sabotage lands at O(1e-1) .. O(1), ~100x above this.
    const TOL: f32 = 5e-3;
    let mut worst_seen = 0.0f32;
    let mut worst_case = String::new();
    let mut failures: Vec<String> = Vec::new();
    for (shape, key_count, queries_count, query_abs, scale) in cases() {
        let case = fx.case(shape, key_count, queries_count, query_abs, scale);
        let scalar = run(&fx, &case, &case.output_a, AttentionQk::Scalar);
        let mma = run(&fx, &case, &case.output_b, AttentionQk::Mma);
        let (diff, nans) = worst(&scalar, &mma);
        println!("{} -> diff={diff:.6} nans={nans}", case.label());
        if diff > worst_seen {
            worst_seen = diff;
            worst_case = case.label();
        }
        if nans > 0 || diff > TOL {
            failures.push(format!(
                "{}: diff={diff:.6} nans={nans}",
                case.label()
            ));
        }
    }
    println!(
        "worst |mma - scalar| fp16 abs diff: {worst_seen:.6} at {worst_case} (tol {TOL})"
    );
    assert!(
        failures.is_empty(),
        "MMA QK diverges from scalar on {} case(s); only the QK scoring differs \
         between the two kernels, so suspect the simdgroup fragment layout \
         (C[key][row] lane map), the extraction shuffle source lane, the \
         transposed Q load, the tail staging, or the post-extraction scale: {:#?}",
        failures.len(),
        failures
    );
}

/// T3 (`spec/prefill-24k-90tps`): the flash body replaces the scalar QK *and*
/// the scalar PV for the full-attention shape, so it diverges from scalar in two
/// more places than the MMA arm does -- the softmax is fed an 8x8 MMA tile and
/// the PV fold is a second `simdgroup_multiply_accumulate` with the weights
/// stored as fp16 in `wq`. That fp16 P is a real, deliberate precision change,
/// so this assertion is the thing that decides whether the rewrite is usable.
#[test]
fn attention_flash_matches_scalar_on_the_full_shape() {
    let Some(fx) = Fixture::new() else {
        return;
    };
    if !fx.kernels.flash_attention_available() {
        panic!(
            "the T3 flash pipeline (online_attention_flash_512_8) did not build; a \
             Metal compile error is silent here because a failed library makes \
             every test in this file no-op"
        );
    }
    const TOL: f32 = 5e-3;
    let mut worst_seen = 0.0f32;
    let mut worst_case = String::new();
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for (shape, key_count, queries_count, query_abs, scale) in cases() {
        if !(shape.head_dim == 512 && shape.q_heads / shape.kv_heads == 8) {
            continue;
        }
        checked += 1;
        let case = fx.case(shape, key_count, queries_count, query_abs, scale);
        let scalar = run(&fx, &case, &case.output_a, AttentionQk::Scalar);
        let flash = run(&fx, &case, &case.output_b, AttentionQk::Flash);
        let (diff, nans) = worst(&scalar, &flash);
        println!("flash {} -> diff={diff:.6} nans={nans}", case.label());
        if diff > worst_seen {
            worst_seen = diff;
            worst_case = case.label();
        }
        if nans > 0 || diff > TOL {
            failures.push(format!("{}: diff={diff:.6} nans={nans}", case.label()));
        }
    }
    assert!(checked > 0, "no full-attention cases exercised the flash arm");
    println!(
        "worst |flash - scalar| fp16 abs diff: {worst_seen:.6} at {worst_case} \
         over {checked} cases (tol {TOL})"
    );
    assert!(
        failures.is_empty(),
        "the flash prefill body diverges from scalar on {} case(s). Suspect the \
         64-key m-tile ownership (`sg <= partial_sub`), the sth [key][head] \
         publish lane map, the 2-keys-per-lane softmax reduction, the wq \
         [head][key] layout (ld must be 64), the PV n-tile ownership \
         (`sg + j * NSG`), the `tail` K/V straddle staging, or the per-head msb \
         rescale row index ({:#?})",
        failures.len(),
        failures
    );
}

/// The production default is **shape-aware** (T2/T3): `encode_online_attention`
/// must dispatch the flash arm on the full-attention shape (`head_dim == 512`,
/// `groups == 8`) and the scalar arm on the sliding shape (`groups == 2`, NSG 2),
/// bitwise, with no environment override set.
///
/// This replaces the T1 guard that pinned Scalar globally. The T1 partitioned
/// MMA loses on NSG 2 (1.6x slower) and wins on NSG 8 (1.2x faster), measured in
/// `attention_qk_mma_bench`, so the selection is per shape. Both arms are the
/// same kernels `attention_mma_matches_scalar` compares, so comparing the
/// default's output to the explicit arm is a real dispatch-identity check (a
/// default that silently stayed scalar on FULL would fail the FULL case).
#[test]
fn the_default_prefill_path_is_shape_aware() {
    let Some(fx) = Fixture::new() else {
        return;
    };
    let full_arm = if fx.kernels.flash_attention_available() {
        AttentionQk::Flash
    } else {
        AttentionQk::Mma
    };
    for (shape, expect_specialised) in [(FULL, full_arm), (SLIDING, AttentionQk::Scalar)] {
        let case = fx.case(shape, 65, 2, 65, 1.0);
        let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
        fx.kernels
            .encode_online_attention(&mut enc, case.request(&case.output_a))
            .unwrap_or_else(|e| panic!("prefill default on {}: {e}", shape.name));
        enc.commit_and_wait().expect("wait");
        let n = case.queries_count * case.shape.q_heads * case.shape.head_dim;
        let default_out = read_u16(&case.output_a, n);
        let expected_qk = expect_specialised;
        let expected = run(&fx, &case, &case.output_b, expected_qk);
        assert_eq!(
            default_out, expected,
            "encode_online_attention on {} must dispatch the {expected_qk:?} arm \
             by default; it did not. Reconcile attention_single_pass_bit_exact \
             and the T3 measurements before changing the shape rule",
            shape.name
        );
    }
}
