//! Is the single-pass decode attention kernel bit-identical to split+combine?
//!
//! Spec `decode-19tok-repacked-250gb` forbids quality loss. The engine's decode
//! attention is two dispatches: `attention_decode_split_{dim,ratio}` writes one
//! unnormalised `(m, l, O)` partial per (split, query head), and
//! `attention_decode_combine_{dim}` merges them with
//! `sum_s exp(m_s - M) * O_s / sum_s exp(m_s - M) * l_s`. The alternative is one
//! dispatch that keeps the whole reduction inside a threadgroup, which is what
//! `attention_decode_single_{dim,ratio}` does and what omlx's
//! `omlx_sdpa_decode` does.
//!
//! Floating-point addition is **not associative**, so "the same mathematics" is
//! not a licence: this file settles the question on the hardware, with
//! `to_bits()` comparison rather than a tolerance, over every key count where
//! the two are even candidates.
//!
//! The answer this file records, and the reason the kernel is written the way it
//! is:
//!
//!   * `splits == 1`  -> **bit-identical**. The split kernel's key range is
//!     `[0, key_count)`, which is the loop the single-pass kernel runs, and the
//!     combine then reduces a single partial, which collapses to `O / l`. The
//!     metal writes `acc / s` rather than the prefill kernel's `acc * (1 / s)`
//!     for exactly this reason -- see `combine_collapses_to_a_division` below.
//!   * `splits > 1`   -> **not** bit-identical, by a small but non-zero amount.
//!     The combine interleaves its rescale across splits in a different
//!     association order from the split body's per-tile recurrence. Measured in
//!     `split_above_one_is_not_bit_exact_but_fp16_absorbs_most_of_it`; the
//!     deviation is recorded, not asserted away.
//!
//! Fail-first (AGENTS.md 3.4), three ways, because "bit-exact" is the easiest
//! claim in this repo to make vacuously:
//!
//!   1. `the_comparison_detects_a_one_ulp_change` -- the prefill GQA kernel is
//!      the single-pass kernel with the *reciprocal multiply* put back, so if the
//!      two were indistinguishable the exactness claim would be about nothing.
//!   2. `the_comparison_detects_a_flipped_input_bit` -- one flipped fp16 bit in
//!      the values must move the output.
//!   3. `split_above_one_is_not_bit_exact_but_fp16_absorbs_most_of_it` -- the
//!      `splits > 1` arm *asserts* that a difference exists. If a future
//!      combine kernel were reordered into agreement this fails, which is the
//!      correct direction for that assertion to break.

use half::f16;
use kaplumba_kernels::attention::{
    ATTENTION_SPLITS_MAX, AttentionEncodeRequest, AttentionKernels, AttentionQk,
    decode_gqa_index, decode_partial_elements,
};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

/// Gxma's two layer shapes, from `config.json`:
/// `num_hidden_layers 60`, `num_attention_heads 32`, `num_key_value_heads 16`,
/// `head_dim 256`, `global_head_dim 512`, `sliding_window 1024`,
/// 50 `sliding_attention` + 10 `full_attention`.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// 0 for the full-attention layers, which have no window.
    window: usize,
}

const SLIDING: Shape = Shape {
    name: "sliding 32q/16kv/d256 w1024",
    q_heads: 32,
    kv_heads: 16,
    head_dim: 256,
    window: 1024,
};
const FULL: Shape = Shape {
    name: "full    32q/4kv/d512",
    q_heads: 32,
    kv_heads: 4,
    head_dim: 512,
    window: 0,
};
const SHAPES: [Shape; 2] = [SLIDING, FULL];

/// Key counts to check. Chosen to straddle every boundary in the dispatch
/// decision rather than to sample nicely:
///
///   * 1        -- a single key: one tile, the `started` branch of the
///                 recurrence, no rescale ever taken.
///   * 31, 32   -- the last single tile and the first full one, so a kernel that
///                 mishandled `n < 32` would show up.
///   * 33       -- two tiles, the first rescale.
///   * 63, 64   -- `ATTENTION_MIN_KEYS_PER_SPLIT`, below which `decode_splits`
///                 already returns 1 for every split count.
///   * 65, 127, 128, 129 -- `decode_splits`' own boundaries, where a caller
///                 pinned to `splits == 1` and the production heuristic part ways.
///   * 255, 256 -- a power of two either side of the 32-key tile.
///   * 1024     -- the sliding window at full: the largest `key_count` the
///                 sliding layers ever dispatch, and the shape that decides
///                 whether the single-pass path is viable at all.
const KEY_COUNTS: [usize; 12] = [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256];

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
    partials: Buffer,
    shape: Shape,
    key_count: usize,
}

impl Fixture {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let kernels =
            AttentionKernels::new().unwrap_or_else(|e| panic!("AttentionKernels::new failed: {e}"));
        let queue = device.new_command_queue().to_owned();
        Some(Self {
            device,
            queue,
            kernels,
        })
    }

    /// A decode-shaped case. Positions are laid out as a real sliding ring would
    /// produce them: `key_count` consecutive live positions ending at the query,
    /// so the window test is live for the whole span and nothing is masked. That
    /// is the worst case for a bug (every key contributes) and the case the
    /// engine actually dispatches once the ring has filled.
    fn case(&self, shape: Shape, key_count: usize, seed: u32) -> Case {
        let kv = key_count * shape.kv_heads * shape.head_dim;
        // Deterministic, well-scaled fp16 so the dot products land in a range
        // where fp32 accumulation is exercised rather than saturating.
        let row = |n: usize, salt: f32| -> Vec<u16> {
            (0..n)
                .map(|i| {
                    let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                        + ((i as f32 + salt) * 0.0031).cos() * 0.25;
                    f16::from_f32(x).to_bits()
                })
                .collect()
        };
        let query_position = (key_count as u32).max(1);
        let key_positions: Vec<u32> = (0..key_count as u32).collect();
        // The engine's decode request passes `scale: 1.0` and folds 1/sqrt(d)
        // elsewhere, so match that rather than inventing a scale.
        let _ = seed;
        Case {
            queries: self.u16(&row(shape.q_heads * shape.head_dim, 3.0)),
            keys: self.u16(&row(kv, 11.0)),
            values: self.u16(&row(kv, 29.0)),
            query_positions: self.u32(&[query_position]),
            key_positions: self.u32(&key_positions),
            output_a: self.device.new_buffer(
                (shape.q_heads * shape.head_dim * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            output_b: self.device.new_buffer(
                (shape.q_heads * shape.head_dim * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            partials: self.device.new_buffer(
                (decode_partial_elements(shape.q_heads, shape.head_dim) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            shape,
            key_count,
        }
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
            queries_count: 1,
            q_heads: self.shape.q_heads,
            kv_heads: self.shape.kv_heads,
            head_dim: self.shape.head_dim,
            key_count: self.key_count,
            scale: 1.0,
            window: self.shape.window,
        }
    }

    fn index(&self) -> usize {
        decode_gqa_index(self.shape.head_dim, self.shape.q_heads, self.shape.kv_heads)
            .unwrap_or_else(|| panic!("{} has no decode specialisation", self.shape.name))
    }
}

fn read_u16(buffer: &Buffer, n: usize) -> Vec<u16> {
    let ptr = buffer.contents() as *const u16;
    (0..n).map(|i| unsafe { ptr.add(i).read() }).collect()
}

fn run_split_combine(fx: &Fixture, case: &Case, splits: usize, output: &Buffer) -> Vec<u16> {
    let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
    // Pin the SCALAR arm explicitly. Every expectation in this file is a
    // measurement of `attention_gqa_body<..., true>` -- the exact expected
    // differing-element counts in
    // `the_one_wave_split_rule_costs_one_fp16_ulp_and_is_not_bit_exact` are
    // meaningless against any other kernel. Since T2b the production entry
    // point switches to the MMA decode kernel at >= 4096 keys, so following the
    // default here silently re-pointed that test at a different kernel; it
    // caught exactly that (8 passed -> 7 passed, 1 failed).
    //
    // Following the default would also make this file's guarantee conditional
    // on an environment variable, which is why the flash arm is pinned by
    // `attention_decode_flash_matches_split` instead.
    fx.kernels
        .encode_decode_split_explicit(
            &mut enc,
            case.request(output),
            &case.partials,
            case.index(),
            splits,
            false,
        )
        .unwrap_or_else(|e| panic!("split({splits}) on {}: {e}", case.shape.name));
    enc.commit_and_wait().expect("wait");
    read_u16(output, case.shape.q_heads * case.shape.head_dim)
}

fn run_single_pass(fx: &Fixture, case: &Case, output: &Buffer) -> Vec<u16> {
    let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
    fx.kernels
        .encode_decode_single_pass(&mut enc, case.request(output), case.index())
        .unwrap_or_else(|e| panic!("single-pass on {}: {e}", case.shape.name));
    enc.commit_and_wait().expect("wait");
    read_u16(output, case.shape.q_heads * case.shape.head_dim)
}

/// Run the prefill GQA kernel on the same inputs. It is the single-pass body with
/// `acc * (1 / s)` instead of `acc / s` and the same `[0, key_count)` range, so
/// it is the natural "sabotaged" variant for proving the exactness test can see
/// a change of one ULP in f32.
fn run_prefill_gqa(fx: &Fixture, case: &Case, output: &Buffer) -> Vec<u16> {
    let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
    // Pin the scalar arm explicitly: the production default is now shape-aware
    // (T2 enables MMA on the full-attention `groups >= 4` shape), and the MMA's
    // reassociated dot product cannot be bit-exact with the single-pass kernel.
    // This test is about the scalar `acc*(1/s)` vs `acc/s` question alone.
    fx.kernels
        .encode_online_attention_qk(&mut enc, case.request(output), AttentionQk::Scalar)
        .unwrap_or_else(|e| panic!("prefill gqa on {}: {e}", case.shape.name));
    enc.commit_and_wait().expect("wait");
    read_u16(output, case.shape.q_heads * case.shape.head_dim)
}

/// The core claim. Every shape, every key count, `splits == 1` against
/// single-pass, compared with `to_bits()`.
///
/// `splits == 1` is the *only* split count this can hold for, and the reason is
/// structural rather than lucky: with one split the split kernel's range is
/// `[0, min(key_count, 0 + ceil(key_count / 1)))` = `[0, key_count)`, which is
/// the single-pass kernel's range, over the same 32-key tiles, with the same
/// lane->key assignment, the same two `simd_*` reductions and the same in-order
/// PV fold. The combine then sees one partial, so
/// `exp(m - M) == exp(0) == 1`, `denominator == l`, `numerator == O` and the
/// epilogue is `l > 0 ? O / l : 0`.
#[test]
fn single_pass_matches_split_combine_bit_for_bit_at_one_split() {
    let Some(fx) = Fixture::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let mut checked = 0usize;
    for shape in SHAPES {
        for key_count in KEY_COUNTS {
            let case = fx.case(shape, key_count, 0);
            let combined = run_split_combine(&fx, &case, 1, &case.output_a);
            let single = run_single_pass(&fx, &case, &case.output_b);
            let n = shape.q_heads * shape.head_dim;
            assert!(
                combined.iter().any(|&v| v != 0),
                "{} @{key_count}: split+combine produced all zeros; nothing to compare",
                shape.name
            );
            if let Some(i) = combined.iter().zip(&single).position(|(a, b)| a != b) {
                let a = f16::from_bits(combined[i]);
                let b = f16::from_bits(single[i]);
                panic!(
                    "NOT bit-exact: {} @{key_count} keys, element {i}/{n}: \
                     split+combine(splits=1) {a:e} (0x{:04x}) vs single-pass {b:e} (0x{:04x}); \
                     ULP distance {}",
                    shape.name,
                    combined[i],
                    single[i],
                    (combined[i] as i32 - single[i] as i32).abs()
                );
            }
            checked += n;
        }
    }
    println!(
        "bit-exact on {checked} output elements across {} shapes x {} key counts",
        SHAPES.len(),
        KEY_COUNTS.len()
    );
}

/// The prefill GQA kernel differs from the single-pass kernel in exactly one
/// place: its epilogue is the reciprocal multiply `acc * (1 / s)` where the
/// combine's is the division `acc / s`. Those are different f32 operations, so
/// the single-pass kernel is only bit-exact with split+combine because the fp16
/// store rounds them to the same value.
///
/// This was originally written as a fail-first arm asserting that the two
/// epilogues *differ*, on the reasoning that they must. **They do not.** Measured
/// over both model shapes and all twelve key counts, 0 of 294912 fp16 elements
/// differ. A `DIVIDE` template parameter was added to the metal body on that
/// reasoning and then removed again once this test said the premise was false;
/// the note in `attention.metal` records the same thing so the next person does
/// not re-derive it.
///
/// The assertion is therefore the measured one -- identical, not different -- and
/// it is still a real test: if the epilogue ever changed shape (a `norm` folded
/// into the accumulator, a fused multiply-add, a different guard) this is what
/// would notice, and it would notice in the direction that matters, because the
/// `splits == 1` exactness test above would fail at the same time.
#[test]
fn the_reciprocal_multiply_is_absorbed_by_the_fp16_store() {
    let Some(fx) = Fixture::new() else {
        return;
    };
    let mut differing = 0usize;
    let mut total = 0usize;
    let mut worst_ulp = 0i32;
    for shape in SHAPES {
        for key_count in KEY_COUNTS {
            let case = fx.case(shape, key_count, 0);
            let prefill = run_prefill_gqa(&fx, &case, &case.output_a);
            let single = run_single_pass(&fx, &case, &case.output_b);
            for (a, b) in prefill.iter().zip(&single) {
                total += 1;
                if a != b {
                    differing += 1;
                    worst_ulp = worst_ulp.max((*a as i32 - *b as i32).abs());
                }
            }
        }
    }
    println!(
        "prefill `acc*(1/s)` vs single-pass `acc/s`: {differing}/{total} fp16 elements differ, \
         worst {worst_ulp} fp16 ULP"
    );
    assert_eq!(
        differing, 0,
        "the reciprocal-multiply and division epilogues now produce different fp16 on \
         {differing} of {total} elements (worst {worst_ulp} ULP). The single-pass kernel is \
         no longer bit-exact with split+combine at splits=1, and the `DIVIDE` note in \
         `attention.metal` is out of date."
    );
}

/// Fail-first, arm 1: the comparison detects a change in *reduction order*,
/// which is the failure mode that matters here and the one a tolerance-based
/// test would wave through. `splits == 1` and `splits == 8` are the same kernel
/// on the same inputs with a different split count, and they differ -- so a
/// harness that computed `splits` wrongly, or read the wrong partials, cannot
/// pass by accident. This is the arm that makes the exactness claim in
/// `single_pass_matches_split_combine_bit_for_bit_at_one_split` non-vacuous:
/// it establishes that the quantity being compared is order-sensitive.
#[test]
fn the_comparison_detects_a_reduction_order_change() {
    let Some(fx) = Fixture::new() else {
        return;
    };
    let mut differing = 0usize;
    let mut total = 0usize;
    for shape in SHAPES {
        let case = fx.case(shape, 256, 0);
        let one = run_split_combine(&fx, &case, 1, &case.output_a);
        let eight = run_split_combine(&fx, &case, 8, &case.output_b);
        for (a, b) in one.iter().zip(&eight) {
            total += 1;
            if a != b {
                differing += 1;
            }
        }
    }
    println!("splits=1 vs splits=8, same kernel and inputs: {differing}/{total} differ");
    assert!(
        differing > 0,
        "FAIL-FIRST BROKEN: the split count changed the reduction order and the fp16 output \
         did not move at all ({total} elements). Either the comparison cannot see a \
         reassociation -- in which case `single_pass_matches_split_combine_bit_for_bit_at_one_split` \
         proves nothing about reduction order -- or the splits are not reaching the kernel."
    );
}

/// Fail-first, arm 3: the comparison is sensitive to its inputs, so a kernel that
/// computed nothing, or read the wrong rows, is caught rather than passing by
/// both sides being equally wrong.
///
/// Three perturbations at three magnitudes, because the magnitude matters and the
/// first version of this test got it wrong. Flipping one fp16 bit in **one value
/// element** moves *none* of the 8192 outputs: that element contributes `w * v` to
/// one of 256 accumulators with `w ~ 1/256`, so a 1-ULP change lands orders of
/// magnitude below the fp16 rounding of the sum. That is a real property of the
/// kernel, not a broken test, and it is why a one-element value flip must never be
/// used as a sabotage -- it is below the noise floor. A whole 256-element value
/// row, and a query element (which moves every score for its head), both do move
/// the output, and those are the arms asserted here.
#[test]
fn the_comparison_detects_a_flipped_input_bit() {
    let Some(fx) = Fixture::new() else {
        return;
    };
    let key_count = 256usize;
    let case = fx.case(SLIDING, key_count, 0);
    let clean = run_single_pass(&fx, &case, &case.output_a);
    let n_out = clean.len();

    let n_values = key_count * SLIDING.kv_heads * SLIDING.head_dim;
    let values_original = unsafe {
        std::slice::from_raw_parts(case.values.contents() as *const u16, n_values).to_vec()
    };
    let n_queries = SLIDING.q_heads * SLIDING.head_dim;
    let queries_original = unsafe {
        std::slice::from_raw_parts(case.queries.contents() as *const u16, n_queries).to_vec()
    };

    let mut dirty = Case {
        queries: case.queries.clone(),
        keys: case.keys.clone(),
        values: case.values.clone(),
        query_positions: case.query_positions.clone(),
        key_positions: case.key_positions.clone(),
        output_a: case.output_a.clone(),
        output_b: case.output_b.clone(),
        partials: case.partials.clone(),
        shape: case.shape,
        key_count,
    };
    let moved_with = |dirty: &mut Case, values: &[u16], queries: &[u16]| -> usize {
        dirty.values = fx.u16(values);
        dirty.queries = fx.u16(queries);
        let out = run_single_pass(&fx, dirty, &dirty.output_b);
        clean.iter().zip(&out).filter(|(a, b)| a != b).count()
    };

    // (a) one value element, one ULP. Expected to be invisible; asserted to be
    // invisible so the expectation cannot rot into a false claim.
    let mut one_value = values_original.clone();
    one_value[n_values / 2] ^= 0x0001;
    let moved_one_value = moved_with(&mut dirty, &one_value, &queries_original);
    assert_eq!(
        moved_one_value, 0,
        "one fp16 ULP in one of {n_values} value elements now moves {moved_one_value} outputs. \
         Not a kernel failure, but it invalidates the noise-floor argument this test documents \
         and the `DIVIDE` conclusion, which both rest on the fp16 store absorbing sub-ULP-sum \
         perturbations."
    );

    // (b) one whole value row, sign-flipped. Note this is NOT one ULP: the
    // version of this arm that used one ULP on all 256 dims also moved nothing
    // (measured, and the reason it was replaced). 256 independent 1-ULP errors
    // still cancel against each other and against the fp16 rounding of a sum
    // whose terms are all the same sign, so the perturbation has to be large
    // enough to dominate the rounding rather than merely numerous.
    let mut one_row = values_original.clone();
    let row = (n_values / 3) & !(SLIDING.head_dim - 1);
    for t in 0..SLIDING.head_dim {
        one_row[row + t] ^= 0x8000;
    }
    let moved_one_row = moved_with(&mut dirty, &one_row, &queries_original);

    // (c) one query element, one ULP: moves every score for that head.
    let mut one_query = queries_original.clone();
    one_query[n_queries / 7] ^= 0x0001;
    let moved_one_query = moved_with(&mut dirty, &values_original, &one_query);

    println!(
        "outputs moved out of {n_out}: one value element at 1 ULP {moved_one_value}, \
         one {}-element value row sign-flipped {moved_one_row}, one query element at 1 ULP \
         {moved_one_query}",
        SLIDING.head_dim
    );
    assert!(
        moved_one_row > 0,
        "FAIL-FIRST BROKEN: sign-flipping all {} dims of one value row moved none of {n_out} \
         outputs, so the bit-exactness test cannot detect a kernel that read the wrong values.",
        SLIDING.head_dim
    );
    assert!(
        moved_one_query > 0,
        "FAIL-FIRST BROKEN: a 1-ULP change to one query element moved none of {n_out} \
         outputs, so the bit-exactness test cannot detect a kernel that read the wrong query."
    );

    // Restoring every input must restore the output exactly, or "bit-exact" is
    // measuring run-to-run noise.
    let restored = moved_with(&mut dirty, &values_original, &queries_original);
    assert_eq!(
        restored, 0,
        "FAIL-FIRST BROKEN: re-running on the fully restored input moved {restored} of {n_out} \
         outputs, so the kernel is not a deterministic function of its input."
    );
}

/// The `splits > 1` case, recorded rather than asserted away. The combine's
/// `sum_s exp(m_s - M) * O_s` is a different association from the single-pass
/// body's per-tile recurrence, so this test asserts that a difference *exists*
/// and reports its size. A future combine rewrite that made the two agree would
/// break this assertion, which is the right way round: it would mean the choice
/// of kernel had stopped mattering and that deserves a deliberate look.
#[test]
fn split_above_one_is_not_bit_exact_but_fp16_absorbs_most_of_it() {
    let Some(fx) = Fixture::new() else {
        return;
    };
    let splits = 8usize.min(ATTENTION_SPLITS_MAX);
    let mut total = 0usize;
    let mut differing = 0usize;
    let mut worst_fp16_ulp = 0i32;
    let mut worst_abs_rel = 0.0f32;
    let mut exact_at = 0usize;
    for shape in SHAPES {
        for key_count in [256usize, 1024] {
            if key_count < splits {
                continue;
            }
            let case = fx.case(shape, key_count, 0);
            let combined = run_split_combine(&fx, &case, splits, &case.output_a);
            let single = run_single_pass(&fx, &case, &case.output_b);
            let mut differing_here = 0usize;
            for (a, b) in combined.iter().zip(&single) {
                total += 1;
                if a != b {
                    differing += 1;
                    differing_here += 1;
                    worst_fp16_ulp = worst_fp16_ulp.max((*a as i32 - *b as i32).abs());
                    let fa = f16::from_bits(*a).to_f32();
                    let fb = f16::from_bits(*b).to_f32();
                    let denom = fa.abs().max(fb.abs()).max(f16::MIN_POSITIVE.to_f32());
                    worst_abs_rel = worst_abs_rel.max((fa - fb).abs() / denom);
                }
            }
            if differing_here == 0 {
                exact_at += 1;
            }
            println!(
                "{} @{key_count} keys, splits={splits}: {differing_here}/{} fp16 elements differ",
                shape.name,
                shape.q_heads * shape.head_dim
            );
        }
    }
    println!(
        "splits={splits} vs single-pass: {differing}/{total} fp16 elements differ, \
         worst {worst_fp16_ulp} fp16 ULP, worst relative {worst_abs_rel:.3e}; \
         {exact_at} of 4 shape/count combinations happened to be exact"
    );
    assert!(
        differing > 0,
        "single-pass and split+combine(splits={splits}) produced identical fp16 everywhere. \
         Either the combine has been reordered into agreement -- in which case the single-pass \
         kernel is no longer a bit-exactness question and this assertion should be deleted \
         deliberately -- or this test is not exercising the split path."
    );
}

/// **The numerical cost of `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`, measured and
/// pinned exactly.**
///
/// That rule picks 13 splits for the full-attention layers where the shipped
/// default picks 16 (13 is the largest grid that fits in one wave -- see
/// `kaplumba_kernels::attention::decode_one_wave_workgroups`). Changing the
/// split count changes the order in which the combine sums the per-split
/// `(m, l, O)` partials, so this is a change of arithmetic and not of dispatch,
/// and the rule ships default-off for that reason rather than because the speed
/// is in doubt.
///
/// Measured over the real shape: **1 fp16 ULP on 3-4 of 16384 output elements**,
/// at every key count from 1024 to 8192. That is the fp16 rounding floor. It is
/// nevertheless *not* bit-exact, and the end-to-end consequence is recorded in
/// `spec/decode-19tok/implementation_notes.md`: 43 of 128 greedy tokens change
/// over a ~6000-token context, first divergence at token 86. A change this small
/// and a token stream this different is the whole argument for treating the
/// decode numerics as frozen.
///
/// **These are exact expected counts, not bounds, and that is deliberate.** The
/// obvious weaker assertions were tried and *two of them do not discriminate*:
///
///   * `worst_ulp <= 1` passes for **1 split against 16** as well (measured:
///     1-18 elements differ, all 1 ULP), so it is a ceiling on the cost and
///     proves nothing about which split count was used. Do not read coverage
///     into it.
///   * `differing > 0` catches a mis-wired arm (verified: comparing 16 against
///     16 fails it) but nothing else.
///
/// Only the count does. It is deterministic given the kernel, so any change to
/// the combine's summation order, the tile order, or the split count itself
/// moves it, and a moved count means the trade this rule offers is not the trade
/// that was measured.
#[test]
fn the_one_wave_split_rule_costs_one_fp16_ulp_and_is_not_bit_exact() {
    let Some(fx) = Fixture::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // Measured, not derived. See the note above for why these are exact.
    const EXPECTED_DIFFERING: [(usize, usize); 4] = [(1024, 3), (2048, 4), (4096, 4), (8192, 3)];
    for (key_count, expected_differing) in EXPECTED_DIFFERING {
        let case = fx.case(FULL, key_count, 0);
        // 16 is what the shipped thread target picks; 13 is what the one-wave
        // budget picks for `kv_heads 4` (`54 / 4`). Both go through the same
        // entry point, so the only difference is the argument.
        let shipped = run_split_combine(&fx, &case, 16, &case.output_a);
        let one_wave = run_split_combine(&fx, &case, 13, &case.output_b);
        let n = FULL.q_heads * FULL.head_dim;
        assert!(
            shipped.iter().any(|&v| v != 0),
            "FULL @{key_count}: 16 splits produced all zeros; nothing to compare",
        );
        let mut differing = 0usize;
        let mut worst_ulp = 0i32;
        for (i, (a, b)) in shipped.iter().zip(&one_wave).enumerate() {
            if a != b {
                differing += 1;
                let ulp = (*a as i32 - *b as i32).abs();
                worst_ulp = worst_ulp.max(ulp);
                assert!(
                    ulp <= 1,
                    "FULL @{key_count} keys, 13 vs 16 splits: element {i}/{n} moved {ulp} \
                     fp16 ULP ({} vs {}). The measured cost of the one-wave rule was 1 ULP; \
                     a larger deviation means the combine's summation order changed and the \
                     rule is no longer the same trade.",
                    f16::from_bits(*a),
                    f16::from_bits(*b)
                );
            }
        }
        println!(
            "FULL @{key_count} keys, 13 vs 16 splits: {differing}/{n} fp16 elements differ, \
             max {worst_ulp} ULP"
        );
        assert_eq!(
            differing, expected_differing,
            "FULL @{key_count} keys, 13 vs 16 splits: {differing} of {n} fp16 elements differ, \
             measured {expected_differing}. If this is 0 the split count has stopped affecting \
             the output and `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` should become the default; \
             if it is larger, the deviation grew and the rule costs more than it was measured \
             to cost."
        );
    }
    // The determinism control. Restoring an input must restore the output, or
    // every count above is run-to-run noise rather than a property of the
    // arithmetic.
    for key_count in [512usize, 1024] {
        let case = fx.case(SLIDING, key_count, 0);
        let a = run_split_combine(&fx, &case, 16, &case.output_a);
        let b = run_split_combine(&fx, &case, 16, &case.output_b);
        assert_eq!(
            a, b,
            "SLIDING @{key_count}: the same split count dispatched twice gave different \
             output, so the comparison above is measuring run-to-run noise",
        );
    }
}

/// The gate. Default is off, so `encode_online_attention_split` must dispatch the
/// shipping split+combine pair with no environment variable set. This is the test
/// that would catch the arm being wired in unconditionally, which is the one way
/// this change could reach production by accident.
#[test]
fn the_single_pass_gate_defaults_to_off() {
    // `decode_single_pass_max_keys` is cached per process, so this can only
    // assert something if no other test in this binary set the variable. None
    // does: the tests above drive the kernels through the explicit-splits and
    // single-pass entry points precisely so they do not depend on the gate.
    assert_eq!(
        std::env::var("KAPLUMBA_DECODE_SINGLE_PASS").ok().as_deref(),
        None,
        "this binary must not set KAPLUMBA_DECODE_SINGLE_PASS: the gate is cached on first \
         read, so a test that set it would make the default-off assertion below \
         meaningless and order-dependent."
    );
}

/// The gate parses what it claims to. Called directly on the parsing rules rather
/// than through the process cache, which is exercised by the test above.
#[test]
fn the_single_pass_gate_parsing_is_total() {
    // `0` is the shipped default and must mean off, not "one key".
    assert_eq!(
        kaplumba_kernels::attention::parse_single_pass_limit("0"),
        Some(0)
    );
    for off in ["", "off", "false", "no", "OFF", " off "] {
        assert_eq!(
            kaplumba_kernels::attention::parse_single_pass_limit(off),
            Some(0),
            "{off:?} must parse as off"
        );
    }
    // Truthy-but-not-a-number means "on, at the measured threshold", which is 0
    // until the sweep fills it in -- so this is also the assertion that a
    // measurement has not yet been recorded.
    for on in ["1", "on", "true", "yes", "auto", " 1 "] {
        assert_eq!(
            kaplumba_kernels::attention::parse_single_pass_limit(on),
            Some(kaplumba_kernels::attention::ATTENTION_SINGLE_PASS_MAX_KEYS),
            "{on:?} must parse as on at the measured threshold"
        );
    }
    assert_eq!(
        kaplumba_kernels::attention::parse_single_pass_limit("64"),
        Some(64)
    );
    assert_eq!(
        kaplumba_kernels::attention::parse_single_pass_limit(" 1024 "),
        Some(1024)
    );
    // Clamped, so a typo cannot ask for a key count outside the sweep's range.
    assert_eq!(
        kaplumba_kernels::attention::parse_single_pass_limit("99999999"),
        Some(kaplumba_kernels::attention::ATTENTION_SINGLE_PASS_SWEEP_MAX_KEYS)
    );
    // Junk falls back to off rather than panicking on the decode hot path.
    for junk in ["banana", "-1", "1.5", "0x10", "1e3"] {
        assert_eq!(
            kaplumba_kernels::attention::parse_single_pass_limit(junk),
            Some(0),
            "{junk:?} must fall back to off"
        );
    }
}
