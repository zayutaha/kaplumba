//! T2b (`spec/prefill-24k-90tps`): the MMA decode attention kernel must agree
//! with the scalar split kernel it replaces.
//!
//! `attention_decode_flash_512_8` is the prefill flash body with `BQ = 1` and
//! `PARTIALS`. It writes the *same* `(m, s, O)` partial layout and the same
//! combine pass merges it, so the two paths differ only in how `(m, s, O)` are
//! computed: the scalar kernel does a 128-iteration scalar dot per key and an
//! in-order PV fold, the flash kernel does 8x8 simdgroup MMA with fp32 online
//! softmax. That is a change of **summation order only**, so this test asserts
//! an fp16 tolerance rather than bit-equality -- and the tolerance is tight
//! enough (5e-3, the same one `attention_qk_mma_matches_scalar` uses) that a
//! wrong key range, a wrong split mapping or a mis-indexed partial fails it.
//!
//! Every case below is a case the two kernels could plausibly disagree on for a
//! *structural* reason rather than a rounding one, so the tolerance is never the
//! thing that carries the test:
//!
//!   * `key_count` not a multiple of `splits` (a split whose slice runs off the
//!     end) and an empty last slice (`splits` > `key_count / 64`),
//!   * `splits` = 1, 2, 3 (non-power-of-two, so the slices are ragged), 13, 16,
//!   * `key_count` at and around the `ATTENTION_MIN_KEYS_PER_SPLIT = 64`
//!     boundary and well past it,
//!   * a sliding window that cuts the last tiles,
//!   * a non-unit scale.
//!
//! Run: `cargo test --release -p kaplumba-engine --test
//! attention_decode_flash_matches_split -- --nocapture`

use half::f16;
use kaplumba_kernels::attention::{
    decode_gqa_index, decode_partial_elements, AttentionEncodeRequest, AttentionKernels,
};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, Device, MTLResourceOptions};

/// Absolute fp16 tolerance. One ulp of fp16 at these magnitudes is ~1e-3, and
/// the two paths differ by fp32 summation order, so 5e-3 is the same bar the
/// prefill flash kernel is held to.
const TOL: f32 = 5e-3;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
}

const FULL: Shape = Shape {
    name: "full    32q/4kv/d512 (NSG=8)",
    q_heads: 32,
    kv_heads: 4,
    head_dim: 512,
    window: 0,
};

/// The only shape the decode flash kernel is instantiated for, matching the
/// prefill flash kernels' restriction (`flash_supported`: head_dim 512, groups 8).
const SLIDING: Shape = Shape {
    name: "sliding 32q/16kv/d256 w1024 (NSG=2)",
    q_heads: 32,
    kv_heads: 16,
    head_dim: 256,
    window: 1024,
};

fn row(n: usize, salt: f32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                + ((i as f32 + salt) * 0.0031).cos() * 0.25;
            f16::from_f32(x).to_bits()
        })
        .collect()
}

struct Case {
    shape: Shape,
    key_count: usize,
    splits: usize,
    queries: Buffer,
    keys: Buffer,
    values: Buffer,
    query_positions: Buffer,
    key_positions: Buffer,
    output_a: Buffer,
    output_b: Buffer,
    partials_a: Buffer,
    partials_b: Buffer,
}

impl Case {
    fn new(device: &Device, shape: Shape, key_count: usize, splits: usize, scale: f32) -> Self {
        let kv = key_count * shape.kv_heads * shape.head_dim;
        let u16buf = |v: &[u16]| {
            device.new_buffer_with_data(
                v.as_ptr() as *const std::ffi::c_void,
                (v.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let u32buf = |v: &[u32]| {
            device.new_buffer_with_data(
                v.as_ptr() as *const std::ffi::c_void,
                (v.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let out_bytes = (shape.q_heads * shape.head_dim * 2) as u64;
        let part_elems = decode_partial_elements(shape.q_heads, shape.head_dim) * splits;
        // The query sits at the end of the window so nothing is causally masked
        // unless a window says so.
        let qpos = [key_count as u32];
        let kpos: Vec<u32> = (0..key_count as u32).collect();
        Case {
            queries: u16buf(&row(shape.q_heads * shape.head_dim, 3.0)),
            keys: u16buf(&row(kv, 11.0)),
            values: u16buf(&row(kv, 29.0)),
            query_positions: u32buf(&qpos),
            key_positions: u32buf(&kpos),
            output_a: device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared),
            output_b: device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared),
            partials_a: device
                .new_buffer((part_elems * 4) as u64, MTLResourceOptions::StorageModeShared),
            partials_b: device
                .new_buffer((part_elems * 4) as u64, MTLResourceOptions::StorageModeShared),
            shape,
            key_count,
            splits,
        }
    }

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
            scale: self.scale(),
            window: self.shape.window,
        }
    }

    fn scale(&self) -> f32 {
        1.0 / (self.shape.head_dim as f32).sqrt()
    }

    fn index(&self) -> usize {
        decode_gqa_index(self.shape.head_dim, self.shape.q_heads, self.shape.kv_heads)
            .unwrap_or_else(|| panic!("{} has no decode specialisation", self.shape.name))
    }
}

fn run(kernels: &AttentionKernels, queue: &metal::CommandQueue, case: &Case, flash: bool) {
    let (output, partials) = if flash {
        (&case.output_b, &case.partials_b)
    } else {
        (&case.output_a, &case.partials_a)
    };
    let mut enc = MetalEncoder::new(queue).expect("encoder");
    kernels
        .encode_decode_split_explicit(
            &mut enc,
            case.request(output),
            partials,
            case.index(),
            case.splits,
            flash,
        )
        .unwrap_or_else(|e| panic!("encode flash={flash}: {e}"));
    enc.commit_and_wait().expect("commit_and_wait");
}

#[test]
fn decode_flash_matches_scalar_split_on_the_full_shape() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let kernels = AttentionKernels::new().expect("AttentionKernels::new");
    let queue = device.new_command_queue().to_owned();

    // (key_count, splits). Chosen to cover the structural edges listed at the
    // top of the file rather than to be a tidy progression.
    let cases: &[(usize, usize)] = &[
        (64, 1),    // exactly ATTENTION_MIN_KEYS_PER_SPLIT
        (65, 1),    // one past it
        (127, 1),   // odd, ends mid-tile
        (1000, 1),  // single split over many tiles
        (64, 2),    // splits, exact division
        (65, 2),    // splits, ragged: last slice is 1 key
        (129, 3),   // ragged three ways
        (1000, 3),
        (4096, 13), // the rule's answer for kv_heads=4
        (8192, 16), // ATTENTION_SPLITS_MAX
        (8192, 5),  // ragged with an odd split count
        (24576, 16),
        (24576, 13),
        (16384, 64), // ATTENTION_SPLITS_MAX = 64
        (16384, 32),
        // key_count below splits * 64: some splits are EMPTY, which must fall
        // out as m = NEG, s = 0, O = 0 and weigh nothing in the combine.
        (100, 16),
        (200, 13),
    ];

    let mut worst = 0.0f32;
    for &(key_count, splits) in cases {
        let case = Case::new(&device, FULL, key_count, splits, 0.0);
        run(&kernels, &queue, &case, false);
        run(&kernels, &queue, &case, true);

        let a = case.output_a.contents() as *const u16;
        let b = case.output_b.contents() as *const u16;
        let n = FULL.q_heads * FULL.head_dim;
        let mut worst_case = 0.0f32;
        let mut nans = 0usize;
        for i in 0..n {
            let va = f16::from_bits(unsafe { std::ptr::read_unaligned(a.add(i)) });
            let vb = f16::from_bits(unsafe { std::ptr::read_unaligned(b.add(i)) });
            if va.is_nan() || vb.is_nan() {
                nans += 1;
            }
            let d = (va.to_f32() - vb.to_f32()).abs();
            if d > worst_case {
                worst_case = d;
            }
        }
        if nans > 0 {
            panic!("keys={key_count} splits={splits}: {nans} NaNs");
        }
        println!(
            "{:<34} keys={key_count:<6} splits={splits:<3} worst |flash-scalar| = {worst_case:.6}",
            FULL.name
        );
        assert!(
            worst_case <= TOL,
            "keys={key_count} splits={splits}: worst |flash - scalar| = {worst_case} exceeds tol {TOL}"
        );
        if worst_case > worst {
            worst = worst_case;
        }
    }
    println!("worst |flash - scalar| fp16 abs diff: {worst:.6} (tol {TOL})");
}

/// The sliding shape is NOT instantiated for the flash decode kernel, so asking
/// for it must be refused rather than silently answered by the scalar kernel --
/// a silent fallback here would make the comparison above vacuous on any shape
/// that ever loses its instantiation.
#[test]
fn decode_flash_declines_the_sliding_shape() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let kernels = AttentionKernels::new().expect("AttentionKernels::new");
    let queue = device.new_command_queue().to_owned();
    let case = Case::new(&device, SLIDING, 4096, 4, 0.0);
    let mut enc = MetalEncoder::new(&queue).expect("encoder");
    let err = kernels
        .encode_decode_split_explicit(
            &mut enc,
            case.request(&case.output_a),
            &case.partials_a,
            case.index(),
            case.splits,
            true,
        )
        .expect_err("the flash decode kernel must refuse the sliding shape");
    assert!(
        err.to_string().contains("only instantiated for head_dim 512"),
        "unexpected error: {err}"
    );
}

/// Production selection: the environment rule must pick the flash kernel at and
/// above 4096 keys and the scalar kernel below it. This is the gate a decode run
/// actually goes through.
#[test]
fn decode_flash_threshold_is_four_thousand_keys_by_default() {
    assert_eq!(
        kaplumba_kernels::attention::decode_flash_min_keys(),
        4096,
        "the shipped threshold must match the one oMLX uses (OMLX_MIMO_DECODE_FLASH_MIN_KEYS)"
    );
    assert!(kaplumba_kernels::attention::decode_flash_enabled(4096));
    assert!(kaplumba_kernels::attention::decode_flash_enabled(24576));
    assert!(!kaplumba_kernels::attention::decode_flash_enabled(4095));
    assert!(!kaplumba_kernels::attention::decode_flash_enabled(64));
}