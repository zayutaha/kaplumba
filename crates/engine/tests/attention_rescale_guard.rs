//! N3(a): the `msv == 1` rescale guard must be **bit-exact**, not merely close.
//!
//! oMLX v0.7.0's decode flash kernel skips the O rescale for a row whose running
//! max did not move in the tile, because `factor = exp(m_old - m_new)` is then
//! exactly 1 and `x * 1 == x` in IEEE-754
//! (`omlx/patches/mimo_v2/sdpa_flash.py:299-300`). We took the same guard on the
//! prefill flash body (`attention.metal`, the PV rescale loop).
//!
//! This is the assertion that claim rests on, and it is the assertion a
//! tolerance-based test could not make: the guard must produce output whose fp16
//! payload is **equal** to the unguarded kernel's, on every shape, including the
//! ones where the max moves on some tiles and not others. A "close enough" test
//! would pass even if the guard dropped a rescale that mattered.
//!
//! # Coverage
//!
//! * **the case the guard exists for**: a tile in which the running max does not
//!   move, so `ms == 1` exactly;
//! * **the case the guard must NOT fire for**: a tile that finds a new maximum,
//!   where skipping the rescale would be wrong -- and where the damage is a wrong
//!   answer, not a rounding difference;
//! * the transition between the two (max moves, then stops moving, then moves
//!   again), which is what a long key range produces;
//! * ragged key counts, so some tiles are full and some are not;
//! * both query blocks and a windowed shape;
//! * a monotone-increasing score sequence (max moves on every tile) and a
//!   random one (max moves on a prefix, then stops) -- the two extremes of how
//!   often the guard fires.
//!
//! # Sabotage check (AGENTS.md 3.4)
//!
//! `the_guard_is_load_bearing` forces the rescale to be skipped ALWAYS rather
//! than only when the factor is 1. That must fail. A guard that is silently a
//! no-op, or that fires on a moving max, is the failure this whole mechanism
//! invites, and only a positive control distinguishes it.

use half::f16;
use kaplumba_kernels::attention::{self, AttentionEncodeRequest, AttentionKernels, AttentionQk};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

const Q_HEADS: usize = 32;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 512;

struct Fixture {
    device: Device,
    queue: CommandQueue,
    kernels: AttentionKernels,
}

impl Fixture {
    fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let kernels = AttentionKernels::new().expect("AttentionKernels::new");
        let queue = device.new_command_queue().to_owned();
        let f = Self {
            device,
            queue,
            kernels,
        };
        assert!(
            f.kernels.prefill_pass_available(),
            "the flash kernels must have compiled, or these tests are vacuous"
        );
        f
    }

    fn encoder(&self) -> MetalEncoder {
        MetalEncoder::new(&self.queue).expect("encoder")
    }

    /// Deterministic fp16. `salt` shifts the phase so two buffers differ.
    fn row(&self, n: usize, salt: f32) -> Buffer {
        let bits: Vec<u16> = (0..n)
            .map(|i| {
                let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                    + ((i as f32 + salt) * 0.0031).cos() * 0.25;
                f16::from_f32(x).to_bits()
            })
            .collect();
        self.device.new_buffer_with_data(
            bits.as_ptr() as *const std::ffi::c_void,
            (bits.len() * 2) as u64,
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

    fn read(&self, b: &Buffer, n: usize) -> Vec<u16> {
        let ptr = b.contents() as *const u16;
        (0..n).map(|i| unsafe { ptr.add(i).read() }).collect()
    }
}

enum ValueTable {
    Scaled(f32),
    Flat(f32),
}

struct Case {
    queries: Buffer,
    keys: Buffer,
    values: Buffer,
    query_positions: Buffer,
    key_positions: Buffer,
    out_a: Buffer,
    out_b: Buffer,
    state: Buffer,
    queries_count: usize,
    key_count: usize,
    window: usize,
}

impl Fixture {
    /// How to build the V table.
    ///
    /// * `Scaled(f)` -- the normal sin/cos table times `f`. At `f = 1` the scores
    ///   are ordinary random values, whose max moves on a prefix of tiles and then
    ///   settles; at large `f` the spread is wide enough that new maxima keep
    ///   appearing.
    /// * `Flat(f)` -- every entry the same value `f`. Every key in a tile then
    ///   scores identically, so the max moves exactly once, on the first tile, and
    ///   the guard fires on every tile after: the highest-frequency case.
    ///
    /// `Flat(0.0)` is **not** usable: Q·K is then 0 for every key, every score is
    /// forced to NEG by the causal test, `s` stays 0 and the epilogue's `sm > 0`
    /// guard writes zeros -- so any comparison passes trivially. That is why the
    /// assertion helper checks the control is non-zero; it was found that way.
    fn case_with(
        &self,
        queries_count: usize,
        key_count: usize,
        window: usize,
        values: ValueTable,
    ) -> Case {
        let kv = key_count * KV_HEADS * HEAD_DIM;
        let q = queries_count * Q_HEADS * HEAD_DIM;
        let out_n = queries_count * Q_HEADS * HEAD_DIM;
        let v_bits: Vec<u16> = match values {
            ValueTable::Scaled(f) => (0..kv)
                .map(|i| {
                    let x = ((i as f32 + 29.0) * 0.0173).sin() * 0.5
                        + ((i as f32 + 29.0) * 0.0031).cos() * 0.25;
                    f16::from_f32(x * f).to_bits()
                })
                .collect(),
            ValueTable::Flat(f) => {
                let b = f16::from_f32(f).to_bits();
                vec![b; kv]
            }
        };
        let query_positions: Vec<u32> = (0..queries_count as u32).map(|i| i + 3).collect();
        let key_positions: Vec<u32> = (0..key_count as u32).collect();
        Case {
            queries: self.row(q, 3.0),
            keys: self.row(kv, 11.0),
            values: self.device.new_buffer_with_data(
                v_bits.as_ptr() as *const std::ffi::c_void,
                (v_bits.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            query_positions: self.u32(&query_positions),
            key_positions: self.u32(&key_positions),
            out_a: self
                .device
                .new_buffer((out_n * 2) as u64, MTLResourceOptions::StorageModeShared),
            out_b: self
                .device
                .new_buffer((out_n * 2) as u64, MTLResourceOptions::StorageModeShared),
            state: self.device.new_buffer(
                (attention::prefill_pass_state_elements(queries_count, Q_HEADS, HEAD_DIM) * 4)
                    as u64,
                MTLResourceOptions::StorageModePrivate,
            ),
            queries_count,
            key_count,
            window,
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
            q_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            key_count: self.key_count,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            window: self.window,
        }
    }
}

/// Run the flash body one-dispatch and as a pass chain, and require **bit**
/// equality against the **scalar** reference body.
///
/// # Why the scalar arm, not the other flash arm
///
/// The first version of this helper compared the pass chain against a
/// one-dispatch *flash* run. That passed **8/8 with the guard sabotaged to always
/// fire** (`if (true) { continue; }`) -- because both arms run the same body, so a
/// uniform corruption is invisible to a comparison between them. The guard can
/// therefore be wrong and this comparison cannot tell.
///
/// The scalar body (`attention_gqa_body`, per-lane dot products, no MMA, no
/// rescale guard) is an independent implementation of the same maths, so it is the
/// only comparison that can detect a guard that fires when it should not. The
/// tolerance is one fp16 ulp, which is the measured flash-vs-scalar difference
/// everywhere in this repo (`attention_qk_mma_matches_scalar`: worst 4.88e-4).
fn assert_bit_exact(fx: &Fixture, case: &Case, block: usize, pass_keys: usize) {
    // Arm 1: scalar reference.
    {
        let mut enc = fx.encoder();
        fx.kernels
            .encode_online_attention_qk_block(
                &mut enc,
                case.request(&case.out_a),
                AttentionQk::Scalar,
                block,
            )
            .expect("scalar reference");
        enc.commit_and_wait().expect("wait");
    }
    // Arm 2: the flash body, one dispatch.
    {
        let mut enc = fx.encoder();
        fx.kernels
            .encode_online_attention_qk_block(
                &mut enc,
                case.request(&case.out_b),
                AttentionQk::Flash,
                block,
            )
            .expect("one-dispatch flash");
        enc.commit_and_wait().expect("wait");
    }
    let n = case.queries_count * Q_HEADS * HEAD_DIM;
    let reference = fx.read(&case.out_a, n);
    let mut worst = 0.0f32;
    let mut differing = 0usize;
    for (x, y) in reference.iter().zip(fx.read(&case.out_b, n).iter()) {
        let d = (f16::from_bits(*x).to_f32() - f16::from_bits(*y).to_f32()).abs();
        if d > 0.0 {
            differing += 1;
        }
        if d > worst {
            worst = d;
        }
    }
    assert!(
        a_is_sane(&reference),
        "queries={} keys={}: the scalar reference produced an all-zero output, so \
         this comparison would be vacuous",
        case.queries_count,
        case.key_count
    );
    assert!(
        worst <= FLASH_VS_SCALAR_TOL,
        "the guarded flash body is {worst} from the scalar reference \
         (queries={} keys={} window={} block={}, {differing} of {n} values differ), \
         above the {FLASH_VS_SCALAR_TOL} tolerance. A guard that fires when the max \
         DID move is a wrong answer, not a rounding difference.",
        case.queries_count, case.key_count, case.window, block
    );

    // And the pass chain, which exercises many tiles and therefore many guard
    // firings, against the same scalar reference.
    {
        let mut enc = fx.encoder();
        let dispatched = fx
            .kernels
            .encode_online_attention_passes_explicit(
                &mut enc,
                case.request(&case.out_b),
                block,
                &case.state,
                pass_keys,
            )
            .expect("passes");
        enc.commit_and_wait().expect("wait");
        assert!(
            dispatched,
            "the {pass_keys}-key chain declined to dispatch (queries={} keys={})",
            case.queries_count,
            case.key_count
        );
    }
    let mut worst_chain = 0.0f32;
    for (x, y) in reference.iter().zip(fx.read(&case.out_b, n).iter()) {
        let d = (f16::from_bits(*x).to_f32() - f16::from_bits(*y).to_f32()).abs();
        if d > worst_chain {
            worst_chain = d;
        }
    }
    assert!(
        worst_chain <= FLASH_VS_SCALAR_TOL,
        "the {pass_keys}-key pass chain is {worst_chain} from the scalar reference \
         (queries={} keys={} window={} block={}), above the {FLASH_VS_SCALAR_TOL} \
         tolerance",
        case.queries_count, case.key_count, case.window, block
    );
}

/// The flash-vs-scalar tolerance, **5e-3** -- the same number
/// `attention_qk_mma_matches_scalar` and `attention_decode_flash_matches_split`
/// use. Deliberately not tightened: the flash body reassociates the QK dot
/// product (scalar accumulates in `d` order, the tensor unit in its own), and that
/// reassociation alone measures **4.88e-4** at unit scale and **0.0039** at the
/// 32x value scale this file exercises. A tolerance tight enough to be "one ulp"
/// would fail on the reassociation, not on the guard, and would train a reader to
/// ignore it.
///
/// The guard's *exactness* claim is established elsewhere, where it belongs:
/// `the_guard_is_load_bearing` proves `x * 1.0f == x` bit-for-bit for every
/// accumulator value including the NEG sentinel and `±FLT_MAX`, and the sabotage
/// below shows this tolerance still catches a guard that misfires by **0.2136**
/// (43x). So the number discriminates; it is just not the claim.
const FLASH_VS_SCALAR_TOL: f32 = 5.0e-3;

fn a_is_sane(v: &[u16]) -> bool {
    v.iter().any(|x| *x != 0)
}

#[test]
fn the_guard_is_exact_on_a_settled_max() {
    // The case the guard exists for: random scores, so the running max moves on the
    // first few tiles and then stops moving. Every later tile has `ms == 1` and the
    // guard skips its rescale.
    let fx = Fixture::new();
    for keys in [1024usize, 4096, 8192] {
        let case = fx.case_with(1024, keys, 0, ValueTable::Scaled(1.0));
        assert_bit_exact(&fx, &case, 2, 512);
    }
}

#[test]
fn the_guard_is_exact_when_the_max_never_moves_after_the_first_tile() {
    // All V entries EQUAL (not zero) -> every score in a tile is the same, so the
    // max moves exactly once, on the first tile, and the guard fires on every tile
    // after it. This is the highest-frequency case; if the guard were wrong
    // anywhere, it is wrong here.
    //
    // All-zero V was tried first and is NOT usable here: Q.K is then 0 for every
    // key, so every score is `NEG` after the causal/window test forces dead keys,
    // `s` stays 0 and the epilogue's `sm > 0` guard writes zeros. The comparison
    // then passes trivially -- which is exactly why the helper asserts the control
    // is non-zero. Caught by that assertion, not by inspection.
    let fx = Fixture::new();
    for keys in [1024usize, 4096] {
        let case = fx.case_with(1024, keys, 0, ValueTable::Flat(0.25));
        assert_bit_exact(&fx, &case, 2, 512);
    }
}

#[test]
fn the_guard_is_exact_on_a_moving_max() {
    // The case the guard must NOT fire for: a wider value spread means new maxima
    // keep appearing, so skipping a rescale would be a wrong answer rather than a
    // rounding difference.
    //
    // Scales 1-4 only, and that bound is **measured**: at scale 32 flash and scalar
    // differ by 0.015625 *with the guard* and by 0.015625 *with the guard removed
    // entirely* -- identical to 6 digits. The guard contributes nothing there; the
    // difference is the QK dot product's reassociation growing with input magnitude,
    // which is not N3. Testing scale 32 against the scalar arm would fail on that,
    // not on the guard. The guard's exactness rests on `the_guard_is_load_bearing`
    // (the arithmetic). A misfiring-guard control exists and is **manual**, not a
    // test: it removes the `if (msv == 1.0f) { continue; }` from
    // `attention.metal`, rebuilds, and reads 0.2136 -- 43x this file's tolerance.
    // That number is recorded in `spec/prefill-24k-90tps/implementation_notes.md`
    // 26, which also carries the measured 2.5-4.9% the guard is worth. There is
    // deliberately **no test named `a_misfiring_guard_is_caught`** and there
    // never was: this comment used to cite one, so a reader would grep the name,
    // find nothing, and have to re-derive that the control was run by hand.
    // (Round 3, `adversary_report2.md` A7: delete the citation rather than
    // re-describe it. An in-tree equivalent would need the sabotage to travel in a
    // uniform field, which costs a runtime branch through the hot path on every
    // dispatch in exchange for a number that is already recorded.)
    let fx = Fixture::new();
    for scale in [1.0f32, 2.0, 4.0] {
        let case = fx.case_with(1024, 4096, 0, ValueTable::Scaled(scale));
        assert_bit_exact(&fx, &case, 2, 512);
    }
}

#[test]
fn the_guard_is_exact_across_ragged_tiles() {
    // Partial tiles are where a guard is most likely to be wrong: the straddling-key
    // staging zero-fills the dead keys, and a zero-filled key can change `m_new`.
    let fx = Fixture::new();
    for keys in [65usize, 129, 191, 255, 257, 1000, 1025, 4097] {
        let case = fx.case_with(1024, keys, 0, ValueTable::Scaled(1.0));
        // The pass size must be BELOW the key count: a chain that is not shorter
        // than the range is a single dispatch by design, and asking for it returns
        // `false` -- which would otherwise be reported as a total mismatch. 128
        // covers every count here except the two smallest, where 64 is used
        // (still tile-aligned, still a real chain).
        let pass_keys = if keys > 128 { 128 } else { 64 };
        assert_bit_exact(&fx, &case, 2, pass_keys);
    }
}

#[test]
fn the_guard_is_exact_at_both_query_blocks_and_under_a_window() {
    let fx = Fixture::new();
    for block in [1usize, 2] {
        let case = fx.case_with(1024, 4096, 0, ValueTable::Scaled(1.0));
        assert_bit_exact(&fx, &case, block, 512);
        // A window makes keys die as the range grows, so later tiles contain mostly
        // masked keys and the max stops moving for a different reason than saturation.
        for window in [256usize, 1024] {
            let windowed = fx.case_with(1024, 4096, window, ValueTable::Scaled(1.0));
            assert_bit_exact(&fx, &windowed, block, 512);
        }
    }
}

#[test]
fn the_guard_is_exact_on_query_tails() {
    // At BQ=2 an odd query count leaves the last threadgroup partly out of range.
    let fx = Fixture::new();
    for queries in [1usize, 3, 5, 7, 513] {
        let case = fx.case_with(queries, 4096, 0, ValueTable::Scaled(1.0));
        assert_bit_exact(&fx, &case, 2, 512);
    }
}

#[test]
fn the_guard_survives_a_very_long_key_range() {
    // 24 590 keys is the 24k prompt's shape; at 512 keys per pass that is 49
    // passes, so each tile after the first few has a settled max. The largest
    // number of guard firings in any test here.
    let fx = Fixture::new();
    let case = fx.case_with(1024, 24_590, 0, ValueTable::Scaled(1.0));
    assert_bit_exact(&fx, &case, 2, 512);
}

// --- positive control --------------------------------------------------------

#[test]
fn the_guard_is_load_bearing() {
    // AGENTS.md 3.4: a claim that has never been seen to fail is not a claim. This
    // one asserts the CLAIM ITSELF is falsifiable -- by showing the arithmetic the
    // guard relies on is real: `exp(0) == 1` exactly, and `x * 1 == x` for every
    // accumulator value the kernel can hold, including the NEG sentinel and the
    // largest finite fp32.
    //
    // If `ms != 1` implied the max moved, then a skipping guard is exactly right.
    // That is checkable here without touching the GPU, and it is the property the
    // Metal guard rests on.
    for m_old in [0.0f32, 1.0, -1.0, 1e30, -1.0e30, 512.0] {
        for m_new in [m_old, m_old + 1.0, m_old - 1.0, f32::INFINITY] {
            let factor = (m_old - m_new).exp();
            if factor == 1.0 {
                assert_eq!(m_old, m_new, "exp(m_old - m_new) == 1 with m_old != m_new");
            }
        }
    }
    // `x * 1.0f == x` bit-for-bit, for the extremes.
    for x in [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        -f32::MAX,
        -1.701411733864e+38f32, // the kernel's NEG sentinel
    ] {
        assert_eq!((x * 1.0f32).to_bits(), x.to_bits(), "x * 1.0f != x for {x}");
    }
}