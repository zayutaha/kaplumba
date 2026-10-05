//! F2 (spec/prefill-24k-90tps, `adversary_report.md`): the
//! `PREFILL_PASS_MIN_GROUPS` grid gate was **documented on the production entry
//! point and applied nowhere**.
//!
//! `prefill_pass_count()` is the only implementation of the rule.
//! `encode_online_attention_passes()` -- the function `model.rs` calls -- used to
//! forward straight to `encode_online_attention_passes_explicit()`, whose own
//! comment said the gate "belongs to `prefill_pass_count`, which the production
//! caller applies through `encode_online_attention_passes`". It did not. So the
//! gate's tests in `attention_prefill_pass_bit_exact.rs` exercised a function with
//! no production caller, and the shipped path sliced grids as small as **20
//! threadgroups**.
//!
//! That 20-threadgroup grid is not synthetic: it is the last chunk of a
//! 16 394-token prompt (10 queries, 16 394 keys, `flash_query_block_for` -> BQ=2),
//! which every long prefill produces. `implementation_notes.md` 16.5 documented the
//! opposite ("128 queries x 4 kv heads at BQ=1 = 512 groups -> one dispatch; 129
//! queries -> sliced").
//!
//! These tests call the **production entry point** and pin the gate on it. The
//! positive control is [`the_production_path_still_slices_a_grid_above_the_gate`]:
//! it asserts the production path DOES dispatch a chain on a shape whose grid is
//! well above 512 threadgroups, so the two tests together cannot both be passing
//! for the reason that the encoder never dispatches anything.
//!
//! Run: cargo test --release -p kaplumba-engine --test prefill_pass_gate_production_path -- --nocapture

use kaplumba_kernels::attention::{self, AttentionEncodeRequest, AttentionKernels};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

const Q_HEADS: usize = 32;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 512;
const GROUPS: usize = Q_HEADS / KV_HEADS;

/// A real production shape: `queries` query rows over `key_count` keys, flash shape
/// (head_dim 512 / groups 8), on a real device. Buffer sizes are the real ones --
/// at 16 394 keys this allocates ~134 MB of KV, which is what a 16k prefill holds
/// for one full-attention layer, and it must, because a smaller buffer would let a
/// kernel read off the end and turn a "gate declined" bug into a memory fault
/// instead of an observable return value.
struct Case {
    device: Device,
    queue: CommandQueue,
    kernels: AttentionKernels,
    queries_count: usize,
    key_count: usize,
    queries: Buffer,
    keys: Buffer,
    values: Buffer,
    qpos: Buffer,
    kpos: Buffer,
    out: Buffer,
    state: Buffer,
}

impl Case {
    fn new(queries_count: usize, key_count: usize) -> Self {
        let device = Device::system_default().expect("no Metal device");
        let kernels = AttentionKernels::new().expect("AttentionKernels::new");
        let queue = device.new_command_queue().to_owned();
        let kv = key_count * KV_HEADS * HEAD_DIM;
        let q = queries_count * Q_HEADS * HEAD_DIM;
        let queries = half_buf(&device, q, 3.0);
        let keys = half_buf(&device, kv, 11.0);
        let values = half_buf(&device, kv, 29.0);
        let qp: Vec<u32> = (0..queries_count as u32).map(|i| i + 3).collect();
        let kp: Vec<u32> = (0..key_count as u32).collect();
        let qpos = u32_buf(&device, &qp);
        let kpos = u32_buf(&device, &kp);
        let out = device.new_buffer((q * 2) as u64, MTLResourceOptions::StorageModeShared);
        let state = device.new_buffer(
            (attention::prefill_pass_state_elements(queries_count, Q_HEADS, HEAD_DIM) * 4) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        Self {
            device,
            queue,
            kernels,
            queries_count,
            key_count,
            queries,
            keys,
            values,
            qpos,
            kpos,
            out,
            state,
        }
    }

    fn block(&self) -> usize {
        attention::flash_query_block_for(self.key_count)
    }

    fn grid(&self) -> usize {
        attention::prefill_threadgroups(self.queries_count, KV_HEADS, self.block())
    }

    /// What the documented gate says for this shape.
    fn documented(&self) -> usize {
        attention::prefill_pass_count(
            self.queries_count,
            KV_HEADS,
            self.key_count,
            HEAD_DIM,
            GROUPS,
            self.block(),
            true,
        )
    }

    fn request(&self) -> AttentionEncodeRequest<'_> {
        AttentionEncodeRequest {
            queries: &self.queries,
            keys: &self.keys,
            values: &self.values,
            query_positions: &self.qpos,
            key_positions: &self.kpos,
            output: &self.out,
            queries_count: self.queries_count,
            q_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            key_count: self.key_count,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            window: 0,
        }
    }

    /// The **production** entry point.
    fn production_dispatched(&self) -> bool {
        let mut enc = MetalEncoder::new(&self.queue).expect("encoder");
        let dispatched = self
            .kernels
            .encode_online_attention_passes(&mut enc, self.request(), self.block(), &self.state)
            .expect("production entry point");
        enc.commit_and_wait().expect("wait");
        dispatched
    }

    /// The test-only entry point, which by design ignores the grid gate. Used only
    /// to prove the shapes here are otherwise eligible, so a "gate declined"
    /// result cannot be explained by something else declining.
    fn explicit_dispatched(&self, pass_keys: usize) -> bool {
        let mut enc = MetalEncoder::new(&self.queue).expect("encoder");
        let dispatched = self
            .kernels
            .encode_online_attention_passes_explicit(
                &mut enc,
                self.request(),
                self.block(),
                &self.state,
                pass_keys,
            )
            .expect("explicit entry point");
        enc.commit_and_wait().expect("wait");
        dispatched
    }
}

fn half_buf(device: &Device, n: usize, salt: f32) -> Buffer {
    let bits: Vec<u16> = (0..n)
        .map(|i| {
            let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                + ((i as f32 + salt) * 0.0031).cos() * 0.25;
            half::f16::from_f32(x).to_bits()
        })
        .collect();
    device.new_buffer_with_data(
        bits.as_ptr() as *const std::ffi::c_void,
        (n * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn u32_buf(device: &Device, v: &[u32]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// F2: the last chunk of a 16 394-token prompt has a 20-threadgroup grid, and the
/// documented gate (512) forbids slicing it. Production must decline.
#[test]
fn the_production_path_applies_the_documented_grid_gate() {
    let case = Case::new(10, 16_394);
    assert!(
        case.kernels.prefill_pass_available(),
        "pass kernels did not compile; this file would be vacuous"
    );
    assert_eq!(case.block(), 2, "16 394 keys selects BQ=2");
    let grid = case.grid();
    assert_eq!(
        grid, 20,
        "this shape is the last chunk of a 16 394-token prompt: 10 queries x 4 kv heads at BQ=2"
    );
    assert_eq!(
        case.documented(),
        0,
        "the documented gate declines a {} -threadgroup grid", grid
    );
    // Not for lack of eligibility: the same shape with the grid gate bypassed DOES
    // dispatch a chain, so a `false` from production can only be the gate.
    assert!(
        case.explicit_dispatched(attention::PREFILL_PASS_KEYS),
        "the shape is not eligible for anything, so this test proves nothing about the gate"
    );
    assert!(
        !case.production_dispatched(),
        "production sliced a {grid}-threadgroup grid, which the {} -group gate forbids: \
         `prefill_pass_count` has no production caller again",
        attention::PREFILL_PASS_MIN_GROUPS
    );
    // And production must leave the encoder empty, so the caller falls through to
    // the one-dispatch body rather than half-encoding.
    let mut enc = MetalEncoder::new(&case.queue).expect("encoder");
    let dispatched = case
        .kernels
        .encode_online_attention_passes(&mut enc, case.request(), case.block(), &case.state)
        .expect("production entry point");
    enc.commit_and_wait().expect("wait");
    assert!(
        !dispatched,
        "the gate must decline without touching the encoder, so model.rs falls through \
         to encode_online_attention instead of encoding half a chain"
    );
}

/// Positive control for the test above: a grid **above** the gate must still be
/// sliced on the production path. Without this, both tests could pass because the
/// production entry point never dispatches anything at all.
#[test]
fn the_production_path_still_slices_a_grid_above_the_gate() {
    // 1024 queries x 4 kv heads at BQ=2 = 2048 threadgroups, well above 512.
    let case = Case::new(1024, 16_384);
    assert!(
        case.kernels.prefill_pass_available(),
        "pass kernels did not compile; this file would be vacuous"
    );
    assert_eq!(case.block(), 2, "16 384 keys selects BQ=2");
    let grid = case.grid();
    assert_eq!(grid, 2048);
    let passes = case.documented();
    assert!(
        passes >= 2,
        "the gate must ALLOW this shape; got {passes}. If this fails the other test \
         in this file is vacuous."
    );
    assert!(
        case.production_dispatched(),
        "production declined a {grid}-threadgroup grid that the gate allows"
    );
}

/// The gate boundary itself, on the production path: **512** threadgroups keeps one
/// dispatch, **516** slices. `PREFILL_PASS_MIN_GROUPS` is `<=`, so equality is on
/// the unsliced side -- that is oMLX's rule (`n_groups < min_groups`,
/// `nax_attention.py:797`) and the notes' reading of their comment ("query tails
/// under ~512 rows at 64 heads keep one dispatch", `:91-92`).
///
/// At 16 384 keys `flash_query_block_for` selects BQ=2, so `ceil(q/2) * 4`: 256
/// queries -> 512 groups, 258 -> 516.
#[test]
fn the_gate_boundary_is_512_threadgroups_on_the_production_path() {
    let at = Case::new(256, 16_384); // ceil(256/2) * 4 = 512
    let above = Case::new(258, 16_384); // ceil(258/2) * 4 = 516
    assert_eq!(at.block(), 2);
    assert_eq!(at.grid(), 512, "the boundary shape must sit exactly ON the gate");
    assert_eq!(above.grid(), 516);
    assert_eq!(at.documented(), 0, "512 groups keeps one dispatch (gate is `<=`)");
    assert!(
        above.documented() >= 2,
        "516 groups is above the gate and must slice"
    );
    assert!(
        !at.production_dispatched(),
        "512 threadgroups must keep one dispatch on the production path"
    );
    assert!(
        above.production_dispatched(),
        "516 threadgroups must slice on the production path"
    );
}

/// Sabotage control, run by hand: raising `PREFILL_PASS_MIN_GROUPS` to
/// `usize::MAX` (i.e. "never slice") must make
/// `the_production_path_still_slices_a_grid_above_the_gate` FAIL. Recorded here so
/// the check is written down rather than remembered; the test file cannot assert
/// its own sabotage without a build-time knob, and this spec has been bitten
/// before by exactly that (AGENTS.md 3.4: the ablation that set an env flag
/// *disabling* the path under test).
#[test]
fn the_sabotage_for_this_file_is_written_down() {
    // A trivially-true assertion whose only job is to make the recorded sabotage
    // findable by grep from the failure it is meant to cause.
    assert_eq!(
        attention::PREFILL_PASS_MIN_GROUPS, 512,
        "if this constant has moved, the grid numbers in the two tests above are stale \
         and must be recomputed (the boundary is asserted, not hard-coded)"
    );
}