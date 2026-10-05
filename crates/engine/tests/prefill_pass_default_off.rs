//! F3 (spec/prefill-24k-90tps, `adversary_report.md`): **the OQ9 prefill
//! key-range passes ship OFF**, and this is the test that pins it.
//!
//! The finding was that three documents disagreed with the code: `PREFILL_PASS_KEYS`
//! was 8192, so every prefill dispatch with `key_count > 8192` sliced by default,
//! while `implementation_notes.md` 16.4 said "the default must stay one dispatch"
//! and `spec.md` OQ9 said the implementation "ships off with
//! `KAPLUMBA_PREFILL_PASS_KEYS=0`". Two of the three were false.
//!
//! `PREFILL_PASS_KEYS` is now **0**. The implementation, its 17 bit-exactness tests
//! and F2's `PREFILL_PASS_MIN_GROUPS` grid gate all stay;
//! `KAPLUMBA_PREFILL_PASS_KEYS=8192` is the opt-in.
//!
//! **Why the default is off, in one testable sentence:** the mechanism has no
//! measured benefit at any context this box can measure (-0.13% at 16 394 tokens
//! over 3 interleaved rounds, inside a 0.30-0.63% spread), and a mechanism with no
//! measured benefit should not be a *default*, because a default is a claim.
//!
//! **This file must never set `KAPLUMBA_PREFILL_PASS_KEYS`.** It is a separate test
//! binary from `prefill_pass_gate_production_path.rs` precisely because
//! `prefill_pass_keys_for` reads the variable through a `OnceLock`: one process can
//! only ever see one value of it. That file turns passes on and tests F2's grid
//! gate; this one leaves them off and tests F3's decision. Together they cover both
//! halves of the rule and neither can pass vacuously.
//!
//! Positive control: `the_shape_is_eligible_when_passes_are_enabled` dispatches the
//! **same** buffers and the **same** kernel through the explicit entry point with
//! an explicit 8192. Without it, the "production declines" assertions below could
//! pass because the device, the kernels or the buffers are broken.
//!
//! Run: cargo test --release -p kaplumba-engine --test prefill_pass_default_off -- --nocapture

use kaplumba_kernels::attention::{self, AttentionEncodeRequest, AttentionKernels};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

const Q_HEADS: usize = 32;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 512;
const GROUPS: usize = Q_HEADS / KV_HEADS;

/// The one shape that matters: the last full chunk of a 16 394-token prefill.
/// 1024 queries at `BQ = 2` is a 2048-threadgroup grid, comfortably **above** the
/// 512 gate, and 16 384 keys is above the 8192 pass size -- so with passes enabled
/// this shape *is* sliced, and with the shipped default it is not. Choosing a shape
/// that is above both thresholds is what stops the test from being vacuous: the
/// only thing standing between it and a chain is the default.
struct Case {
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

    fn production_dispatched(&self) -> bool {
        let mut enc = MetalEncoder::new(&self.queue).expect("encoder");
        let dispatched = self
            .kernels
            .encode_online_attention_passes(&mut enc, self.request(), self.block(), &self.state)
            .expect("production entry point");
        enc.commit_and_wait().expect("wait");
        dispatched
    }

    /// The explicit entry point, which by design ignores the grid gate and takes the
    /// pass size as an argument. Used only as the positive control.
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

/// The shipped constant is zero. One line, and it is the whole of F3's code change.
#[test]
fn the_shipped_default_is_one_dispatch() {
    assert_eq!(
        attention::PREFILL_PASS_KEYS,
        0,
        "F3: the OQ9 pass chain is off by default. If this was changed to 8192 then \
         implementation_notes.md 16.4 paragraph 1 and spec.md OQ9 must be updated \
         too, and the measured cost re-quoted."
    );
}

/// ...which means `prefill_pass_keys_for` answers 0 at **every** key count,
/// including ones far above oMLX's 8192. Checked over a range rather than at one
/// point because the failure mode being guarded against is "someone raised the
/// default and only a short prompt still declines".
#[test]
fn no_key_count_selects_a_pass_chain_by_default() {
    for key_count in [
        1usize, 64, 1024, 8191, 8192, 8193, 16_384, 16_394, 24_590, 1_048_576,
    ] {
        assert_eq!(
            attention::prefill_pass_keys_for(key_count),
            0,
            "key_count {key_count} must not select a pass chain by default"
        );
    }
}

/// And the production entry point therefore declines **without touching the
/// encoder**, on a shape that is above *both* thresholds (16 384 keys > 8192 and a
/// 2048-threadgroup grid > 512). Returning `false` without encoding is what lets
/// `model.rs` fall through to the one-dispatch body instead of half-encoding a
/// chain.
#[test]
fn production_dispatches_one_dispatch_by_default_on_a_shape_that_would_slice() {
    let case = Case::new(1024, 16_384);
    assert!(
        case.kernels.prefill_pass_available(),
        "pass kernels did not compile; this file would be vacuous"
    );
    assert_eq!(case.block(), 2, "16 384 keys selects BQ=2");
    assert_eq!(
        case.grid(),
        2048,
        "the shape must be above PREFILL_PASS_MIN_GROUPS = {} or the test is \
         vacuous",
        attention::PREFILL_PASS_MIN_GROUPS
    );
    let mut enc = MetalEncoder::new(&case.queue).expect("encoder");
    let dispatched = case
        .kernels
        .encode_online_attention_passes(&mut enc, case.request(), case.block(), &case.state)
        .expect("production entry point");
    enc.commit_and_wait().expect("wait");
    assert!(
        !dispatched,
        "production sliced a {}-threadgroup grid with {} keys although \
         PREFILL_PASS_KEYS = 0",
        case.grid(),
        case.key_count
    );
}

/// **Positive control.** The same buffers, the same kernels, the same grid, the
/// explicit entry point and an explicit 8192: a chain *is* encoded. So the three
/// assertions above are about the default, not about a broken device or a kernel
/// that failed to build.
#[test]
fn the_shape_is_eligible_when_passes_are_enabled() {
    let case = Case::new(1024, 16_384);
    assert!(case.explicit_dispatched(8192));
    // ...and a pass size at or below the key count is not a chain at all, which is
    // what `prefill_pass_keys_for` used to be relied on to decide.
    assert!(
        !case.explicit_dispatched(16_384),
        "one pass over the whole key range is a single dispatch, not a chain"
    );
}