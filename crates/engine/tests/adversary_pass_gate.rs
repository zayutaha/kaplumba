//! ADVERSARY ARTIFACT (spec/prefill-24k-90tps) -- evidence for the finding that
//! `PREFILL_PASS_MIN_GROUPS` is **not applied on the production path**.
//!
//! Claim under test (implementation_notes.md 16.5):
//!   "The grid gate is `<= 512` threadgroups ... 128 queries x 4 kv heads at BQ=1
//!    = 512 groups -> one dispatch; 129 queries -> sliced."
//!
//! `prefill_pass_count()` implements that gate, and the gate test in
//! `attention_prefill_pass_bit_exact.rs` exercises `prefill_pass_count()`
//! directly. The production caller (`model.rs` -> `encode_online_attention_passes`)
//! does **not** call `prefill_pass_count()`: it forwards straight to
//! `encode_online_attention_passes_explicit()`, whose own doc comment says the
//! grid gate "belongs to `prefill_pass_count`, which the production caller
//! applies through `encode_online_attention_passes`". It does not.
//!
//! This file calls the PRODUCTION entry point and shows it returns `true` (i.e.
//! dispatches a pass chain) on a shape whose grid is 20 threadgroups, where
//! `prefill_pass_count()` returns 0.
//!
//! Run: cargo test --release -p kaplumba-engine --test adversary_pass_gate -- --nocapture

use kaplumba_kernels::attention::{self, AttentionEncodeRequest, AttentionKernels};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

const Q_HEADS: usize = 32;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 512;

fn fixture() -> (Device, CommandQueue, AttentionKernels) {
    let device = Device::system_default().expect("no Metal device");
    let kernels = AttentionKernels::new().expect("AttentionKernels::new");
    let queue = device.new_command_queue().to_owned();
    (device, queue, kernels)
}

fn u16_buf(device: &Device, n: usize, salt: f32) -> Buffer {
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

#[test]
fn the_production_entry_point_ignores_the_documented_grid_gate() {
    let (device, queue, kernels) = fixture();
    assert!(
        kernels.prefill_pass_available(),
        "pass kernels did not compile; this file would be vacuous"
    );

    // The production shape of the 16 394-token prompt's LAST chunk:
    // 10 queries, 16 394 keys, block = 2 (key_count >= FLASH_QUERY_BLOCK_MIN_KEYS).
    let queries_count = 10usize;
    let key_count = 16_394usize;
    let block = attention::flash_query_block_for(key_count);
    assert_eq!(block, 2, "the last chunk of a 16 394-token prompt uses BQ=2");

    let kv = key_count * KV_HEADS * HEAD_DIM;
    let q = queries_count * Q_HEADS * HEAD_DIM;
    let queries = u16_buf(&device, q, 3.0);
    let keys = u16_buf(&device, kv, 11.0);
    let values = u16_buf(&device, kv, 29.0);
    let qp: Vec<u32> = (0..queries_count as u32).map(|i| i + 3).collect();
    let kp: Vec<u32> = (0..key_count as u32).collect();
    let qpb = device.new_buffer_with_data(
        qp.as_ptr() as *const std::ffi::c_void,
        (qp.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let kpb = device.new_buffer_with_data(
        kp.as_ptr() as *const std::ffi::c_void,
        (kp.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out = device.new_buffer(
        (q * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let state = device.new_buffer(
        (attention::prefill_pass_state_elements(queries_count, Q_HEADS, HEAD_DIM) * 4) as u64,
        MTLResourceOptions::StorageModePrivate,
    );

    let request = AttentionEncodeRequest {
        queries: &queries,
        keys: &keys,
        values: &values,
        query_positions: &qpb,
        key_positions: &kpb,
        output: &out,
        queries_count,
        q_heads: Q_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        key_count,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        window: 0,
    };

    // 1. What the DOCUMENTED gate says. `prefill_pass_count` is what the notes
    //    and the gate test exercise.
    let grid = attention::prefill_threadgroups(queries_count, KV_HEADS, block);
    let documented = attention::prefill_pass_count(
        queries_count,
        KV_HEADS,
        key_count,
        HEAD_DIM,
        8,
        block,
        true,
    );
    println!(
        "ADV: grid = {grid} threadgroups (queries={queries_count}, kv_heads={KV_HEADS}, \
         block={block}); PREFILL_PASS_MIN_GROUPS = {}; \
         prefill_pass_count (the documented gate) = {documented}",
        attention::PREFILL_PASS_MIN_GROUPS
    );
    assert_eq!(documented, 0, "the documented gate declines this shape");

    // 2. What the PRODUCTION entry point does with the same shape.
    let mut enc = MetalEncoder::new(&queue).expect("encoder");
    let dispatched = kernels
        .encode_online_attention_passes(&mut enc, request, block, &state)
        .expect("production entry point");
    enc.commit_and_wait().expect("wait");

    println!("ADV: encode_online_attention_passes (production) dispatched = {dispatched}");
    println!(
        "ADV: pass edges for this key_count at the 8192 default = {:?}",
        attention::prefill_pass_edges(key_count, attention::PREFILL_PASS_KEYS)
    );
    assert!(
        dispatched,
        "the production path declined: the grid gate IS applied, and the finding is void"
    );
    println!(
        "ADV: CONFIRMED -- the shipped path slices a {grid}-threadgroup grid, which the \
         documented gate ({} groups) forbids. `prefill_pass_count` has no production caller.",
        attention::PREFILL_PASS_MIN_GROUPS
    );
}