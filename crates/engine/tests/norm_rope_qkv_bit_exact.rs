//! `norm_rope_qkv` must be BIT-IDENTICAL to the five dispatches it replaces.
//!
//! The unfused decode path is, per layer:
//!
//! ```text
//! rms_norm(q, q_norm, q)          weighted,   rows = q_heads
//! rms_norm(k, k_norm, k)          weighted,   rows = kv_heads
//! rms_norm_unweighted(v, v)       unweighted, rows = kv_heads
//! rope(q, q)                      heads = q_heads
//! rope(k, k)                      heads = kv_heads
//! ```
//!
//! All five are elementwise and the fp16 buffer between them is a *rounding*
//! step, so they are exactly fusible -- but only if the fused kernel reproduces
//! that rounding. Carrying the normed value in an fp32 register across the
//! rotation skips the narrowing, and `half(lo*cos - hi*sin)` is not a function
//! of the twice-rounded form, so the "obvious" fusion is off by up to half an
//! fp16 ULP on nearly every rotated element. The project rule is no quality
//! loss, which means `to_bits()` equality and not a tolerance, so this asserts
//! exact bit equality on the fp16 output.
//!
//! The test is un-`#[ignore]`d and needs no checkpoint: it builds buffers of the
//! model's real shapes and runs both paths on them, so it costs milliseconds and
//! runs on every build. It covers BOTH head dims (256 sliding, 512
//! full-attention), both rope modes (default and proportional), and both a
//! 1-row and a many-row `rows` so the dispatch-selection branch in
//! `encode_rms_norm` is on the same code path in both arms of the comparison.

use half::f16;
use kaplumba_kernels::elements::{ElementKernels, RopeBufferRequest};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, Device, MTLResourceOptions};

const THETA: f32 = 10_000.0;
const EPS: f32 = 1e-6;

/// Shapes worth testing: the model's two head dims, and `rows` on both sides of
/// `THREADGROUP_NORM_MAX_ROWS` (8192) so `encode_rms_norm` selects the same
/// threadgroup kernel in the reference arm that `norm_rope_qkv` always uses.
#[derive(Clone, Copy, Debug)]
struct Case {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    offset: usize,
    proportional: bool,
    partial_factor: f32,
    name: &'static str,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            q_heads: 32,
            kv_heads: 16,
            head_dim: 256,
            offset: 0,
            proportional: false,
            partial_factor: 1.0,
            name: "sliding/head_dim=256/default-rope",
        },
        Case {
            q_heads: 32,
            kv_heads: 16,
            head_dim: 512,
            offset: 0,
            proportional: false,
            partial_factor: 1.0,
            name: "full-attn/head_dim=512/default-rope",
        },
        // Non-zero offset: a decode step past the first token. A kernel that
        // ignored `offset` would still match at offset 0, which is why the
        // cases below do not all use 0.
        Case {
            q_heads: 4,
            kv_heads: 2,
            head_dim: 256,
            offset: 37,
            proportional: false,
            partial_factor: 1.0,
            name: "offset=37/head_dim=256",
        },
        Case {
            q_heads: 4,
            kv_heads: 2,
            head_dim: 512,
            offset: 1023,
            proportional: false,
            partial_factor: 1.0,
            name: "offset=1023/head_dim=512",
        },
        // Proportional RoPE leaves an unrotated TAIL. The tail is where a
        // one-thread-per-pair kernel that forgets the high half of the tail
        // writes half its output, so it is the case that catches that class of
        // bug, and it is exactly the bug `rope` itself once had.
        Case {
            q_heads: 4,
            kv_heads: 2,
            head_dim: 256,
            offset: 5,
            proportional: true,
            partial_factor: 0.5,
            name: "proportional=0.5/head_dim=256 (unrotated tail)",
        },
        Case {
            q_heads: 4,
            kv_heads: 2,
            head_dim: 512,
            offset: 5,
            proportional: true,
            partial_factor: 0.75,
            name: "proportional=0.75/head_dim=512 (unrotated tail)",
        },
        // `rows` far above THREADGROUP_NORM_MAX_ROWS, so the reference norms go
        // down the per-row path. Included because if the two ever disagree on
        // which kernel `rows` selects, the comparison would silently be
        // reference-fused vs fused rather than reference vs fused.
        Case {
            q_heads: 32,
            kv_heads: 16,
            head_dim: 256,
            offset: 11,
            proportional: false,
            partial_factor: 1.0,
            name: "offset=11/model-sliding-shape",
        },
    ]
}

/// Deterministic, well-conditioned, full-entropy fp16 bits.
fn pseudo_half(count: usize, seed: u32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // A spread over [-1, 1) so cos/sin genuinely mix the halves and a
            // dropped rotation cannot pass by accident.
            f16::from_f32(((state >> 8) as f32 / 8_388_608.0) - 1.0).to_bits()
        })
        .collect()
}

fn upload(device: &Device, bits: &[u16]) -> Buffer {
    let buffer = device.new_buffer(
        (bits.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    assert!(!buffer.contents().is_null(), "buffer allocation failed");
    unsafe {
        std::ptr::copy_nonoverlapping(bits.as_ptr(), buffer.contents() as *mut u16, bits.len());
    }
    buffer
}

fn download(buffer: &Buffer, len: usize) -> Vec<u16> {
    (0..len)
        .map(|index| {
            // Volatile so the read is not hoisted out of the loop or cached
            // across the two arms; the buffers are the same one here, so a
            // stale read would be a false PASS.
            unsafe { std::ptr::read_volatile((buffer.contents() as *const u16).add(index)) }
        })
        .collect()
}

/// The five dispatches the fused kernel replaces, run exactly as
/// `model.rs::forward_buffer` runs them for a one-token decode.
fn reference(
    device: &Device,
    case: &Case,
    q_in: &[u16],
    k_in: &[u16],
    v_in: &[u16],
    q_weight: &[u16],
    k_weight: &[u16],
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let kernels = ElementKernels::shared().expect("element kernels");
    let q = upload(device, q_in);
    let k = upload(device, k_in);
    let v = upload(device, v_in);
    let qw = upload(device, q_weight);
    let kw = upload(device, k_weight);
    let queue = device.new_command_queue();
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");

    // rows = 1 token x heads, width = head_dim: what a decode step passes.
    kernels
        .encode_rms_norm(&mut encoder, &q, &qw, &q, case.q_heads, case.head_dim, EPS)
        .expect("q norm");
    kernels
        .encode_rms_norm(&mut encoder, &k, &kw, &k, case.kv_heads, case.head_dim, EPS)
        .expect("k norm");
    kernels
        .encode_rms_norm_unweighted(&mut encoder, &v, &v, case.kv_heads, case.head_dim, EPS)
        .expect("v norm");
    kernels
        .encode_rope(
            &mut encoder,
            &q,
            &q,
            RopeBufferRequest {
                tokens: 1,
                heads: case.q_heads,
                head_dim: case.head_dim,
                offset: case.offset,
                theta: THETA,
                partial_factor: case.partial_factor,
                proportional: case.proportional,
            },
        )
        .expect("q rope");
    kernels
        .encode_rope(
            &mut encoder,
            &k,
            &k,
            RopeBufferRequest {
                tokens: 1,
                heads: case.kv_heads,
                head_dim: case.head_dim,
                offset: case.offset,
                theta: THETA,
                partial_factor: case.partial_factor,
                proportional: case.proportional,
            },
        )
        .expect("k rope");

    encoder.commit_and_wait().expect("reference commit");
    (
        download(&q, q_in.len()),
        download(&k, k_in.len()),
        download(&v, v_in.len()),
    )
}

fn fused(
    device: &Device,
    case: &Case,
    q_in: &[u16],
    k_in: &[u16],
    v_in: &[u16],
    q_weight: &[u16],
    k_weight: &[u16],
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let kernels = ElementKernels::shared().expect("element kernels");
    let q = upload(device, q_in);
    let k = upload(device, k_in);
    let v = upload(device, v_in);
    let qw = upload(device, q_weight);
    let kw = upload(device, k_weight);
    let queue = device.new_command_queue();
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    kernels
        .encode_norm_rope_qkv(
            &mut encoder,
            &q,
            &k,
            &v,
            &qw,
            &kw,
            case.q_heads,
            case.kv_heads,
            case.head_dim,
            case.offset,
            THETA,
            case.partial_factor,
            case.proportional,
            EPS,
        )
        .expect("encode_norm_rope_qkv");
    encoder.commit_and_wait().expect("fused commit");
    (
        download(&q, q_in.len()),
        download(&k, k_in.len()),
        download(&v, v_in.len()),
    )
}

/// Compare two buffers element-wise, reporting the first divergence with the
/// head/column it lands in so a failure names the layer shape that broke.
#[allow(clippy::too_many_arguments)]
fn assert_bit_identical(
    name: &str,
    tensor: &str,
    heads: usize,
    head_dim: usize,
    got: &[u16],
    want: &[u16],
) {
    assert_eq!(got.len(), want.len(), "{name}/{tensor} length");
    let mut differing = 0usize;
    let mut first = 0usize;
    let mut worst_ulp = 0i64;
    for index in 0..got.len() {
        if got[index] == want[index] {
            continue;
        }
        if differing == 0 {
            first = index;
        }
        differing += 1;
        // Signed-magnitude fp16 -> ordered integer, so "ULP" is a real distance.
        let key = |bits: u16| -> i64 {
            let signed = bits as i16 as i64;
            if signed < 0 {
                i64::from(i16::MIN) - signed
            } else {
                signed
            }
        };
        worst_ulp = worst_ulp.max((key(got[index]) - key(want[index])).abs());
    }
    assert_eq!(
        differing,
        0,
        "{name}: {tensor} is NOT bit-identical to the five-dispatch reference. \
         {differing} of {} elements differ, worst {worst_ulp} ULP, first at flat index \
         {first} = head {} column {} (head_dim {head_dim}, {heads} heads). \
         expected bits {:#06x} ({:e}), got {:#06x} ({:e}). \
         A nonzero count here means the fusion is not bit-exact: the usual causes are \
         skipping the fp16 rounding the separate norm wrote (the reference `rope` reads a \
         half back), or reducing the sum of squares in a different order than \
         `rms_norm_weighted_tg`.",
        got.len(),
        first / head_dim,
        first % head_dim,
        want[first],
        f16::from_bits(want[first]).to_f32(),
        got[first],
        f16::from_bits(got[first]).to_f32(),
    );
}

#[test]
fn norm_rope_qkv_is_bit_identical_to_the_five_dispatch_path() {
    let device = Device::system_default().expect("Metal device");
    for (index, case) in cases().into_iter().enumerate() {
        let q_in = pseudo_half(case.q_heads * case.head_dim, 0x51ed_0001 + index as u32);
        let k_in = pseudo_half(case.kv_heads * case.head_dim, 0x9e37_79b9 + index as u32);
        let v_in = pseudo_half(case.kv_heads * case.head_dim, 0x85eb_ca6b + index as u32);
        // Weights are shared across heads (one `head_dim` vector per projection),
        // which is the model's layout, and near 1.0 so the norm is not what
        // carries the test.
        let q_weight: Vec<u16> = (0..case.head_dim)
            .map(|c| f16::from_f32(0.75 + (c as f32 * 0.01).sin() * 0.25).to_bits())
            .collect();
        let k_weight: Vec<u16> = (0..case.head_dim)
            .map(|c| f16::from_f32(0.75 + (c as f32 * 0.013).cos() * 0.25).to_bits())
            .collect();

        let (rq, rk, rv) = reference(&device, &case, &q_in, &k_in, &v_in, &q_weight, &k_weight);
        let (fq, fk, fv) = fused(&device, &case, &q_in, &k_in, &v_in, &q_weight, &k_weight);

        assert_bit_identical(case.name, "q", case.q_heads, case.head_dim, &fq, &rq);
        assert_bit_identical(case.name, "k", case.kv_heads, case.head_dim, &fk, &rk);
        assert_bit_identical(case.name, "v", case.kv_heads, case.head_dim, &fv, &rv);
    }
}

/// The fused kernel must leave no element of any head unwritten.
///
/// A distinct failure mode from bit-exactness: if a branch covers fewer columns
/// than the head has (the unrotated-tail bug `rope` once had), the output is
/// *wrong* rather than *different*, and it is wrong by reading whatever was in
/// the buffer before.
///
/// The kernel is IN PLACE, so the "whatever was there before" is also the
/// input. The first version of this test uploaded real input over a poison
/// sentinel and then looked for the sentinel surviving -- which can never
/// happen, because the input overwrite already removed every one of them. It
/// passed while the high half of the unrotated tail was being dropped. The
/// input IS the poison instead: fill every buffer with 0x7bff and set the
/// weights to 1.0, and a written column comes out near 1.0 (the norm of a
/// constant vector times weight 1.0) while an unwritten column is still
/// 0x7bff. The test input does not have to be realistic; it has to make written
/// and unwritten distinguishable, and this does.
#[test]
fn norm_rope_qkv_writes_every_column_of_every_head() {
    let device = Device::system_default().expect("Metal device");
    // 0x7bff is the largest finite fp16. No norm or rotation of a constant
    // vector of 0x7bff produces 0x7bff back, so a survivor marks a column
    // nobody wrote.
    const POISON: u16 = 0x7bff;
    for case in cases() {
        let kernels = ElementKernels::shared().expect("element kernels");
        let q = upload(&device, &vec![POISON; case.q_heads * case.head_dim]);
        let k = upload(&device, &vec![POISON; case.kv_heads * case.head_dim]);
        let v = upload(&device, &vec![POISON; case.kv_heads * case.head_dim]);
        // Weight 1.0 keeps the norm output as close to the input scale as a norm
        // can, which is the case furthest from 0x7bff.
        let qw = upload(&device, &vec![f16::from_f32(1.0).to_bits(); case.head_dim]);
        let kw = upload(&device, &vec![f16::from_f32(1.0).to_bits(); case.head_dim]);
        let queue = device.new_command_queue();
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        kernels
            .encode_norm_rope_qkv(
                &mut encoder,
                &q,
                &k,
                &v,
                &qw,
                &kw,
                case.q_heads,
                case.kv_heads,
                case.head_dim,
                case.offset,
                THETA,
                case.partial_factor,
                case.proportional,
                EPS,
            )
            .expect("encode_norm_rope_qkv");
        encoder.commit_and_wait().expect("commit");

        for (tensor, buffer, heads) in [
            ("q", &q, case.q_heads),
            ("k", &k, case.kv_heads),
            ("v", &v, case.kv_heads),
        ] {
            let len = heads * case.head_dim;
            let bits = download(buffer, len);
            let unwritten: Vec<usize> = bits
                .iter()
                .enumerate()
                .filter(|(_, value)| **value == POISON)
                .map(|(position, _)| position)
                .collect();
            assert!(
                unwritten.is_empty(),
                "{}: {tensor} left {} of {len} columns unwritten (poison {POISON:#06x} \
                 survived), first at flat index {} = head {} column {} (head_dim {}, \
                 {} heads, rotated_pairs {}). A surviving poison means a branch does not \
                 cover the whole head -- for the unrotated tail that is the high half, \
                 since a pair spans columns p and p + head_dim/2.",
                case.name,
                unwritten.len(),
                unwritten.first().copied().unwrap_or(0),
                unwritten.first().copied().unwrap_or(0) / case.head_dim,
                unwritten.first().copied().unwrap_or(0) % case.head_dim,
                case.head_dim,
                heads,
                if case.proportional {
                    (case.partial_factor * case.head_dim as f32 / 2.0) as usize
                } else {
                    case.head_dim / 2
                },
            );
        }
    }
}

/// Input validation. Every rejected shape must be an `Err`, never a dispatch
/// with a zero group count (which Metal treats as a no-op that silently skips
/// the work) and never an out-of-bounds threadgroup access.
#[test]
fn norm_rope_qkv_rejects_invalid_shapes() {
    let device = Device::system_default().expect("Metal device");
    let kernels = ElementKernels::shared().expect("element kernels");
    let weight = upload(&device, &[f16::from_f32(1.0).to_bits(); 512]);
    let tensor = upload(&device, &vec![0u16; 32 * 512]);
    let queue = device.new_command_queue();
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    let call = |encoder: &mut MetalEncoder,
                q_heads: usize,
                kv_heads: usize,
                head_dim: usize,
                theta: f32,
                partial_factor: f32,
                proportional: bool,
                eps: f32| {
        kernels.encode_norm_rope_qkv(
            encoder,
            &tensor,
            &tensor,
            &tensor,
            &weight,
            &weight,
            q_heads,
            kv_heads,
            head_dim,
            0,
            theta,
            partial_factor,
            proportional,
            eps,
        )
    };
    // (label, q_heads, kv_heads, head_dim, theta, partial_factor, proportional, eps)
    let rejects: &[(&str, usize, usize, usize, f32, f32, bool, f32)] = &[
        ("zero q_heads", 0, 2, 256, THETA, 1.0, false, EPS),
        ("zero kv_heads", 4, 0, 256, THETA, 1.0, false, EPS),
        ("zero head_dim", 4, 2, 0, THETA, 1.0, false, EPS),
        ("odd head_dim", 4, 2, 255, THETA, 1.0, false, EPS),
        ("head_dim over staging", 4, 2, 514, THETA, 1.0, false, EPS),
        ("zero theta", 4, 2, 256, 0.0, 1.0, false, EPS),
        ("negative theta", 4, 2, 256, -1.0, 1.0, false, EPS),
        ("NaN theta", 4, 2, 256, f32::NAN, 1.0, false, EPS),
        ("zero eps", 4, 2, 256, THETA, 1.0, false, 0.0),
        ("negative eps", 4, 2, 256, THETA, 1.0, false, -1e-6),
        ("proportional factor 0", 4, 2, 256, THETA, 0.0, true, EPS),
        ("proportional factor > 1", 4, 2, 256, THETA, 1.5, true, EPS),
    ];
    for (label, q_heads, kv_heads, head_dim, theta, partial_factor, proportional, eps) in rejects {
        assert!(
            call(
                &mut encoder,
                *q_heads,
                *kv_heads,
                *head_dim,
                *theta,
                *partial_factor,
                *proportional,
                *eps
            )
            .is_err(),
            "norm_rope_qkv accepted an invalid request: {label}"
        );
    }
    // A non-proportional request must NOT be rejected for its partial_factor:
    // the factor is unused then, and validating it anyway would reject a
    // perfectly good default-rope config.
    assert!(
        call(&mut encoder, 4, 2, 256, THETA, 0.0, false, EPS).is_ok(),
        "norm_rope_qkv rejected a non-proportional request over its unused partial_factor"
    );
}

#[test]
fn norm_rope_qkv_cache_is_bit_identical_to_separate_append() {
    let device = metal::Device::system_default().expect("no metal device");
    let kernels = ElementKernels::shared().expect("element kernels");
    let queue = device.new_command_queue();

    for case in cases() {
        let q_len = case.q_heads * case.head_dim;
        let kv_len = case.kv_heads * case.head_dim;
        let q_in = pseudo_half(q_len, 0x1111);
        let k_in = pseudo_half(kv_len, 0x2222);
        let v_in = pseudo_half(kv_len, 0x3333);
        let qw = pseudo_half(case.head_dim, 0x4444);
        let kw = pseudo_half(case.head_dim, 0x5555);

        let capacity = 32usize;
        let slot = 3usize;

        // --- Arm 1: Reference (norm_rope_qkv + cache_append) ---
        let q_ref = upload(&device, &q_in);
        let k_ref = upload(&device, &k_in);
        let v_ref = upload(&device, &v_in);
        let qw_buf = upload(&device, &qw);
        let kw_buf = upload(&device, &kw);

        let key_cache_ref = device.new_buffer(
            (capacity * kv_len * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let val_cache_ref = device.new_buffer(
            (capacity * kv_len * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pos_cache_ref = device.new_buffer(
            (capacity * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        kernels
            .encode_norm_rope_qkv(
                &mut encoder,
                &q_ref,
                &k_ref,
                &v_ref,
                &qw_buf,
                &kw_buf,
                case.q_heads,
                case.kv_heads,
                case.head_dim,
                case.offset,
                THETA,
                case.partial_factor,
                case.proportional,
                EPS,
            )
            .expect("norm_rope_qkv");
        kernels
            .encode_cache_append(
                &mut encoder,
                &k_ref,
                &v_ref,
                &key_cache_ref,
                &val_cache_ref,
                &pos_cache_ref,
                1,
                kv_len,
                capacity,
                slot,
                case.offset,
            )
            .expect("cache_append");
        encoder.flush_and_wait().expect("wait");

        // --- Arm 2: Fused norm_rope_qkv_cache ---
        let q_fused = upload(&device, &q_in);
        let raw_k = upload(&device, &k_in);
        let raw_v = upload(&device, &v_in);
        let key_cache_fused = device.new_buffer(
            (capacity * kv_len * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let val_cache_fused = device.new_buffer(
            (capacity * kv_len * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pos_cache_fused = device.new_buffer(
            (capacity * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let mut encoder2 = MetalEncoder::new(&queue).expect("encoder2");
        kernels
            .encode_norm_rope_qkv_cache(
                &mut encoder2,
                &q_fused,
                &raw_k,
                &raw_v,
                &key_cache_fused,
                &val_cache_fused,
                &pos_cache_fused,
                &qw_buf,
                &kw_buf,
                case.q_heads,
                case.kv_heads,
                case.head_dim,
                case.offset,
                THETA,
                case.partial_factor,
                case.proportional,
                EPS,
                slot,
                kv_len,
            )
            .expect("norm_rope_qkv_cache");
        encoder2.flush_and_wait().expect("wait2");

        // --- Bit-exact comparison ---
        let q_ref_bits = download(&q_ref, q_len);
        let q_fused_bits = download(&q_fused, q_len);
        assert_eq!(q_ref_bits, q_fused_bits, "Q bits mismatch for case {}", case.name);

        let k_cache_ref_bits = download(&key_cache_ref, capacity * kv_len);
        let k_cache_fused_bits = download(&key_cache_fused, capacity * kv_len);
        assert_eq!(
            k_cache_ref_bits, k_cache_fused_bits,
            "Key cache mismatch for case {}",
            case.name
        );

        let v_cache_ref_bits = download(&val_cache_ref, capacity * kv_len);
        let v_cache_fused_bits = download(&val_cache_fused, capacity * kv_len);
        assert_eq!(
            v_cache_ref_bits, v_cache_fused_bits,
            "Value cache mismatch for case {}",
            case.name
        );

        let pos_ref = unsafe {
            std::slice::from_raw_parts(pos_cache_ref.contents() as *const u32, capacity)
        };
        let pos_fused = unsafe {
            std::slice::from_raw_parts(pos_cache_fused.contents() as *const u32, capacity)
        };
        assert_eq!(pos_ref, pos_fused, "Position cache mismatch for case {}", case.name);
    }
}

