//! Is the *live* decode kernel bit-exact against the reference?
//!
//! The engine dispatches `linear_gemv_ishizuki_k3_colmajor_contig4` (measured with
//! `KAPLUMBA_TRACE_GEMV=1`). `repacked_golden.rs` asserts `to_bits()` equality for
//! `_colmajor` and `_colmajor_simd`, but **never mentions `contig4`** -- the kernel
//! that actually runs. So the engine's output is currently unverified.
//!
//! `contig4` starts each simdgroup at `tk = sg * 4u` and reads four *contiguous*
//! tiles; `_colmajor` starts at `tk = sg` and reads tiles at `tk + {0,SG,2SG,3SG}`.
//! Both cover the same tiles in the same ascending order, but they assign them to
//! different accumulators, so they are a reassociation of the same float sum. Float
//! addition is not associative, so this is not expected to be bit-exact -- and the
//! spec's no-quality-loss rule does not permit it silently.
//!
//! This measures the deviation on the real shapes so the decision is made on numbers.

use half::f16;
use kaplumba_engine::exl3::{pack_integer_states, reorder_tensor_to_colmajor};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Buffer, Device, MTLResourceOptions};

const BITS: u8 = 3;

fn build(in_f: usize, out_f: usize, seed: u32) -> Vec<u16> {
    let tiles = (in_f / 16) * (out_f / 16);
    let mut packed = Vec::with_capacity(tiles * 16 * BITS as usize);
    let mut s = seed;
    for _ in 0..tiles {
        let mut states = [0u16; 256];
        for st in &mut states {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *st = (s & ((1u32 << BITS) - 1)) as u16;
        }
        let (w, _) = pack_integer_states(&states, BITS).expect("pack");
        packed.extend(w);
    }
    packed
}

fn upload(device: &Device, v: &[u16]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[test]
fn live_kernel_contig4_is_bit_exact_against_colmajor() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let kernels = match LinearKernels::new() {
        Ok(k) => k,
        Err(e) => panic!("LinearKernels error: {e:?}"),
    };
    let queue = device.new_command_queue().to_owned();

    // The shapes that carry real weight: the two biggest per-token consumers.
    let cases: &[(usize, usize, &str)] = &[
        (21504, 5376, "down_proj  21504x5376"),
        (5376, 21504, "gate_proj  5376x21504"),
        (4096, 5376, "o_proj     4096x5376"),
    ];

    for (idx, (in_f, out_f, label)) in cases.iter().enumerate() {
        let row = build(*in_f, *out_f, 13 + idx as u32);
        let col = reorder_tensor_to_colmajor(&row, *in_f, *out_f).expect("colmajor");
        let packed = upload(&device, &col);

        let input_v: Vec<u16> = (0..*in_f)
            .map(|i| f16::from_f32((i as f32 * 0.011).sin()).to_bits())
            .collect();
        let ones = |n: usize| upload(&device, &vec![f16::from_f32(1.0).to_bits(); n]);

        let raw_ref = device.new_buffer((out_f * 4 * 4) as u64, MTLResourceOptions::StorageModeShared);
        let raw_live = device.new_buffer((out_f * 4 * 4) as u64, MTLResourceOptions::StorageModeShared);
        let raw_sk2 = device.new_buffer((out_f * 4 * 4) as u64, MTLResourceOptions::StorageModeShared);
        let raw_2col = device.new_buffer((out_f * 4 * 4) as u64, MTLResourceOptions::StorageModeShared);
        let raw_4col = device.new_buffer((out_f * 4 * 4) as u64, MTLResourceOptions::StorageModeShared);
        let raw_2col_sk2 = device.new_buffer((out_f * 4 * 4) as u64, MTLResourceOptions::StorageModeShared);
        let raw_2col_sk4 = device.new_buffer((out_f * 4 * 4) as u64, MTLResourceOptions::StorageModeShared);
        let out_ref = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let out_live = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let out_sk2 = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let out_2col = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let out_4col = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let out_2col_sk2 = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let out_2col_sk4 = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);

        let run = |kernel: &str, raw: &Buffer, out: &Buffer| -> (Vec<u32>, Vec<u16>) {
            let mut e = MetalEncoder::new(&queue).expect("encoder");
            kernels
                .encode_gemv_named(
                    &mut e,
                    kernel,
                    &LinearEncodeRequest {
                        packed: &packed,
                        packed_offset: 0,
                        perm_mode: 1,
                        input: &upload(&device, &input_v),
                        input_signs: &ones(*in_f),
                        output_signs: &ones(*out_f),
                        transformed: &device.new_buffer(
                            (in_f * 2) as u64,
                            MTLResourceOptions::StorageModeShared,
                        ),
                        raw,
                        output: out,
                        in_features: *in_f,
                        out_features: *out_f,
                        rate: Exl3Rate::Integer(BITS),
                        codebook: Exl3Codebook::Mcg,
                        pretransformed: false,
                    },
                )
                .unwrap_or_else(|err| panic!("{kernel} {label}: {err}"));
            e.commit_and_wait().expect("wait");
            let p = raw.as_ref().contents() as *const f32;
            let f = (0..*out_f).map(|i| unsafe { p.add(i).read() }.to_bits()).collect();
            let q = out.as_ref().contents() as *const u16;
            let h = (0..*out_f).map(|i| unsafe { q.add(i).read() }).collect();
            (f, h)
        };

        let (ref_f32, ref_f16) = run("linear_gemv_ishizuki_k3_colmajor", &raw_ref, &out_ref);
        let (live_f32, live_f16) =
            run("linear_gemv_ishizuki_k3_colmajor_contig4", &raw_live, &out_live);
        let (_sk2_f32, sk2_f16) = run(
            "linear_gemv_ishizuki_k3_colmajor_contig4_sk2",
            &raw_sk2,
            &out_sk2,
        );

        assert!(
            ref_f32.iter().any(|&v| v != 0),
            "{label}: reference produced all zeros"
        );

        // ASSERT on the fp16 output, which is what `argmax_logits` reads and the
        // only thing that can change a generated token. The f32 accumulator is
        // allowed to differ -- `contig4` reassociates the sum -- but the rounding
        // to fp16 must absorb it completely, or greedy decode can diverge.
        assert_eq!(
            ref_f16, live_f16,
            "{label}: the live kernel's fp16 output differs from the reference in \
             {} of {} elements, so greedy decode is not bit-exact",
            ref_f16.iter().zip(&live_f16).filter(|(a, b)| a != b).count(),
            ref_f16.len()
        );

        let sk2_diff = ref_f16.iter().zip(&sk2_f16).filter(|(a, b)| a != b).count();
        println!("{label}: sk2 fp16 diff count: {sk2_diff}/{}", ref_f16.len());
        assert_eq!(
            ref_f16, sk2_f16,
            "{label}: sk2 fp16 output differs from reference in {sk2_diff} elements"
        );

        let (_2col_f32, _2col_f16) = run(
            "linear_gemv_ishizuki_k3_colmajor_2col",
            &raw_2col,
            &out_2col,
        );
        let _2col_diff = ref_f16.iter().zip(&_2col_f16).filter(|(a, b)| a != b).count();
        println!("{label}: 2col fp16 diff count: {_2col_diff}/{}", ref_f16.len());
        assert_eq!(
            ref_f16, _2col_f16,
            "{label}: 2col fp16 output differs from reference in {_2col_diff} elements"
        );

        let (_4col_f32, _4col_f16) = run(
            "linear_gemv_ishizuki_k3_colmajor_4col",
            &raw_4col,
            &out_4col,
        );
        let _4col_diff = ref_f16.iter().zip(&_4col_f16).filter(|(a, b)| a != b).count();
        println!("{label}: 4col fp16 diff count: {_4col_diff}/{}", ref_f16.len());
        assert_eq!(
            ref_f16, _4col_f16,
            "{label}: 4col fp16 output differs from reference in {_4col_diff} elements"
        );

        let (_2col_sk2_f32, _2col_sk2_f16) = run(
            "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            &raw_2col_sk2,
            &out_2col_sk2,
        );
        let _2col_sk2_diff = ref_f16.iter().zip(&_2col_sk2_f16).filter(|(a, b)| a != b).count();
        println!("{label}: 2col_sk2 fp16 diff count: {_2col_sk2_diff}/{}", ref_f16.len());
        assert_eq!(
            ref_f16, _2col_sk2_f16,
            "{label}: 2col_sk2 fp16 output differs from reference in {_2col_sk2_diff} elements"
        );

        let (_2col_sk4_f32, _2col_sk4_f16) = run(
            "linear_gemv_ishizuki_k3_colmajor_2col_sk4",
            &raw_2col_sk4,
            &out_2col_sk4,
        );
        let _2col_sk4_diff = ref_f16.iter().zip(&_2col_sk4_f16).filter(|(a, b)| a != b).count();
        println!("{label}: 2col_sk4 fp16 diff count: {_2col_sk4_diff}/{}", ref_f16.len());
        assert_eq!(
            ref_f16, _2col_sk4_f16,
            "{label}: 2col_sk4 fp16 output differs from reference in {_2col_sk4_diff} elements"
        );

        let first = ref_f32.iter().zip(&live_f32).position(|(a, b)| a != b);
        let differing = first.map_or(0, |_| {
            ref_f32.iter().zip(&live_f32).filter(|(a, b)| a != b).count()
        });
        let max_ulp = ref_f32
            .iter()
            .zip(&live_f32)
            .map(|(a, b)| (*a as i64 - *b as i64).abs())
            .max()
            .unwrap_or(0);

        // fp16 output is what the sampler actually sees, so that is the number
        // that decides whether greedy decode can change.
        let f16_differing = ref_f16.iter().zip(&live_f16).filter(|(a, b)| a != b).count();
        let f16_max_ulp = ref_f16
            .iter()
            .zip(&live_f16)
            .map(|(a, b)| (*a as i64 - *b as i64).abs())
            .max()
            .unwrap_or(0);

        println!(
            "{label}: f32 differs at {first:?} ({differing}/{} outputs, max {max_ulp} ULP); \
             f16 differs on {f16_differing}/{} (max {f16_max_ulp} ULP)",
            ref_f32.len(),
            ref_f16.len()
        );
    }
}
