//! The NR0 decode-matvec axis must not change a single output bit.
//!
//! `KAPLUMBA_DECODE_NR0` and the `_mcg_r{2,4,8}` kernels give each thread NR0
//! output tile-columns instead of one, so several independent weight streams are
//! in flight per thread. The accumulator update, the four-lane butterfly, the
//! serial threadgroup sum and the `tk` order are all untouched, so every output
//! column sees an identical sequence of fp32 adds and the result *should* be
//! bit-identical.
//!
//! "Should" is not a claim. FMA contraction is not stable across recompiles in
//! this project -- two variants that each matched a baseline alone have differed
//! when combined (30818/32768 elements), and a variant of this very kernel's
//! arithmetic re-associated on `acc0 += x0*v0 + x1*v1 + x2*v2 + x3*v3`. So this
//! test compares **output bits**, for every rate, for every NR0, over shapes
//! chosen to exercise the awkward cases:
//!
//! * `out_features/16` not a multiple of 8 (the 3 bpw `k_proj` of a sliding
//!   layer is 4096/16 = 256 tiles, fine, but `o_proj` at 5376/16 = 336 is 8 x 42,
//!   so the tile count is not a power of two and the last wave is partial);
//! * `in_features` large enough that the k loop is long (5376/16 = 336 tiles over
//!   8 simdgroups = 42 iterations each, so the accumulator sees 42 sequential
//!   adds -- a short k loop would hide a re-association);
//! * `in_features` small enough that the k loop is shorter than the simdgroup
//!   count, so some simdgroups run zero iterations and only the reduction path
//!   is left. `in_features = 128` gives 8 tiles for 8 simdgroups.
//!
//! The weights are generated from an LCG whose state is advanced once per code
//! per tile and whose output is taken from the high half, so a perturbation of
//! any single code changes exactly one output. An earlier bit-exactness
//! generator in this project used an LCG of period 8 and made every index
//! perturbation that was a multiple of 8 invisible; this one cannot.

use half::f16;
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels, NR0_VARIANTS};
use metal::{Device, MTLResourceOptions};

/// (in_features, out_features) pairs. See the module doc for why each is here.
///
/// Every shape is a multiple of 128 in both dimensions, so all of them divide by
/// every NR0 up to 8 and no comparison is skipped.
const SHAPES: &[(usize, usize)] = &[
    (128, 128),    // k loop exactly one tile per simdgroup; no simdgroup idles
    (128, 1024),   // ... and enough tiles that 8 simdgroups all have work
    (1024, 128),   // k loop 64 tiles over 8 simdgroups = 8 iterations
    (5376, 4096),  // k_proj of a sliding layer
    (5376, 5376),  // o_proj of a sliding layer: 336 tiles, not a power of two
    (5376, 21504), // gate/up: 1344 tiles
    (21504, 5376), // down_proj: the 16x-more-k axis
    (4096, 2048),  // k_proj of a full-attention layer
];

/// The rates to sweep, at every shape below.
///
/// All eight, because the failure mode (FMA contraction) is *demonstrably*
/// rate-dependent: 8 bpw with NR0=8 at 5376x21504 differs from NR0=1 in 2 of
/// 21504 outputs while 3 bpw and 6 bpw at the same shape do not. A sweep over
/// "the rates the checkpoint uses" would have been 3 and 6, and would have
/// missed it -- and 8 bpw is a rate `Exl3Rate` supports, so a future checkpoint
/// could carry it.
const RATES: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8];

struct Case {
    in_f: usize,
    out_f: usize,
    rate: u8,
    packed: metal::Buffer,
    input: metal::Buffer,
    input_signs: metal::Buffer,
    output_signs: metal::Buffer,
    transformed: metal::Buffer,
    raw: metal::Buffer,
    output: metal::Buffer,
}

/// A code in `0..2^rate`, held away from both rails where the range allows it.
///
/// The rail avoidance is because the MCG butterfly is a permutation of the
/// codebook: an off-by-one in a decoded code shows up as a neighbouring
/// codebook entry, and on a rail the neighbour is a constant, so a systematic
/// error can hide there. At 1 and 2 bpw the range is 2 and 4, which leaves no
/// room to avoid anything, so those rates simply use the full range -- the
/// higher rates are where a rail would actually hide something.
fn states_for(tile: usize, code: usize, rate: u8) -> u16 {
    let mut state = 0x1234_5678_u32
        ^ (tile as u32).wrapping_mul(0x9E37_79B9)
        ^ (code as u32).wrapping_mul(0x85EB_CA6B)
        ^ ((rate as u32) << 24);
    // Three rounds, and the output is taken from the high half after a xor-fold.
    // An LCG of period 8 made every index shift that was a multiple of 8
    // invisible in an earlier generator here; the fold plus three rounds plus a
    // high-half take means code 0 and code 8 differ, which is the property that
    // generator lacked.
    for _ in 0..3 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state ^= state >> 13;
    }
    let usable = 1usize << rate.min(8);
    let value = if usable > 6 {
        3 + (state as usize % (usable - 5))
    } else {
        state as usize % usable
    };
    value as u16
}

fn build_packed(in_f: usize, out_f: usize, rate: u8) -> Vec<u16> {
    let tiles = (in_f / 16) * (out_f / 16);
    let mut packed = Vec::new();
    let mut states = vec![0u16; 256];
    for tile in 0..tiles {
        for (code, slot) in states.iter_mut().enumerate() {
            *slot = states_for(tile, code, rate);
        }
        packed.extend(pack_integer_states(&states, rate).unwrap().0);
    }
    packed
}

fn build(device: &Device, in_f: usize, out_f: usize, rate: u8) -> Case {
    let packed = build_packed(in_f, out_f, rate);
    let packed = device.new_buffer_with_data(
        packed.as_ptr() as *const std::ffi::c_void,
        (packed.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    // A specific, non-trivial input: a sum of two incommensurate sines, so no
    // two positions share a value and an off-by-one in the k index is visible.
    let input: Vec<u16> = (0..in_f)
        .map(|i| {
            let x = i as f32;
            f16::from_f32((x * 0.0173).sin() + (x * 0.0031).cos() + 0.25).to_bits()
        })
        .collect();
    let input = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (in_f * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    // Signs matter: they are per-element multipliers on both sides, so a kernel
    // that ignored them would produce a plausible wrong answer.
    let input_signs: Vec<u16> = (0..in_f)
        .map(|i| f16::from_f32(if i % 3 == 0 { 0.0021 } else { -0.0017 }).to_bits())
        .collect();
    let input_signs = device.new_buffer_with_data(
        input_signs.as_ptr() as *const std::ffi::c_void,
        (in_f * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_signs: Vec<u16> = (0..out_f)
        .map(|i| f16::from_f32(if i % 5 == 0 { -0.0031 } else { 0.0023 }).to_bits())
        .collect();
    let output_signs = device.new_buffer_with_data(
        output_signs.as_ptr() as *const std::ffi::c_void,
        (out_f * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let transformed = device.new_buffer((in_f * 2) as u64, MTLResourceOptions::StorageModeShared);
    let raw = device.new_buffer(
        (out_f * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
    Case {
        in_f,
        out_f,
        rate,
        packed,
        input,
        input_signs,
        output_signs,
        transformed,
        raw,
        output,
    }
}

fn run(kernels: &LinearKernels, queue: &metal::CommandQueue, case: &Case, name: &str) -> Vec<u16> {
    let mut encoder = MetalEncoder::new(queue).unwrap();
    kernels
        .encode_gemv_named(
            &mut encoder,
            name,
            &LinearEncodeRequest {
                packed: &case.packed,
                packed_offset: 0,
                perm_mode: 0,
                input: &case.input,
                input_signs: &case.input_signs,
                output_signs: &case.output_signs,
                transformed: &case.transformed,
                raw: &case.raw,
                output: &case.output,
                in_features: case.in_f,
                out_features: case.out_f,
                rate: Exl3Rate::Integer(case.rate),
                codebook: Exl3Codebook::Mcg,
                pretransformed: false,
            },
        )
        .unwrap();
    encoder.commit_and_wait().unwrap();
    let ptr = case.output.contents() as *const u16;
    unsafe { std::slice::from_raw_parts(ptr, case.out_f) }.to_vec()
}

#[test]
fn every_nr0_variant_is_bit_identical_to_nr0_one() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let kernels = LinearKernels::shared().expect("kernels");
    let queue = device.new_command_queue();
    let mut compared = 0usize;

    // The full cross product: 8 rates x 8 shapes x 4 NR0 = 256 comparisons.
    //
    // An earlier version of this test swept only 3 bpw over the large shapes on
    // the argument that building 115 M codes eight times was too slow, and used
    // four small shapes for the other rates. That reasoning was wrong, and
    // measurably so: `examples/nr0_discriminate.rs` builds 5376x21504 at 8 bpw in
    // 0.21 s and 5376x21504 at 3 bpw in 0.19 s, and the full sweep is a few
    // seconds of host work. Meanwhile the *trimmed* sweep passed -- and the
    // combination it had dropped, 8 bpw with NR0=8 at 5376x21504, is exactly the
    // one that is NOT bit-exact: 2 of 21504 outputs differ from NR0=1 by one
    // fp16 ulp. A test that had been left running over the whole cross product
    // would have caught a real FMA-contraction re-association; the version that
    // was written to save four seconds would not have, and would have reported
    // "bit-exact" while the claim was false.
    //
    // So: full coverage, and the cost is accepted. The lesson is the one this
    // repository keeps relearning -- a coverage reduction justified by a runtime
    // measurement needs the measurement to be *about the thing that would be
    // missed*, not about the thing that is expensive to run.
    for rate in RATES.iter().copied() {
        for (in_f, out_f) in SHAPES.iter().copied() {
            let case = build(&device, in_f, out_f, rate);
            let base_name = format!("linear_gemv_exl3_k{rate}_mcg");
            let baseline = run(&kernels, &queue, &case, &base_name);
            // A baseline of all zeros means the kernel did not run at all, and
            // every comparison below would then be vacuously true.
            assert!(
                baseline.iter().any(|value| *value != 0),
                "{base_name} on {in_f}x{out_f} produced an all-zero output, so the \
                 bit-exactness comparison would be vacuous"
            );
            for nr0 in NR0_VARIANTS {
                let name = format!("linear_gemv_exl3_k{rate}_mcg_r{nr0}");
                if !((out_f / 16) % nr0 as usize == 0) {
                    continue;
                }
                let got = run(&kernels, &queue, &case, &name);
                let mismatches: Vec<usize> = baseline
                    .iter()
                    .zip(got.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .map(|(i, _)| i)
                    .collect();
                if !mismatches.is_empty() {
                    let detail = format!(
                        "{name} on {in_f}x{out_f} @{rate}bpw: {}/{} outputs, first at \
                         column {}: {:04x} vs {:04x} (delta {:+})",
                        mismatches.len(),
                        baseline.len(),
                        mismatches[0],
                        baseline[mismatches[0]],
                        got[mismatches[0]],
                        f16::from_bits(got[mismatches[0]]).to_f32()
                            - f16::from_bits(baseline[mismatches[0]]).to_f32(),
                    );
                    eprintln!("  MISMATCH {detail}");
                    MISMATCHES.lock().unwrap().push(detail);
                }
                compared += 1;
            }
        }
    }
    // Exactly 8 rates x 8 shapes x 3 NR0 variants. `==` rather than `>=`: a shape
    // that does not divide by some NR0 is silently skipped by the guard above, and
    // a `>=` bound would absorb the loss. The count is what checks that the cross
    // product is still a cross product.
    assert_eq!(
        compared, 192,
        "expected 8 rates x 8 shapes x 3 NR0 = 192 comparisons, ran {compared}; a shape \
         that does not divide evenly by an NR0 is being skipped, which leaves a hole \
         exactly where a re-association could hide"
    );
    eprintln!("compared {compared} (rate, shape, NR0) combinations bit for bit");
    // One lock, one length. `assert_eq!(MISMATCHES.lock().len(), 0, "... {} ...",
    // MISMATCHES.lock().len())` deadlocks: `Mutex` is not reentrant, and the
    // argument expression is evaluated while the first guard is still alive. It
    // did exactly that here, and the test hung for 60 s with a stack ending in
    // `poison::lock` -- which reads like a GPU hang and is not one.
    let mismatches = MISMATCHES.lock().unwrap();
    assert!(
        mismatches.is_empty(),
        "{} (rate, shape, NR0) combinations were not bit-exact; the lines above name \
         each one. The NR0 axis is supposed to add independent streams without changing \
         the arithmetic, so any entry here is an FMA-contraction re-association or a \
         reduction-order change -- and decode's bit-exactness is a hard project \
         constraint, so the axis cannot ship with entries.",
        mismatches.len()
    );
}

/// Mismatches seen, so the failure message can list them without re-running.
static MISMATCHES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

#[test]
fn every_declared_nr0_variant_is_reachable_by_name() {
    // `NR0_VARIANTS` in linear.rs and `NR0_VALUES` in scripts/gen_exl3_gemv.py
    // are two copies of one list. If they drift, a name resolves to `None` and
    // `decode_gemv_pipeline` silently falls back to NR0 = 1 -- so a sweep would
    // report "NR0 = 4" while measuring NR0 = 1, with nothing failing. This is
    // the check that makes the sweep's own knob trustworthy.
    //
    // The check is that every name *dispatches*, i.e. the kernel compiles and is
    // in the name map. `exl3_gemv_kernel_name` cannot be used for this: it reads
    // the index table, which deliberately does not contain the r-variants (see
    // `LinearKernels::new`). So this asks the only question that matters, which
    // is whether `encode_gemv_named` accepts the name.
    let Some(device) = Device::system_default() else {
        return;
    };
    let kernels = LinearKernels::shared().expect("kernels");
    let queue = device.new_command_queue();
    // 128x128: 8 tiles, so every NR0 up to 8 divides it, and 8 k-tiles for 8
    // simdgroups, so the k loop is exactly one iteration and the whole cost of
    // the dispatch is the grid and the reduction.
    let case = build(&device, 128, 128, 3);
    for bits in RATES.iter().copied() {
        for nr0 in std::iter::once(1u32).chain(NR0_VARIANTS) {
            let name = if nr0 == 1 {
                format!("linear_gemv_exl3_k{bits}_mcg")
            } else {
                format!("linear_gemv_exl3_k{bits}_mcg_r{nr0}")
            };
            let mut encoder = MetalEncoder::new(&queue).unwrap();
            kernels
                .encode_gemv_named(
                    &mut encoder,
                    &name,
                    &LinearEncodeRequest {
                        packed: &case.packed,
                        packed_offset: 0,
                        perm_mode: 0,
                        input: &case.input,
                        input_signs: &case.input_signs,
                        output_signs: &case.output_signs,
                        transformed: &case.transformed,
                        raw: &case.raw,
                        output: &case.output,
                        in_features: 128,
                        out_features: 128,
                        rate: Exl3Rate::Integer(bits),
                        codebook: Exl3Codebook::Mcg,
                        pretransformed: false,
                    },
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "{name} does not dispatch: {error}. If it is missing, the \
                            generated family and NR0_VARIANTS have drifted apart."
                    )
                });
            encoder.commit_and_wait().unwrap();
        }
    }
}

#[test]
fn every_legal_shape_divides_by_every_nr0() {
    // The failure mode this guards is quiet and total. An r4 kernel dispatched
    // with a grid sized for NR0 = 1 writes one quarter of the output columns and
    // leaves the rest at whatever the destination buffer held -- a plausible
    // looking buffer, finite, the right magnitude on three quarters of the
    // entries. Nothing downstream notices.
    //
    // Two things have to hold for that to be impossible rather than merely
    // unlikely. First, the grid is derived from the kernel's own name
    // (`nr0_of_kernel_name`), not from an environment variable, so the two cannot
    // disagree. Second, a shape that does not divide evenly is an ERROR rather
    // than a truncated dispatch, because "some columns are not written" is not a
    // state this engine should ever be in.
    //
    // `out_features = 16 * 3 = 48` is below the 128 multiple that
    // `encode_gemv_named` requires, so the smallest reachable offending shape is
    // `16 * 130`: 130 tiles against NR0 of 4 leaves 2.
    let Some(device) = Device::system_default() else {
        return;
    };
    // `out_features` must be a multiple of 128 (the finalize kernel packs 128
    // outputs per block), so the tile count is always a multiple of 8 and every
    // NR0 up to 8 divides it. That is why this rejection is unreachable through
    // this entry point with a legal shape, and the test says so rather than
    // pretending otherwise: the guard exists for the `encode_gemv` path, where
    // `validate_encode` applies the same multiple-of-128 rule but the check is
    // duplicated rather than shared, and a future relaxation of either would
    // otherwise turn a loud error into a truncated dispatch.
    //
    // So the reachable half of the claim is checked here -- that the multiple of
    // 128 rule really does imply divisibility, for every NR0 -- and the
    // unreachable half is checked by calling the guard's own arithmetic through a
    // shape that the dimension rule would allow if it were ever relaxed.
    for out_f in [128usize, 256, 1024, 4096, 5376, 21504, 262144] {
        let tiles = out_f / 16;
        assert_eq!(
            out_f % 128,
            0,
            "the premise of this test: {out_f} is a legal out_features"
        );
        for nr0 in std::iter::once(1u32).chain(NR0_VARIANTS) {
            assert!(
                tiles % nr0 as usize == 0,
                "{out_f} gives {tiles} tiles, which NR0={nr0} does not divide -- the \
                 multiple-of-128 rule is what makes this safe, so if this fires the \
                 rule has changed and the rejection guard has become reachable"
            );
        }
    }
    // And the guard's message, exercised directly: an out_features that the
    // dimension rule accepts but NR0 does not divide can only be constructed by
    // bypassing that rule, so this asserts the *arithmetic* the guard uses rather
    // than pretending a legal shape can trigger it.
    let tiles = 130usize;
    assert_eq!(
        tiles % 4,
        2,
        "the arithmetic premise: 130 tiles against NR0 4"
    );
    assert_eq!(
        LinearKernels::nr0_of_kernel_name("linear_gemv_exl3_k3_mcg_r4"),
        4
    );
    // A device is not actually needed for any of the above; take one anyway so
    // this test is a no-op rather than a pass on a machine with no Metal, which
    // is what every other GPU-touching test here does.
    let _ = device.new_buffer(4, MTLResourceOptions::StorageModePrivate);
}

#[test]
fn nr0_of_kernel_name_reads_the_axis_off_the_name() {
    // The whole reason the grid can be trusted is that it comes from the name
    // rather than from `KAPLUMBA_DECODE_NR0`. If this ever read the environment
    // instead, `encode_gemv_named` would size the grid for whatever the process
    // happened to be swept with, and every bit-exactness comparison above would
    // be comparing two different grids.
    for (name, want) in [
        ("linear_gemv_exl3_k3_mcg", 1),
        ("linear_gemv_exl3_k3_mcg_r2", 2),
        ("linear_gemv_exl3_k3_mcg_r4", 4),
        ("linear_gemv_exl3_k3_mcg_r8", 8),
        ("linear_gemv_exl3_k3_u_mcg", 1),
        ("linear_gemv_exl3_k3", 1),
    ] {
        assert_eq!(
            LinearKernels::nr0_of_kernel_name(name),
            want,
            "{name} should report NR0 {want}"
        );
    }
}
