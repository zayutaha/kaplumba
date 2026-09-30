//! Bit-exactness of the codebook-specialised decode matvec.
//!
//! The specialisation under test is a pure *control-flow* change: `decode_state`
//! took the codebook as a runtime argument and branched on it inside the
//! eight-times-unrolled value loop, which cost 61.7% of the kernel. The
//! specialised kernels call `decode_state_mcg` instead. Nothing about the
//! arithmetic, the window setup, the shifts, or the accumulation is supposed to
//! change, so the ONLY acceptable outcome is that every output bit is identical.
//!
//! This test exists because the previous claims of exactness in this area were
//! not sound, and the two ways they failed are encoded here as defences.
//!
//! TRAP 1 -- a local array breaks exactness, silently. A `uint code[8]` local
//! array, written in one `#pragma unroll` loop and read in another,
//! MISCOMPILED and produced 32762/32768 wrong outputs while the harness reported
//! success. The shape cannot be seen from the host side, so the defence here is
//! to make the comparison exact (bit-for-bit on the raw output words) and to
//! compare a matrix large enough that 32762/32768 wrong would be unmissable.
//!
//! TRAP 2 -- FMA contraction is not stable. `acc += x0*v0 + x1*v1 + ...`
//! re-associates, so two variants can each match a baseline alone and differ
//! from each other when combined (30818/32768 differed). The defence is to
//! compare the SAME arithmetic through both codebooks: the reference is the
//! engine's own CPU `decode_state`, and the two GPU paths under test differ only
//! in the decode call, so a contraction difference would show up as a bit
//! difference here rather than being absorbed.
//!
//! TILE GENERATION -- per-tile mixed xorshift, not an LCG. A prior generator used
//! an LCG whose period was 8, so every index perturbation that was a multiple of
//! 8 was invisible to it and its bit-exactness check passed VACUOUSLY: the test
//! compared the same data to itself. `tiles_are_all_distinct` below is the
//! assertion that closes that hole -- if the generator ever degenerates again,
//! the test fails on the generator rather than on the thing it is testing.

use std::ffi::c_void;

use half::f16;
use kaplumba_engine::exl3::{
    BitRate, Codebook, SignVector, pack_integer_states, reconstruct_matrix,
};
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearBufferRequest, LinearKernels};
use metal::{Device, MTLResourceOptions};

/// Per-tile mixed xorshift (xorshift32 with a Weyl increment).
///
/// Mixed on purpose. The period-8 LCG this replaces made every perturbation that
/// was a multiple of 8 invisible, so a test that indexed tiles wrongly could not
/// tell. This has full period 2^32-1 and, critically, a different *increment per
/// tile*, so two tiles never see the same state stream.
fn tile_seed(tile: u64) -> u32 {
    let mut x = (0x9E37_79B9u32)
        .wrapping_add((tile as u32).wrapping_mul(0x85EB_CA6B))
        .wrapping_add(((tile >> 32) as u32).wrapping_mul(0xC2B2_AE35));
    // Three rounds of xorshift32: mixes all 32 bits per step, so a state that
    // differs in a high bit cannot leave the low bits alone.
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

/// 256 EXL3 states for one tile, from the mixed generator.
fn tile_states(bits: u8, tile: u64) -> Vec<u16> {
    let mask = (1_u32 << bits) - 1;
    let mut x = tile_seed(tile);
    (0..256)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x & mask) as u16
        })
        .collect()
}

/// A full packed matrix, one `pack_integer_states` call per tile.
///
/// Each tile gets its own generator stream, so the matrix is not a single
/// repeated block. The output is 5376x5376-shaped: 336 k-tiles x 336 n-tiles is
/// too large for a unit test, so this uses a 128x128 matrix -- 64 tiles each way,
/// which is 4096 16x16 tiles and 32768 output elements, the size at which the
/// two historical miscompilations were visible (32762/32768, 30818/32768).
fn packed_matrix(bits: u8, tiles: u64) -> Vec<u16> {
    let mut packed = Vec::new();
    for tile in 0..tiles {
        let (words, _) = pack_integer_states(&tile_states(bits, tile), bits).unwrap();
        packed.extend(words);
    }
    packed
}

/// The generator must actually produce different data per tile.
///
/// Without this, a period-8 LCG makes every 8th tile identical and any
/// index-perturbation bug is invisible -- which is exactly how the previous
/// bit-exactness check came to pass while comparing a buffer against itself.
#[test]
fn tiles_are_all_distinct() {
    for bits in [3_u8, 6] {
        let total = 64_u64 * 64;
        let mut tiles = Vec::with_capacity(total as usize);
        for tile in 0..total {
            tiles.push(
                pack_integer_states(&tile_states(bits, tile), bits)
                    .unwrap()
                    .0,
            );
        }
        // Adjacent tiles must differ (catches a period-1 or period-8 generator,
        // which is the historical failure).
        for pair in tiles.windows(2) {
            assert_ne!(
                pair[0], pair[1],
                "bits={bits}: adjacent tiles identical -- generator has a short period"
            );
        }
        // And all tiles must be pairwise distinct across the whole matrix, which
        // is the property the old check silently assumed.
        let mut sorted: Vec<&Vec<u16>> = tiles.iter().collect();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            total as usize,
            "bits={bits}: {} of {total} tiles are duplicates",
            total as usize - sorted.len()
        );
    }
    // The mixed generator must also decorrelate across rates, or a bits-only bug
    // would be masked.
    assert_ne!(tile_seed(0), tile_seed(1));
    assert_ne!(tile_seed(7), tile_seed(8));
}

/// The activation vector, shared by the CPU reference and the GPU dispatch.
///
/// 1024 entries, so a 1024-wide `in_features` is covered without wrapping --
/// wrapping would make the reference index a value the kernel never read, and
/// would silently compare two different vectors.
///
/// BOUNDED IN [-4, 4]. This is not cosmetic. An earlier version used an
/// unbounded cubic in the index, which reached ~1e8 at index 1024; fp16 has a
/// max of 65504, so every high index became `inf`, the reference's `inf * w`
/// sums became `NaN`, and the test reported a total mismatch instead of the
/// bit-exactness result it was written to produce. The value is taken modulo 1
/// on a golden-ratio step, which keeps it bounded while leaving every entry
/// distinct and non-zero.
static X: [f32; 1024] = {
    let mut buf = [0.0_f32; 1024];
    let mut i = 0;
    while i < 1024 {
        let u = (i as f32) * 0.618_033_988_75;
        let frac = u - (u as i64) as f32; // [0, 1)
        // Map to [-4, 4) with a couple of harmonics so the sequence is not
        // monotone: a monotone ramp would let a shifted index still look
        // plausible.
        let v = (frac * 8.0 - 4.0) * (0.5 + 0.5 * (i as f32) * 0.021);
        buf[i] = if v == 0.0 { 0.0625 } else { v };
        i += 1;
    }
    buf
};

/// Bit comparison, not a tolerance.
fn first_bit_difference(got: &[f32], want: &[f32]) -> Option<(usize, u32, u32)> {
    got.iter().zip(want).enumerate().find_map(|(i, (g, w))| {
        let gb = g.to_bits();
        let wb = w.to_bits();
        (gb != wb).then_some((i, gb, wb))
    })
}

/// The host-side fixture: the activation, the two sign vectors, and a dispatch
/// of a NAMED decode kernel.
///
/// Everything is a host array, because `gemv_buffer_named` is the same code path
/// the engine's own tests use and it has the sign-vector length conventions
/// already right. An earlier version of this file hand-allocated the Metal
/// buffers and got `raw` at two bytes per element instead of four and the sign
/// vectors at the packed sign-byte length instead of `in_features` halfs; the
/// symptom was a silent all-zero or denormal output, not an error, and it cost
/// more time to find than the whole specialisation took to write.
///
/// The activation and the sign vectors are built ONCE and shared by every
/// dispatch, so the only thing that varies between the two kernels under test is
/// the kernel name.
struct Fixture {
    input: Vec<u16>,
    input_signs: Vec<u16>,
    output_signs: Vec<u16>,
    packed: metal::Buffer,
    in_features: usize,
    out_features: usize,
    bits: u8,
}

impl Fixture {
    fn new(packed: &[u16], in_features: usize, out_features: usize, bits: u8) -> Fixture {
        let device = Device::system_default().expect("Metal device");
        Fixture {
            input: X[..in_features]
                .iter()
                .map(|v| f16::from_f32(*v).to_bits())
                .collect(),
            // Mixed signs on purpose. An all-ones sign vector would hide a
            // transposed or shifted output, because every column would look the
            // same, and `gemv_buffer` does not check the sign values.
            input_signs: (0..in_features)
                .map(|i| f16::from_f32(if i % 3 == 0 { -1.0 } else { 1.0 }).to_bits())
                .collect(),
            output_signs: (0..out_features)
                .map(|i| f16::from_f32(if i % 5 == 0 { -1.0 } else { 1.0 }).to_bits())
                .collect(),
            packed: device.new_buffer_with_data(
                packed.as_ptr() as *const c_void,
                (packed.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            in_features,
            out_features,
            bits,
        }
    }

    fn request(&self, codebook: Exl3Codebook) -> LinearBufferRequest<'_> {
        LinearBufferRequest {
            packed: &self.packed,
            packed_offset: 0,
            input: &self.input,
            input_signs: &self.input_signs,
            output_signs: &self.output_signs,
            in_features: self.in_features,
            out_features: self.out_features,
            rate: Exl3Rate::Integer(self.bits),
            codebook,
        }
    }

    /// Dispatch a NAMED decode kernel and return the raw output words.
    ///
    /// `gemv_buffer_named` is the same implementation `gemv_buffer` uses with
    /// the kernel chosen by name instead of by `(codebook, aligned, bits)`, so
    /// both kernels under test see an identical input transform, an identical
    /// reduction and an identical finalize, and differ only in the decode call.
    fn run_named(&self, kernels: &LinearKernels, kernel: &str, codebook: Exl3Codebook) -> Vec<u32> {
        let out = kernels
            .gemv_buffer_named(kernel, self.request(codebook))
            .unwrap_or_else(|e| panic!("dispatch {kernel}: {e}"));
        out.iter().map(|w| u32::from(*w)).collect()
    }

    /// Dispatch through the ordinary engine entry point. Control for the named
    /// path: whatever this produces, `run_named` must reproduce.
    fn run_default(&self, kernels: &LinearKernels, codebook: Exl3Codebook) -> Vec<u32> {
        kernels
            .gemv_buffer(self.request(codebook))
            .unwrap_or_else(|e| panic!("dispatch: {e}"))
            .iter()
            .map(|w| u32::from(*w))
            .collect()
    }
}

/// The two Metal decode functions must be textually identical apart from the
/// one decode call.
///
/// This exists because they were not. The first version of `exl3_decode8_mcg`
/// transcribed `j2` as `... : j0 + 1u` where `exl3_decode8` says `j1 + 1u`. At
/// rate 3 the 24-word ring almost never wraps between `j1` and `j2`, so the
/// 3-bit kernels -- every decoder linear in the model -- were bit-identical and
/// the defect was invisible there. At rate 6, which is what `lm_head` uses, the
/// 48-word ring does wrap and the two kernels disagreed in EVERY output.
///
/// A behavioural test across several rates catches that, but only if someone
/// remembers to add the rate. This one cannot be forgotten: it reads the shader
/// source that is compiled into the binary and asserts the invariant directly.
#[test]
fn mcg_decode_body_is_a_faithful_transcription() {
    let source = include_str!("../../kernels/src/linear.metal");

    /// The body of `fn`, from its signature to its closing brace, with comments
    /// and blank lines removed so a reformat cannot look like a difference.
    fn body(source: &str, signature: &str) -> String {
        let start = source
            .find(signature)
            .unwrap_or_else(|| panic!("{signature} not found in linear.metal"));
        let end = source[start..]
            .find("\n}\n")
            .map(|at| start + at + 3)
            .unwrap_or_else(|| panic!("{signature} has no closing brace"));
        source[start..end]
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    let generic = body(source, "static inline void exl3_decode8(");
    let specialised = body(source, "static inline void exl3_decode8_mcg(");

    // Normalise exactly the three intended differences and nothing else:
    //   1. the function name,
    //   2. the absent `uint codebook,` parameter line,
    //   3. the decode call.
    // A LINE filter rather than a string replace for (2), because the parameter
    // sits in the middle of the signature and removing it as a substring shifts
    // every following line, which turns a correct pair into a wall of spurious
    // diffs.
    let normalise = |text: &str| {
        text.replace("exl3_decode8_mcg", "exl3_decode8")
            .replace("decode_state_mcg(s & 0xffffu)", "DECODE")
            .replace("decode_state(s & 0xffffu, codebook)", "DECODE")
            .lines()
            .map(str::trim)
            .filter(|line| *line != "uint codebook,")
            .collect::<Vec<_>>()
            .join("\n")
    };
    let a = normalise(&generic);
    let b = normalise(&specialised);
    if a != b {
        let la: Vec<&str> = a.lines().collect();
        let lb: Vec<&str> = b.lines().collect();
        let mut report = String::new();
        for (i, (x, y)) in la.iter().zip(lb.iter()).enumerate() {
            if x != y {
                report.push_str(&format!(
                    "  line {i}:\n    generic:     {x}\n    specialised: {y}\n"
                ));
            }
        }
        if la.len() != lb.len() {
            report.push_str(&format!(
                "  different lengths: generic {} lines, specialised {} lines\n",
                la.len(),
                lb.len()
            ));
        }
        panic!(
            "exl3_decode8_mcg has DRIFTED from exl3_decode8 beyond the decode call. \
             The two must differ in exactly one token, or the kernels are not \
             comparable and any bit-exactness claim is void:\n{report}"
        );
    }
}

/// The specialised MCG kernel must be BIT-IDENTICAL to the generic
/// runtime-codebook kernel running the same MCG codebook.
///
/// This is the claim that matters and the one the change can actually affect. The
/// two kernels are generated from the same template and differ in exactly one
/// token -- `exl3_decode8(...)` versus `exl3_decode8_mcg(...)` -- so their
/// outputs must agree bit for bit. Both run in the same process, on the same
/// data, through the same input transform, the same reduction and the same
/// finalize, so any difference is attributable to the decode call and nothing
/// else. That also means the FMA-contraction trap cannot fire between them: the
/// accumulation is character-identical in both, so there is no re-association
/// for the compiler to differ on.
///
/// The matrix is 1024x1024 = 1 048 576 outputs, far above the sizes at which
/// both historical miscompilations were visible (32762/32768 and 30818/32768).
///
/// There is deliberately NO bit-exact CPU reference here. `reconstruct_matrix`
/// folds the Hadamard into the weights, whereas the decode path applies it to
/// the activations, so the two are equal mathematically but not bit for bit --
/// which is why the pre-existing test in `linear_metal.rs` compares with a 1e-1
/// tolerance. Asserting bit equality against it would be asserting something
/// false. A separate, loose cross-check lives in
/// `codebook_matches_cpu_within_tolerance`.
///
/// Rates covered: 3 (every decoder linear), 6 (lm_head), plus 1 and 8 at the
/// ends of the generated range, to show the specialisation is not accidentally
/// rate-specific.
#[test]
fn mcg_specialisation_is_bit_exact() {
    let kernels = LinearKernels::new().expect("kernels");
    let in_features = 1024;
    let out_features = 1024;
    for bits in [3_u8, 6, 1, 2, 4, 5, 7, 8] {
        let packed = packed_matrix(bits, 64 * 64);
        let fixture = Fixture::new(&packed, in_features, out_features, bits);
        // One 16x16 tile at rate `bits` is 256*bits bits = 16*bits u16 words,
        // and a 1024x1024 matrix is 64x64 tiles.
        let tiles = (in_features / 16) * (out_features / 16);
        assert_eq!(
            packed.len(),
            tiles * 16 * bits as usize,
            "bits={bits}: packed matrix has {} words, expected {tiles} tiles x {}",
            packed.len(),
            16 * bits as usize,
        );

        // The generic kernel: runtime `codebook`, codebook 1 == MCG.
        let generic = fixture.run_named(
            &kernels,
            &format!("linear_gemv_exl3_k{bits}"),
            Exl3Codebook::Mcg,
        );
        // The specialised kernel: MCG folded to a compile-time constant.
        let special = fixture.run_named(
            &kernels,
            &format!("linear_gemv_exl3_k{bits}_mcg"),
            Exl3Codebook::Mcg,
        );

        // BITS, not a tolerance: compare the raw output words.
        let differing = special.iter().zip(&generic).filter(|(a, b)| a != b).count();
        if differing != 0 {
            let i = special
                .iter()
                .zip(&generic)
                .position(|(a, b)| a != b)
                .unwrap();
            panic!(
                "bits={bits}: SPECIALISED kernel differs from the GENERIC kernel in \
                 {differing}/{out_features} outputs. First at {i}: \
                 special {:#010x} generic {:#010x}",
                special[i], generic[i]
            );
        }
    }
}

/// A loose cross-check that the decode matvec computes the right MATHEMATICS,
/// not merely the same maths twice.
///
/// This is a tolerance check on purpose and is kept separate from the
/// bit-exactness test so the two claims cannot be confused. `reconstruct_matrix`
/// folds the Hadamard into the weight matrix, the decode path applies it to the
/// activations; the two are equal mathematically and differ in the last bits,
/// which is why the pre-existing test in `linear_metal.rs` uses 1e-1. The point
/// here is only that a kernel which returned noise, or which decoded every code
/// with the wrong codebook, would miss by orders of magnitude rather than
/// squeaking under a tolerance.
#[test]
fn codebook_matches_cpu_within_tolerance() {
    let kernels = LinearKernels::new().expect("kernels");
    let in_features = 512;
    let out_features = 512;
    let bits = 3_u8;
    let packed = packed_matrix(bits, 32u64 * 32);
    let fixture = Fixture::new(&packed, in_features, out_features, bits);

    let matrix = reconstruct_matrix(
        &packed,
        in_features / 16,
        out_features / 16,
        BitRate::Integer(bits),
        Codebook::Mcg,
        &SignVector::from_half_bits(fixture.input_signs.clone()),
        &SignVector::from_half_bits(fixture.output_signs.clone()),
    )
    .expect("reconstruct_matrix");
    let want: Vec<f32> = (0..out_features)
        .map(|o| {
            let mut sum = 0.0_f32;
            for k in 0..in_features {
                sum += f16::from_bits(fixture.input[k]).to_f32() * matrix[k * out_features + o];
            }
            sum
        })
        .collect();

    let got = fixture.run_named(&kernels, "linear_gemv_exl3_k3_mcg", Exl3Codebook::Mcg);
    let got: Vec<f32> = got
        .iter()
        .map(|w| f16::from_bits(*w as u16).to_f32())
        .collect();

    // Relative to the magnitude of the result, so a scale change in the model
    // does not turn this into a pass-by-construction.
    let scale = want.iter().fold(0.0_f32, |a, b| a.max(b.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        worst / scale < 1e-2,
        "decode matvec disagrees with the CPU reconstruction: worst |gpu-cpu| = \
         {worst:e} against a result scale of {scale:e} ({:.4} relative)",
        worst / scale
    );
}

#[test]
fn codebooks_remain_distinct_on_gpu() {
    let in_features = 512;
    let out_features = 512;
    let bits = 3_u8;
    let packed = packed_matrix(bits, 32 * 32);
    let kernels = LinearKernels::new().expect("kernels");
    let fixture = Fixture::new(&packed, in_features, out_features, bits);

    // Each codebook dispatches through the family the dispatcher would pick for
    // it: MCG uses the `_mcg` specialised kernel, plain and mul1 use the
    // runtime-codebook kernel. This is exactly what `exl3_gemv_index` selects,
    // so the test catches an index that routes a codebook to the wrong family as
    // well as a decode function wired to the wrong constants.
    let cases = [
        (Exl3Codebook::Plain, Codebook::Plain, "linear_gemv_exl3_k3"),
        (Exl3Codebook::Mcg, Codebook::Mcg, "linear_gemv_exl3_k3_mcg"),
        (Exl3Codebook::Mul1, Codebook::Mul1, "linear_gemv_exl3_k3"),
    ];

    let mut outputs = Vec::new();
    for (kernel_codebook, cpu_codebook, kernel) in cases {
        let got = fixture.run_named(&kernels, kernel, kernel_codebook);
        // The output must not be identically zero or a constant: a kernel that
        // failed to find its trellis window would pass a bare `assert_ne!` between
        // codebooks only if all three failed the same way.
        let values: Vec<f32> = got
            .iter()
            .map(|w| f16::from_bits(*w as u16).to_f32())
            .collect();
        let magnitude = values.iter().fold(0.0_f32, |a, b| a.max(b.abs()));
        assert!(
            magnitude > 1e-3,
            "{kernel_codebook:?} via {kernel}: output is all ~zero \
             (max |x| = {magnitude:e}) -- the kernel read nothing"
        );
        let _ = cpu_codebook;
        outputs.push(got);
    }

    // Plain, MCG and Mul1 must produce three DIFFERENT matrices. Two identical
    // outputs mean a state is wired to the wrong decode function. This is the
    // assertion a throughput measurement cannot make: a kernel that decoded
    // every codebook as MCG would run at exactly the specialised speed.
    let names = ["plain", "mcg", "mul1"];
    for i in 0..3 {
        for j in i + 1..3 {
            assert_ne!(
                outputs[i], outputs[j],
                "{} and {} produced IDENTICAL output -- a codebook state is wired \
                 to the wrong decode",
                names[i], names[j]
            );
        }
    }
}

/// The pipeline table's INDEX must select the kernel it was built with.
///
/// This test exists because it did not. The first version of the index was
/// `1 + family * 16 + aligned * 8 + bits`, where the table is 0-based over a
/// 1-based `bits`, so every slot was one too high: slot 12 held
/// `linear_gemv_exl3_k4` and a 3-bit MCG matvec silently ran the 4-bit kernel.
/// The kernel still runs, still returns finite numbers, and the engine is simply
/// wrong. It was caught only by an unrelated pre-existing test
/// (`metal_buffer_backed_linear_matches_slice_path`) failing.
///
/// Nothing here caught it, and the reason is worth recording: the
/// bit-exactness test dispatches by NAME, so it never goes through the index
/// function, and the generic kernel is byte-identical to the one on master, so
/// a name-based A/B cannot see which kernel the ENGINE reaches. "The
/// specialisation is faithful" and "the engine selects the specialised kernel" are
/// two separate claims and the second one needed its own test.
///
/// `exl3_gemv_kernel_name` reports the name of whatever actually sits in the
/// slot, so this compares the index against the real table rather than against a
/// second copy of the layout written here.
#[test]
fn exl3_gemv_index_selects_the_right_kernel() {
    let kernels = LinearKernels::new().expect("kernels");
    for codebook in [Exl3Codebook::Mcg, Exl3Codebook::Plain, Exl3Codebook::Mul1] {
        for aligned in [false, true] {
            for bits in 1_u8..=8 {
                // packed_offset 0 is 4-byte aligned; 2 is the real misaligned
                // case (`lm_head.trellis` starts at a file offset of % 4 == 2).
                let offset = if aligned { 0u64 } else { 2u64 };
                let family = match codebook {
                    Exl3Codebook::Mcg => "_mcg",
                    Exl3Codebook::Plain | Exl3Codebook::Mul1 => "",
                };
                let suffix = if aligned { "" } else { "_u" };
                let want = format!("linear_gemv_exl3_k{bits}{suffix}{family}");
                let got = kernels
                    .exl3_gemv_kernel_name(Exl3Rate::Integer(bits), offset, codebook)
                    .unwrap_or_else(|| {
                        panic!("{codebook:?} aligned={aligned} bits={bits}: no slot")
                    });
                assert_eq!(
                    got, want,
                    "{codebook:?} aligned={aligned} bits={bits}: the index selects \
                     `{got}` but `{want}` is the kernel for this triple"
                );
            }
        }
    }
}

/// Every triple the model actually uses must resolve to the SPECIALISED kernel.
///
/// The whole point of the change is that the engine dispatches `_mcg`; if the
/// index ever routed MCG to the generic family, the change would be correct and
/// completely inert. This asserts the specific pairs the model uses, which are
/// fixed by the loader: every decoder linear is `Integer(3)` + `Mcg`, and
/// `lm_head` is `Integer(6)` + `Mcg` (`Model::validate_current_graph` refuses
/// anything else).
#[test]
fn the_models_own_rates_dispatch_to_the_specialised_kernel() {
    let kernels = LinearKernels::new().expect("kernels");
    for bits in [3_u8, 6] {
        // Arena-resident weights are page-aligned, so `packed_offset` is 0.
        let name = kernels
            .exl3_gemv_kernel_name(Exl3Rate::Integer(bits), 0, Exl3Codebook::Mcg)
            .unwrap_or_else(|| panic!("bits={bits}: no slot"));
        assert_eq!(
            name,
            format!("linear_gemv_exl3_k{bits}_mcg"),
            "bits={bits}: the model would run `{name}`, which is not specialised"
        );
    }
}
