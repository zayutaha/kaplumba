//! The 4 bpw colmajor measurement kernel must be a real kernel, and it must be
//! unreachable from the engine.
//!
//! `linear_gemv_ishizuki_k4_colmajor_contig4_bench`
//! (crates/kernels/src/linear_repacked.metal) exists only to answer one question:
//! is the 3 bpw colmajor decode's ~188 GB/s the EXL3 rate's lane-load
//! amplification (2.667) or a kernel defect? It is `..._k3_colmajor_contig4`
//! with `KA` 3 -> 4, and nothing selects it.
//!
//! Three claims, and the failure mode each guards:
//!
//! 1. **It computes the right thing.** Measured bandwidth is meaningless if the
//!    kernel reads the wrong bytes, and "reads the wrong bytes" produces a
//!    perfectly plausible GB/s. So it is compared against the *CPU* reference
//!    `decode_tile` -- not against another GPU kernel. `live_kernel_bit_exact.rs`
//!    compares GPU kernels to each other, which cannot catch an error both
//!    kernels share; here the reference is `kaplumba_engine::exl3`, an
//!    independent implementation of the same layout.
//!
//! 2. **The production dispatch cannot reach it.** Two mechanisms could do it,
//!    and they are independent:
//!      * `ishizuki_gemv_pipeline` looks up
//!        `linear_gemv_ishizuki_k{bits}_colmajor_contig4` by *format string*.
//!        A kernel named `..._k4_colmajor_contig4` would be selected for any
//!        4 bpw `perm_mode == 1` tensor. `_bench` is what makes the lookup miss.
//!      * the per-rate *slot* table claims the sorted-first kernel naming each
//!        rate, so a name that sorts before `linear_gemv_ishizuki_k4` would
//!        silently take over 4 bpw dispatch for every path.
//!    Neither is observable from a benchmark, so both are asserted here against
//!    the real `LinearKernels` registry.
//!
//! 3. **The amplification arithmetic is what the kernel comment claims.** 8
//!    codes per lane at `KA` bits each, loaded as two 32-bit words, gives
//!    `8/KA` lane-loads per useful byte -- 2.667 at 3 bpw and 2.000 at 4 bpw.
//!    That is the premise of the whole measurement, and it is checked against
//!    `unpack_integer`'s own window arithmetic rather than restated.

use half::f16;
use kaplumba_engine::exl3::{decode_tile, hadamard_128, pack_integer_states};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Buffer, Device, MTLResourceOptions};

/// 32-bit words of packed trellis per 16x16 tile at an integer rate.
///
/// `RING / 32` where `RING = 256 * KA`: 24 at 3 bpw (= the 48 u16 the
/// production `reorder_tensor_to_colmajor` hardcodes), 32 at 4 bpw.
fn words_per_tile(bits: u8) -> usize {
    8 * bits as usize
}

/// Transpose the tile grid, for any integer rate.
///
/// `exl3::reorder_tensor_to_colmajor` hardcodes 48 u16 per tile and so only
/// works at 3 bpw. The transposition itself is rate-agnostic: tile `(tk, tn)` at
/// row-major offset `(tk * tiles_n + tn) * W` moves to colmajor offset
/// `(tn * tiles_k + tk) * W`, and a tile's `W` words are copied verbatim.
/// `the_colmajor_repack_is_a_pure_tile_permutation` checks that against the
/// production function at 3 bpw, so this is not a second opinion.
fn reorder_to_colmajor(packed: &[u16], in_features: usize, out_features: usize, bits: u8) -> Vec<u16> {
    let tiles_k = in_features / 16;
    let tiles_n = out_features / 16;
    let w = words_per_tile(bits) * 2;
    assert_eq!(
        packed.len(),
        tiles_k * tiles_n * w,
        "packed length must be tiles x {} u16",
        w
    );
    let mut out = vec![0u16; packed.len()];
    for tk in 0..tiles_k {
        for tn in 0..tiles_n {
            let src = (tk * tiles_n + tn) * w;
            let dst = (tn * tiles_k + tk) * w;
            out[dst..dst + w].copy_from_slice(&packed[src..src + w]);
        }
    }
    out
}

/// Row-major packed tiles, deterministic in `(in_f, out_f, bits, seed)`.
///
/// Splitmix64 over `(tile, code)` rather than a running LCG, and the difference
/// is not cosmetic. A running LCG's low 3 bits have period 16 (the multiplier
/// is 5 mod 8, so the low bits alternate with period 2 and close at 16), and a
/// tile is 256 draws: every tile then receives the *same* 256-state sequence and
/// the packed tensor is a byte-for-byte repetition of one tile. Bandwidth is
/// unaffected -- the same bytes are read at the same addresses -- but a
/// colmajor-vs-row-major repack becomes the identity, so the check that the two
/// arms hold different layouts silently tests nothing. That is exactly what
/// happened the first time this generator was written.
fn build_rowmajor(in_f: usize, out_f: usize, bits: u8, seed: u32) -> Vec<u16> {
    let tiles = (in_f / 16) * (out_f / 16);
    let mask = (1u32 << bits) - 1;
    let mut packed = Vec::with_capacity(tiles * 16 * bits as usize);
    for tile in 0..tiles {
        let mut states = [0u16; 256];
        for (code, slot) in states.iter_mut().enumerate() {
            let mut z = u64::from(seed) ^ (tile as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (code as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            *slot = (z as u32 & mask) as u16;
        }
        packed.extend(pack_integer_states(&states, bits).expect("pack").0);
    }
    packed
}

/// The CPU reference matvec, mirroring the whole engine pipeline exactly.
///
/// `encode_ishizuki_gemv_named` dispatches three kernels, and both transforms
/// are part of the output -- skipping either produces a plausible non-zero
/// answer:
///
/// ```text
/// linear_input_transform : x <- half( H128( float(x) * float(input_signs) ) )
/// linear_gemv_*          : z[n] = sum_k W[k][n] * x[k]                    (f32)
/// linear_finalize_simd   : y <- half( H128(z) * float(output_signs) )
/// ```
///
/// Both boundaries round to fp16 (`half(...)` in the Metal above), so the
/// reference rounds too: keeping f32 all the way through would differ from the
/// kernel by more than reassociation and the test would be measuring the wrong
/// thing.
///
/// The weights come from `exl3::decode_tile`, a *different* implementation of
/// the ring-window arithmetic from the Metal one (`unpack_integer` in Rust vs
/// inline extraction in `linear_repacked.metal`), so agreement is evidence
/// rather than tautology.
fn cpu_reference(
    rowmajor: &[u16],
    in_f: usize,
    out_f: usize,
    bits: u8,
    input: &[f16],
    input_signs: &[f16],
    output_signs: &[f16],
) -> Vec<f16> {
    use kaplumba_engine::exl3::{BitRate, Codebook};
    assert_eq!(input.len(), in_f);
    assert_eq!(output_signs.len(), out_f);

    // Input side.
    let mut x: Vec<f32> = input
        .iter()
        .zip(input_signs)
        .map(|(v, s)| v.to_f32() * s.to_f32())
        .collect();
    for block in x.chunks_mut(128) {
        let mut slice: Vec<f32> = block.to_vec();
        hadamard_128(&mut slice).expect("hadamard");
        block.copy_from_slice(&slice);
    }
    let x: Vec<f32> = x.into_iter().map(f16::from_f32).map(f16::to_f32).collect();

    // The matvec itself, tile at a time. `decode_tile` returns the tile already
    // permuted onto (row, col).
    let tiles_k = in_f / 16;
    let tiles_n = out_f / 16;
    let w = words_per_tile(bits) * 2;
    let mut z = vec![0f32; out_f];
    for tn in 0..tiles_n {
        for tk in 0..tiles_k {
            let tile = &rowmajor[(tk * tiles_n + tn) * w..(tk * tiles_n + tn + 1) * w];
            let weights = decode_tile(tile, BitRate::Integer(bits), Codebook::Mcg)
                .expect("decode tile");
            for row in 0..16 {
                let xk = x[tk * 16 + row];
                for col in 0..16 {
                    z[tn * 16 + col] += xk * weights[row * 16 + col];
                }
            }
        }
    }

    // Output side.
    for block in z.chunks_mut(128) {
        let mut slice: Vec<f32> = block.to_vec();
        hadamard_128(&mut slice).expect("hadamard");
        block.copy_from_slice(&slice);
    }
    z.iter()
        .zip(output_signs)
        .map(|(v, s)| f16::from_f32(v * s.to_f32()))
        .collect()
}

fn upload(device: &Device, v: &[u16]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn upload_f16(device: &Device, v: &[f16]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

struct Sig {
    input: Buffer,
    input_signs: Buffer,
    output_signs: Buffer,
    transformed: Buffer,
    raw: Buffer,
    output: Buffer,
}

fn run(
    kernels: &LinearKernels,
    queue: &metal::CommandQueue,
    kernel: &str,
    packed: &Buffer,
    in_f: usize,
    out_f: usize,
    bits: u8,
    sig: &Sig,
) -> Vec<u16> {
    let mut encoder = MetalEncoder::new(queue).expect("encoder");
    kernels
        .encode_gemv_named(
            &mut encoder,
            kernel,
            &LinearEncodeRequest {
                packed,
                packed_offset: 0,
                perm_mode: 1,
                input: &sig.input,
                input_signs: &sig.input_signs,
                output_signs: &sig.output_signs,
                transformed: &sig.transformed,
                raw: &sig.raw,
                output: &sig.output,
                in_features: in_f,
                out_features: out_f,
                rate: Exl3Rate::Integer(bits),
                codebook: Exl3Codebook::Mcg,
                pretransformed: false,
            },
        )
        .unwrap_or_else(|e| panic!("{kernel} {in_f}x{out_f}@{bits}: {e}"));
    encoder.commit_and_wait().expect("wait");
    let p = sig.output.as_ref().contents() as *const u16;
    (0..out_f).map(|i| unsafe { p.add(i).read() }).collect()
}

// ---------------------------------------------------------------------------

#[test]
fn the_colmajor_repack_is_a_pure_tile_permutation() {
    // The harness's rate-generic repack must agree with the production 3 bpw one
    // exactly. If it does not, the 4 bpw arm is measuring a different layout from
    // the one the engine runs at 3 bpw, and the A/B is meaningless.
    let in_f = 512;
    let out_f = 1024;
    let row = build_rowmajor(in_f, out_f, 3, 99);
    let mine = reorder_to_colmajor(&row, in_f, out_f, 3);
    let theirs =
        kaplumba_engine::exl3::reorder_tensor_to_colmajor(&row, in_f, out_f).expect("colmajor");
    assert_eq!(mine, theirs, "the rate-generic repack diverges at 3 bpw");

    // And it is a permutation, not a copy: the two orders must actually differ.
    assert_ne!(mine, row, "the repack did not transpose anything");

    // Word count per tile: 48 u16 at 3 bpw (the production hardcode), 64 at 4.
    assert_eq!(words_per_tile(3) * 2, 48);
    assert_eq!(words_per_tile(4) * 2, 64);
}

#[test]
fn the_two_word_window_is_safe_exactly_at_rates_one_through_four() {
    // Premise of the whole measurement, checked rather than restated.
    //
    // The two-word form `A0 = ((w0 << 32) | w1) << sh` fills A0 bits `[sh, 64)`
    // from the pair and leaves `[0, sh)` zero. Code `i` reads A0 bits
    // `[64 - r, 80 - r)` with `r = i*KA + 16`, so the lowest bit any code touches
    // is `48 - 7*KA`, and the pair suffices exactly when
    //
    //     (a) 7*KA + 16 <= 64                      AND
    //     (b) max over lanes of sh <= 48 - 7*KA
    //
    // (b) is the condition that is easy to miss: (a) alone gives "KA <= 6", which
    // is WRONG, and the 6 bpw kernel built on it decoded 1023 of 1024 outputs as
    // garbage. This test is what that mistake should have been caught by instead.
    //
    // It is measured, not argued: for every rate, recompute all 256 codes from
    // real packed data twice -- once with the two-word form and once with the
    // three-word form -- and require equality only where equality is derivable.
    // The unsafe rates must actually FAIL, or this test could be passing because
    // the comparison is vacuous.
    //
    // `scripts/gen_exl3_gemv.py`'s `TWO_WORD_SAFE_RATES` is the same derivation in
    // Python and evaluates to (1, 2, 3, 4); `decode_matvec_golden.rs` pins that
    // tuple from the other side.
    const SAFE: &[u8] = &[1, 2, 3, 4];

    for bits in 1u8..=8 {
        let ka = bits as u32;
        let ring = 256 * ka;
        let word_count = (ring / 32) as usize;
        // (a): no code falls through to B0.
        let a_ok = 7 * ka + 16 <= 64;
        // (b): no code reads below its own sh.
        let max_sh = (0..32u32)
            .map(|lane| ((lane * 8 + 1) * ka + ring - 16) % 32)
            .max()
            .expect("32 lanes");
        let derivable = a_ok && max_sh <= 48 - 7 * ka;
        assert_eq!(
            derivable,
            SAFE.contains(&bits),
            "KA {ka}: derivation says the two-word window is {state} \
             (7*KA+16 = {} <= 64 is {a_ok}, max sh = {max_sh}, 48 - 7*KA = {}) \
             but SAFE says {expected}",
            7 * ka + 16,
            48i32 - 7 * ka as i32,
            state = if derivable { "safe" } else { "unsafe" },
            expected = SAFE.contains(&bits),
        );
        // At KA 7 and 8 some code has `r > 64`, so the two-word form is not even
        // shiftable -- `A0 >> (64 - r)` would underflow the shift amount. There
        // is nothing to compare, which is exactly what "unsafe" means here.
        if !a_ok {
            continue;
        }

        for seed in [1u32, 7, 4242] {
            let tile = build_rowmajor(16, 16, bits, seed);
            let words: Vec<u32> = tile
                .chunks_exact(2)
                .map(|p| u32::from(p[0]) | (u32::from(p[1]) << 16))
                .collect();
            assert_eq!(words.len(), word_count, "KA {ka}: word count");
            let mut mismatches = 0usize;
            for lane in 0..32u32 {
                let lo = (lane * 8 + 1) * ka + ring - 16;
                let i0 = (lo / 32) as usize;
                let shift = lo % 32;
                let j0 = if i0 >= word_count { i0 - word_count } else { i0 };
                let j1 = (j0 + 1) % word_count;
                let j2 = (j1 + 1) % word_count;
                let w0 = u64::from(words[j0]);
                let w1 = u64::from(words[j1]);
                let w2 = u64::from(words[j2]);
                // Three-word form: `unpack_integer`'s, and the 3/5/6/7 bpw
                // kernels' reference semantics.
                let a3 = ((w0 << 32) | w1) << shift | if shift != 0 { w2 >> (32 - shift) } else { 0 };
                let c3 = ((w1 << 32) | w2) << shift;
                // Two-word form: what `..._contig4_bench` at 3 and 4 bpw computes.
                let a2 = ((w0 << 32) | w1) << shift;
                for i in 0..8u32 {
                    let r = i * ka + 16;
                    let three = if r <= 64 {
                        (a3 >> (64 - r)) & 0xffff
                    } else {
                        (c3 >> (96 - r)) & 0xffff
                    };
                    let two = (a2 >> (64 - r)) & 0xffff;
                    if three != two {
                        mismatches += 1;
                    }
                }
            }
            if SAFE.contains(&bits) {
                assert_eq!(
                    mismatches, 0,
                    "KA {ka}, seed {seed}: the two-word window disagrees with the \
                     three-word one in {mismatches} of 256 codes, so the kernel \
                     built on it would decode garbage"
                );
            } else {
                assert!(
                    mismatches > 0,
                    "KA {ka}, seed {seed}: the two-word window was derived unsafe \
                     but the comparison found no disagreement in 256 codes, so this \
                     test cannot distinguish safe from unsafe and its SAFE set is \
                     untested"
                );
            }
        }
    }
}

/// How far a kernel's fp16 output is from the CPU reference, in ULPs.
///
/// A *relative* measure, because the reference's own f32 summation order
/// (tile at a time, row then column) is not the kernel's (four lanes of partial
/// sums, a 4-wide shuffle butterfly, then eight simdgroups in threadgroup
/// memory). Float addition is not associative, so exact f32 -- and sometimes
/// exact fp16 -- agreement is not available against a different summation order,
/// and pretending otherwise would mean loosening the bound until it could not
/// fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Deviation {
    differing: usize,
    total: usize,
    max_ulp: i64,
}

fn deviation(expected: &[f16], got: &[u16]) -> Deviation {
    assert_eq!(expected.len(), got.len());
    let mut differing = 0;
    let mut max_ulp = 0i64;
    for (a, b) in expected.iter().zip(got) {
        let b = f16::from_bits(*b);
        if a.to_bits() != b.to_bits() {
            differing += 1;
            // Ordered-int ULP distance, which is monotone in |a - b| including
            // across zero, where a plain subtraction is not.
            let key = |v: f16| -> i64 {
                let bits = v.to_bits() as i64;
                if bits & 0x8000 != 0 {
                    -(bits & 0x7fff)
                } else {
                    bits
                }
            };
            max_ulp = max_ulp.max((key(*a) - key(b)).abs());
        }
    }
    Deviation {
        differing,
        total: expected.len(),
        max_ulp,
    }
}

#[test]
fn the_four_and_six_bpw_colmajor_kernels_match_the_cpu_reference() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // A Metal compile failure must NOT be a skip. `LinearKernels::new` returning
    // Err is how a broken kernel reaches a measurement: the harness prints "no
    // device, skipping", the test goes green, and the bandwidth number comes out
    // of a library without the kernel in it.
    let kernels = LinearKernels::new().expect("LinearKernels::new failed; the Metal library did not build");
    let queue = device.new_command_queue().to_owned();

    // Shapes chosen for loop coverage, not for realism:
    //   128x1024  -- 8 k tiles, so `k_unroll_end = (8/32)*32 = 0` and the
    //                4x-unrolled loop runs ZERO times; only the tail loop runs.
    //   1024x1024 -- 64 k tiles, unrolled loop runs twice, tail loop empty.
    //   5376x5376 -- a real shape, 336 k tiles, so `k_unroll_end = 320` leaves 16
    //                tail iterations.
    // Three rates per shape, and the 3 bpw arm is the calibration: it is the
    // engine's live decode kernel, so its deviation from this reference measures
    // the *reference's* reassociation noise. The new rates are then required to
    // be no worse than it. Without that, a loose bound on a wrong reference would
    // pass, and this file has seen that failure mode: two GPU kernels compared to
    // each other agree while both are wrong.
    for (in_f, out_f) in [(128usize, 1024usize), (1024, 1024), (5376, 5376)] {
        let mut baseline: Option<Deviation> = None;
        for (bits, kernel) in [
            (3u8, "linear_gemv_ishizuki_k3_colmajor_contig4"),
            (4, "linear_gemv_ishizuki_k4_colmajor_contig4_bench"),
            (6, "linear_gemv_ishizuki_k6_colmajor_contig4_bench"),
        ] {
            let row = build_rowmajor(in_f, out_f, bits, 13 + bits as u32);
            let col = reorder_to_colmajor(&row, in_f, out_f, bits);
            let packed = upload(&device, &col);

            let input: Vec<f16> = (0..in_f)
                .map(|i| f16::from_f32((i as f32 * 0.017).sin()))
                .collect();
            let input_signs: Vec<f16> = (0..in_f)
                .map(|i| f16::from_f32(if i % 3 == 0 { 0.0021 } else { -0.0017 }))
                .collect();
            let output_signs: Vec<f16> = (0..out_f)
                .map(|i| f16::from_f32(if i % 5 == 0 { -0.0031 } else { 0.0023 }))
                .collect();
            let expected = cpu_reference(
                &row,
                in_f,
                out_f,
                bits,
                &input,
                &input_signs,
                &output_signs,
            );

            let sig = Sig {
                input: upload_f16(&device, &input),
                input_signs: upload_f16(&device, &input_signs),
                output_signs: upload_f16(&device, &output_signs),
                transformed: device.new_buffer(
                    (in_f * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
                raw: device.new_buffer((out_f * 4) as u64, MTLResourceOptions::StorageModeShared),
                output: device.new_buffer(
                    (out_f * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            };
            let got = run(&kernels, &queue, kernel, &packed, in_f, out_f, bits, &sig);

            // A kernel that reads the wrong bytes still produces a plausible GB/s.
            // This is the guard against measuring a wrong kernel's bandwidth, and
            // it is checked on both the reference and the kernel so a reference
            // that degenerated to zeros cannot pass either.
            assert!(
                expected.iter().any(|v| v.to_bits() != 0),
                "{in_f}x{out_f}@{bits}: the CPU reference produced all zeros"
            );
            assert!(
                got.iter().any(|&v| v != 0),
                "{in_f}x{out_f}@{bits} {kernel}: the kernel produced all zeros"
            );

            let dev = deviation(&expected, &got);
            println!(
                "{in_f}x{out_f}@{bits} {kernel}: {}/{} fp16 outputs differ, \
                 max {max_ulp} ULP",
                dev.differing, dev.total, max_ulp = dev.max_ulp
            );
            match (bits, baseline) {
                // The 3 bpw arm sets the bar. It is the production kernel, so
                // anything it cannot reach is the reference's own noise.
                (3, _) => baseline = Some(dev),
                (_, Some(base)) => {
                    assert!(
                        dev.max_ulp <= base.max_ulp,
                        "{in_f}x{out_f}@{bits} {kernel}: max {dev} ULP from the CPU \
                         reference, but the 3 bpw production kernel is within {base} ULP \
                         ({diff}/{total} outputs differ). A new rate must not be \
                         less faithful than the kernel it is transcribed from.",
                        dev = dev.max_ulp,
                        base = base.max_ulp,
                        diff = dev.differing,
                        total = dev.total,
                    );
                    assert!(
                        dev.differing * 200 <= dev.total,
                        "{in_f}x{out_f}@{bits} {kernel}: {diff}/{total} fp16 outputs \
                         differ from the CPU reference (>0.5%), which is reassociation \
                         noise in the reference, not a kernel fault. 3 bpw differs in \
                         {base_diff}/{base_total}.",
                        diff = dev.differing,
                        total = dev.total,
                        base_diff = base.differing,
                        base_total = base.total,
                    );
                }
                (_, None) => unreachable!("the 3 bpw arm is always first"),
            }
        }
    }
}

#[test]
fn the_engine_cannot_dispatch_the_four_bpw_measurement_kernel() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let kernels = LinearKernels::new().expect("LinearKernels::new failed; the Metal library did not build");
    let names = kernels.ishizuki_gemv_kernel_names();
    let mine = "linear_gemv_ishizuki_k4_colmajor_contig4_bench";
    assert!(
        names.iter().any(|n| n == mine),
        "the 4 bpw colmajor kernel is not in the library ({mine}); \
         the whole measurement would be measuring something else"
    );

    // (a) No production key can match it. `ishizuki_gemv_pipeline` builds all
    //     four of its keys from a bit rate, so this is exhaustive, not a sample.
    for bits in 1u8..=8 {
        for key in [
            format!("linear_gemv_ishizuki_k{bits}_colmajor_contig4"),
            format!("linear_gemv_ishizuki_k{bits}_colmajor"),
            format!("linear_gemv_ishizuki_k{bits}_w2"),
            format!("linear_gemv_ishizuki_k{bits}"),
        ] {
            assert_ne!(
                key, mine,
                "the 4 bpw measurement kernel is reachable as the production key \
                 {key}; rename it, do not just change this test"
            );
        }
    }

    // (b) It cannot take the 4 bpw dispatch slot. The slot goes to the
    //     sorted-first kernel naming that rate (`ishizuki_kernel_names` sorts),
    //     so what matters is the ordering, not the presence.
    let slot_owner = names
        .iter()
        .find(|n| LinearKernels::ishizuki_rate_of_name(n) == Some(4))
        .expect("some kernel names rate 4");
    assert_eq!(
        slot_owner.as_str(),
        "linear_gemv_ishizuki_k4",
        "the 4 bpw slot is claimed by {slot_owner}, not by the unsuffixed kernel; \
         production 4 bpw dispatch has changed"
    );

    // (c) The 3 bpw arm of the A/B is the kernel the engine dispatches, which is
    //     the whole point of "the same colmajor path". Assert the name rather
    //     than trusting the harness's constant.
    assert!(
        kernels
            .ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_colmajor_contig4")
            .is_some(),
        "the 3 bpw production colmajor kernel is missing; the 3 bpw baseline is \
         not the live path and the comparison would be against nothing"
    );
    let _ = device;
}
