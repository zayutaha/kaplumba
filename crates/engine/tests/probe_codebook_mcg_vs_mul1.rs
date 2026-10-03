//! PROBE (read-only investigation): MCG vs Mul1 EXL3 codebooks.
//!
//! Question under investigation: the checkpoint is `codebook: "mcg"`, integer
//! rate 3. `Exl3Codebook::Mul1` exists, and `linear.metal:1586-1591` claims
//! branching on the codebook eight times per lane "cost 61.7% of the kernel".
//! So: is MCG's arithmetic more expensive than Mul1's, would Mul1 be faster at
//! an integer rate, and what does switching cost in quality?
//!
//! **No GPU, no model, no `gpuq`.** Nothing here calls
//! `Device::system_default()` or constructs a `LinearKernels`; the only GPU
//! *surface* touched is the Metal source text, read at compile time with
//! `include_str!` and parsed for structure. `LinearKernels::exl3_gemv_index` is
//! an associated pure function, so calling it does not create a device.
//!
//! Findings this file pins (each is an assertion, not a comment):
//!
//!  1. `decode_state_mcg` is 4 ops/code (3 integer + 1 fp16 add);
//!     `decode_state_mul1` is 18 (12 integer + 2 fp32 + 4 cvt). MCG is the
//!     CHEAP one, by ~4.5x in op count.
//!  2. Both transcri�ptions are bit-identical to the engine's CPU reference
//!     `decode_state` for all 256 states.
//!  3. **No Mul1 kernel exists.** `exl3_gemv_index` resolves the `Mul1` family
//!     to the UNSUFFIXED `linear_gemv_exl3_k{bits}` kernel, i.e. the
//!     branchy runtime-codebook one that `linear.metal:322-326` measured at
//!     62.4 GB/s against 163.0 GB/s for the MCG-specialised family.
//!  4. `MCG_MULTIPLIER` is a compile-time literal in all nine Metal sites and
//!     has no field in `LinearParams`; the per-tensor `.mcg` safetensors entry
//!     is validated for existence/shape and never read into a buffer.
//!  5. MCG and Mul1 decode the same trellis state to *completely different*
//!     scalars, so `"codebook"` is not a decode-time flag: it is a property of
//!     the quantiser run that produced the bytes.

use half::f16;
use kaplumba_engine::exl3::{Codebook, MCG_MULTIPLIER, MUL1_MULTIPLIER, decode_state};
use kaplumba_kernels::exl3::Exl3Codebook;
use kaplumba_kernels::linear::LinearKernels;

/// The two Metal decode functions, and the whole dispatch struct, as shipped.
const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const LINEAR_REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");

// ---------------------------------------------------------------------------
// 1. Structural extraction of the decode functions from the shipped Metal.
// ---------------------------------------------------------------------------

/// Return the body of `static ... name(uint` ... up to its closing brace.
fn metal_body<'a>(source: &'a str, name: &str) -> &'a str {
    let anchor = source
        .find(name)
        .unwrap_or_else(|| panic!("{name} not found in Metal source"));
    let open = source[anchor..]
        .find('{')
        .map(|i| anchor + i)
        .unwrap_or_else(|| panic!("no opening brace after {name}"));
    let mut depth = 0_usize;
    for (offset, ch) in source[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open..open + offset + 1];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces after {name}");
}

/// Normalise whitespace so a canonical transcription can be compared literally.
fn normalise(body: &str) -> String {
    let stripped: String = body
        .lines()
        .map(|line| {
            line.split("//")
                .next()
                .unwrap_or("")
                .trim()
                .to_string()
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Per-code operation census of one `decode_state_*` body.
///
/// Counted as the ISA would issue them, from the transcribed body below. The
/// transcription is pinned against `LINEAR_METAL` by
/// `transcription_matches_the_shipped_metal_source`, so these numbers describe
/// the kernel that actually runs rather than a description of it.
#[derive(Debug, Default, PartialEq, Eq)]
struct OpCensus {
    integer: u32,
    fp32: u32,
    fp16: u32,
    convert: u32,
}

impl OpCensus {
    fn total(&self) -> u32 {
        self.integer + self.fp32 + self.fp16 + self.convert
    }
}

/// `linear.metal:121-126`, transcribed, with an op tally.
///
///   value = value * 0xCBAC1FEDu;                     IMUL
///   value = 0x3b603b60u ^ (value & 0x8fff8fffu);     AND + XOR
///   half2 h = as_type<half2>(value);                 bitcast, free
///   return float(h.x + h.y);                         one fp16 add
fn mcg_census_and_value(state: u32) -> (OpCensus, f32) {
    let mut c = OpCensus::default();
    let mut value = state.wrapping_mul(MCG_MULTIPLIER); // IMUL
    c.integer += 1;
    value = 0x3b60_3b60_u32 ^ (value & 0x8fff_8fff); // AND + XOR
    c.integer += 2;
    let h = as_type_half2(value);
    let out = (h.0 + h.1).to_f32(); // fp16 add, then the free f16->f32 widen
    c.fp16 += 1;
    (c, out)
}

/// `linear.metal:128-135`, transcribed, with an op tally.
///
///   IMUL, then 4 AND + 3 SHIFT + 4 ADD for the byte sum, then in f32
///   (because the source writes `float(byte_value) * float(inverse) +
///   float(bias)` and then re-rounds to half) 1 MUL + 1 ADD and 4 converts.
fn mul1_census_and_value(state: u32) -> (OpCensus, f32) {
    let mut c = OpCensus::default();
    let value = state.wrapping_mul(MUL1_MULTIPLIER); // IMUL
    c.integer += 1;
    // Four byte extractions: 4 AND, 3 SHIFT (the low byte needs none).
    let sum = 0x6400_u32
        + (value & 0xff)
        + ((value >> 8) & 0xff)
        + ((value >> 16) & 0xff)
        + ((value >> 24) & 0xff);
    c.integer += 4 + 3 + 4;
    let byte_value = f16::from_bits(sum as u16);
    let inverse = f16::from_bits(0x1eee);
    let bias = f16::from_bits(0xc931);
    // float(...) on all three, multiply, add, then half(...) and float(...).
    let out = f16::from_f32(byte_value.to_f32() * inverse.to_f32() + bias.to_f32()).to_f32();
    c.fp32 += 2;
    c.convert += 4;
    (c, out)
}

/// Minimal `as_type<half2>` so the MCG transcription does not need a SIMD type.
fn as_type_half2(bits: u32) -> (f16, f16) {
    (f16::from_bits(bits as u16), f16::from_bits((bits >> 16) as u16))
}

#[test]
fn mcg_is_four_ops_per_code_and_mul1_is_eighteen() {
    let (mcg, mcg_value) = mcg_census_and_value(0);
    let (mul1, mul1_value) = mul1_census_and_value(0);
    assert_eq!(
        mcg,
        OpCensus {
            integer: 3,
            fp32: 0,
            fp16: 1,
            convert: 0
        },
        "MCG should be IMUL + AND + XOR + one fp16 add"
    );
    assert_eq!(
        mul1,
        OpCensus {
            integer: 12,
            fp32: 2,
            fp16: 0,
            convert: 4
        },
        "Mul1 should be IMUL + (4 AND + 3 SHIFT + 4 ADD) + fp32 MUL/ADD + 4 converts"
    );
    assert_eq!(mcg.total(), 4);
    assert_eq!(mul1.total(), 18);
    assert!(
        mcg.total() * 4 < mul1.total(),
        "MCG is the cheaper codebook, not the dearer one"
    );
    // Both are finite and in a sane weight range for a 3 bpw tensor.
    assert!(mcg_value.is_finite() && mul1_value.is_finite());
}

#[test]
fn transcription_matches_the_shipped_metal_source() {
    // If someone edits the kernel the census above becomes a description of
    // nothing, so pin both bodies literally (comments and whitespace removed).
    let mcg = normalise(metal_body(LINEAR_METAL, "decode_state_mcg"));
    assert!(
        mcg.contains("value = value * 0xCBAC1FEDu;")
            && mcg.contains("value = 0x3b603b60u ^ (value & 0x8fff8fffu);")
            && mcg.contains("half2 h = as_type<half2>(value);")
            && mcg.contains("return float(h.x + h.y);"),
        "decode_state_mcg no longer matches the transcription; \
         `mcg_is_four_ops_per_code_and_mul1_is_eighteen` is now stale. Actual: {mcg}"
    );
    let mul1 = normalise(metal_body(LINEAR_METAL, "decode_state_mul1"));
    for fragment in [
        "value = value * 0x83DCD12Du;",
        "uint sum = 0x6400u + (value & 0xffu) + ((value >> 8) & 0xffu) + ((value >> 16) & 0xffu) + ((value >> 24) & 0xffu);",
        "half byte_value = as_type<half>((ushort)sum);",
        "return float(half(float(byte_value) * float(inverse) + float(bias)));",
    ] {
        assert!(
            mul1.contains(fragment),
            "decode_state_mul1 fragment {fragment} vanished. Actual: {mul1}"
        );
    }
    // And the shared hash step is genuinely three instructions, not a comment.
    assert_eq!(mcg.matches('*').count(), 1, "one multiply in MCG");
    assert!(mul1.matches('&').count() >= 4, "four masked byte extractions in Mul1");
}

#[test]
fn both_transcriptions_are_bit_identical_to_the_cpu_reference() {
    for state in 0_u32..=0xffff {
        let (mcg_census, mcg) = mcg_census_and_value(state);
        let (mul1_census, mul1) = mul1_census_and_value(state);
        assert_eq!(
            mcg.to_bits(),
            decode_state(state as u16, Codebook::Mcg).to_bits(),
            "MCG transcription diverges from the engine reference at state {state}"
        );
        assert_eq!(
            mul1.to_bits(),
            decode_state(state as u16, Codebook::Mul1).to_bits(),
            "Mul1 transcription diverges from the engine reference at state {state}"
        );
        // A fixed op count for every state: no data-dependent path.
        assert_eq!(mcg_census, mcg_census_and_value(0).0);
        assert_eq!(mul1_census, mul1_census_and_value(0).0);
    }
}

// ---------------------------------------------------------------------------
// 2. Is Mul1 at an integer rate actually available? No.
// ---------------------------------------------------------------------------

#[test]
fn no_mul1_decode_kernel_exists_in_either_metal_source() {
    for name in [
        "linear_gemv_exl3_k3_mul1",
        "linear_gemv_exl3_k3_u_mul1",
        "linear_gemv_exl3_k1_mul1",
        "linear_gemv_exl3_k8_mul1",
    ] {
        assert!(
            !LINEAR_METAL.contains(name),
            "{name} unexpectedly exists; finding 3 is stale"
        );
    }
    // `decode_state_mul1` must be reachable ONLY from the generic dispatcher,
    // never from a specialised eight-times-unrolled loop.
    let call_sites = LINEAR_METAL
        .match_indices("decode_state_mul1")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(
        call_sites.len(),
        2,
        "expected exactly two references to decode_state_mul1 in linear.metal \
         (its definition and the fallback in `decode_state`), found {}",
        call_sites.len()
    );
    // The first is the definition; the second must be the runtime fallback.
    let body = normalise(metal_body(LINEAR_METAL, "static float decode_state("));
    assert!(
        body.contains("decode_state_mul1"),
        "the second reference should be the runtime fallback"
    );
    assert!(
        !LINEAR_METAL[call_sites[0]..call_sites[0] + 40].contains("return"),
        "the first reference should be the definition, not a call"
    );
    // The repacked/prefill family is hard-wired to MCG too.
    assert!(
        !LINEAR_REPACKED_METAL.contains("mul1"),
        "linear_repacked.metal unexpectedly mentions mul1"
    );
    assert!(
        LINEAR_REPACKED_METAL.contains("decode_state_mcg"),
        "linear_repacked.metal should decode via MCG"
    );
}

#[test]
fn mul1_dispatch_resolves_to_the_branchy_unsuffixed_kernel() {
    // `linear.rs:1596-1600`: family 0 = mcg, 1 = plain, 2 = mul1. Only mcg gets
    // the `_mcg` suffix in the construction loop at `linear.rs:485-489`, so the
    // Mul1 slot is served by the runtime-codebook kernel.
    let mcg_slot = LinearKernels::exl3_gemv_index(
        kaplumba_kernels::exl3::Exl3Rate::Integer(3),
        0,
        Exl3Codebook::Mcg,
    )
    .expect("mcg 3bpw has a slot");
    let mul1_slot = LinearKernels::exl3_gemv_index(
        kaplumba_kernels::exl3::Exl3Rate::Integer(3),
        0,
        Exl3Codebook::Mul1,
    )
    .expect("mul1 3bpw has a slot");

    // Different slots, and the slot table is 1 + 3*2*8 long (`linear.rs:35`).
    // `packed_offset = 0` is 4-byte aligned, so the `aligned` term is 1.
    assert_ne!(mcg_slot, mul1_slot);
    assert_eq!(mcg_slot, 1 + 0 * 16 + 1 * 8 + 2); // mcg, aligned, bits=3
    assert_eq!(mul1_slot, 1 + 2 * 16 + 1 * 8 + 2); // mul1, aligned, bits=3
    assert!(mul1_slot < 1 + 3 * 2 * 8);

    // Reconstruct the kernel NAME the way `linear.rs:491-493` does, and show the
    // Mul1 name carries no `_mcg` -- i.e. it is the branchy kernel.
    let name_for = |codebook: Exl3Codebook| -> String {
        let family = if codebook == Exl3Codebook::Mcg { "_mcg" } else { "" };
        format!("linear_gemv_exl3_k3{family}")
    };
    assert_eq!(name_for(Exl3Codebook::Mcg), "linear_gemv_exl3_k3_mcg");
    assert_eq!(name_for(Exl3Codebook::Mul1), "linear_gemv_exl3_k3");
    assert!(
        LINEAR_METAL.contains("kernel void linear_gemv_exl3_k3_mcg(")
            && LINEAR_METAL.contains("kernel void linear_gemv_exl3_k3("),
        "both kernel names should exist"
    );
    // The unsuffixed one is the branchy form: it calls `exl3_decode8`, which
    // passes a runtime `codebook` into the unrolled loop.
    let generic = normalise(metal_body(LINEAR_METAL, "static inline void exl3_decode8("));
    assert!(
        generic.contains("decode_state(s & 0xffffu, codebook)"),
        "exl3_decode8 should still dispatch on a runtime codebook"
    );
    let specialised = normalise(metal_body(LINEAR_METAL, "static inline void exl3_decode8_mcg("));
    assert!(
        specialised.contains("decode_state_mcg(s & 0xffffu)"),
        "exl3_decode8_mcg should have no runtime codebook"
    );
    assert!(
        !specialised.contains("codebook"),
        "exl3_decode8_mcg must not mention the codebook at all"
    );
    // Every `constexpr int CB` in the library is 1 (mcg) -- 20 sites.
    let cb_values = LINEAR_METAL
        .match_indices("constexpr int CB")
        .map(|(index, _)| {
            LINEAR_METAL[index..index + 32]
                .chars()
                .find(|c| c.is_ascii_digit())
                .unwrap()
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(cb_values.len(), 20, "expected 20 folded-codebook kernels");
    assert!(
        cb_values.iter().all(|v| v == "1"),
        "every folded codebook constant must be MCG (1), found {cb_values:?}"
    );
}

// ---------------------------------------------------------------------------
// 3. Is the MCG multiplier a runtime load? No.
// ---------------------------------------------------------------------------

#[test]
fn mcg_multiplier_is_a_compile_time_literal_with_no_kernel_field() {
    // The constant is an immediate at every Metal site, never a name.
    let literal_sites = LINEAR_METAL.matches("0xCBAC1FEDu").count()
        + LINEAR_REPACKED_METAL.matches("0xCBAC1FEDu").count();
    assert!(
        literal_sites >= 3,
        "expected the multiplier as an immediate in the Metal sources, found {literal_sites}"
    );
    assert_eq!(MCG_MULTIPLIER, 0xCBAC1FED);
    assert_eq!(MUL1_MULTIPLIER, 0x83DCD12D);

    // `LinearParams` (`linear.metal:4-13`) is what the decode kernels receive. It
    // must carry a codebook id but no multiplier, so there is nothing to load
    // per tensor and nothing to load inside the K-loop.
    let params = normalise(&LINEAR_METAL[LINEAR_METAL
        .find("struct LinearParams")
        .expect("LinearParams")
        ..LINEAR_METAL.find("};").expect("end of LinearParams") + 2]);
    assert!(params.contains("uint rate;"));
    assert!(params.contains("uint codebook;"));
    for forbidden in ["multiplier", "mcg", "MCG", "constant uint *"] {
        assert!(
            !params.contains(forbidden),
            "LinearParams unexpectedly contains {forbidden}: {params}"
        );
    }

    // The Rust side matches: `MCG_MULTIPLIER` is a `const`, and the per-tensor
    // value lives in `quantization_config.json`, never in a Metal buffer.
    let engine_exl3 = include_str!("../src/exl3.rs");
    assert!(
        engine_exl3.contains("pub const MCG_MULTIPLIER: u32 = 0xCBAC1FED;"),
        "MCG_MULTIPLIER should stay a compile-time const"
    );
    // No `static`/`lazy` variant that could be resolved at load time instead.
    assert!(!engine_exl3.contains("static MCG_MULTIPLIER"));
    assert!(!engine_exl3.contains("lazy MCG_MULTIPLIER"));
}

// ---------------------------------------------------------------------------
// 4. What does switching the codebook actually do to the weights?
// ---------------------------------------------------------------------------

#[test]
fn switching_the_codebook_alone_changes_every_single_weight() {
    // This is the load-bearing result for any "just flip the flag" plan: the
    // codebook maps a trellis state to a scalar, so the two codebooks are two
    // different alphabets over the same bits. Same bytes + other codebook = a
    // different model, not a faster one.
    let mut differing = 0_usize;
    let mut max_gap = 0.0_f32;
    let mut agreeing = Vec::new();
    for state in 0_u32..=0xffff {
        let mcg = decode_state(state as u16, Codebook::Mcg);
        let mul1 = decode_state(state as u16, Codebook::Mul1);
        if mcg.to_bits() != mul1.to_bits() {
            differing += 1;
            max_gap = max_gap.max((mcg - mul1).abs());
        } else {
            agreeing.push((state, mcg));
        }
    }
    // They are overwhelmingly different functions. Not *quite* injective-different:
    // a handful of states coincide. The point survives, but the honest number is
    // 65530/65536, and the coincidence set is printed so it can be re-derived.
    println!(
        "MCG vs Mul1 over 2^16 states: differ on {differing}/65536, \
         agree on {} (states {:?}), max |delta| = {max_gap:.6}",
        agreeing.len(),
        agreeing.iter().map(|(s, _)| *s).collect::<Vec<_>>()
    );
    assert!(
        differing >= 65500,
        "MCG and Mul1 must be different functions; only {differing}/65536 differ"
    );
    assert!(max_gap > 0.1, "the disagreement should not be a rounding artefact");
    // And every 3-bit state -- the only ones a 3 bpw trellis stores -- differs.
    for state in 0_u32..8 {
        assert_ne!(
            decode_state(state as u16, Codebook::Mcg).to_bits(),
            decode_state(state as u16, Codebook::Mul1).to_bits(),
            "3 bpw state {state} decodes identically under both codebooks"
        );
    }

    // Their alphabets are also differently shaped, which is the compression
    // property: MCG is a hash (256-entry table, hash-like spread); Mul1 is a
    // monotone affine map of the byte sum (a graded ladder).
    let mut mcg_vals = (0_u32..=0xffff)
        .map(|s| decode_state(s as u16, Codebook::Mcg))
        .collect::<Vec<_>>();
    let mut mul1_vals = (0_u32..=0xffff)
        .map(|s| decode_state(s as u16, Codebook::Mul1))
        .collect::<Vec<_>>();
    let stat = |v: &mut Vec<f32>| -> (usize, f32, f32, f32) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let rms = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();
        let distinct = {
            let mut d = v.clone();
            d.dedup();
            d.len()
        };
        (distinct, v[0], v[v.len() - 1], rms)
    };
    let (mcg_distinct, mcg_min, mcg_max, mcg_rms) = stat(&mut mcg_vals);
    let (mul1_distinct, mul1_min, mul1_max, mul1_rms) = stat(&mut mul1_vals);
    println!("alphabet over all 2^16 trellis windows");
    println!("  MCG : distinct={mcg_distinct} min={mcg_min:.6} max={mcg_max:.6} rms={mcg_rms:.6}");
    println!("  Mul1: distinct={mul1_distinct} min={mul1_min:.6} max={mul1_max:.6} rms={mul1_rms:.6}");

    // A hash codebook should be near-surjective onto a wide range; the affine
    // one is a short graded ladder with far fewer reachable magnitudes.
    assert!(
        mcg_distinct > mul1_distinct,
        "expected MCG (hash) to have the larger alphabet, got mcg={mcg_distinct} mul1={mul1_distinct}"
    );
    assert!(mcg_rms > mul1_rms, "MCG should have the wider spread");
    assert!(mcg_max - mcg_min > mul1_max - mul1_min);

    // At 3 bpw the state that the trellis actually stores is `bits` wide, so the
    // reachable alphabet for a codebook is what it maps a 3-bit field onto.
    // This is the level count a 3 bpw tile actually chooses from, and it is the
    // number that decides reconstruction error, not the op count.
    let distinct_f32 = |mut values: Vec<f32>| -> Vec<f32> {
        values.sort_by(|a, b| a.to_bits().cmp(&b.to_bits()));
        values.dedup();
        values
    };
    for bits in [2_u32, 3, 4] {
        let mask = (1_u32 << bits) - 1;
        let mcg_levels = distinct_f32(
            (0..=mask)
                .map(|v| decode_state(v as u16, Codebook::Mcg))
                .collect(),
        );
        let mul1_levels = distinct_f32(
            (0..=mask)
                .map(|v| decode_state(v as u16, Codebook::Mul1))
                .collect(),
        );
        println!(
            "  {bits} bpw low-field levels: MCG={} Mul1={} (of 2^{bits}={})",
            mcg_levels.len(),
            mul1_levels.len(),
            mask + 1
        );
        // Both must offer the full 2^bits distinct reconstruction levels at the
        // field origin, or the quantiser could not use every code.
        assert_eq!(mcg_levels.len(), mask as usize + 1);
        assert_eq!(mul1_levels.len(), mask as usize + 1);
    }
}