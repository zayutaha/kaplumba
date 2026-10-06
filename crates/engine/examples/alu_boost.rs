//! `alu_boost` — what caps the 3 bpw decode GEMV below memory bandwidth?
//!
//! Model-free. The MSL is generated in Rust, so there is no `include_str!`
//! staleness hazard and every body's op count is a property of the generator.
//! Function names are concrete (`arm_seq`, ...) rather than C++ template
//! instantiations, whose names Metal mangles.
//!
//! # Part 1 — the access pattern, the DRAM ceiling, and the decode
//!
//! All arms except `seq` read the trellis with the SHIPPING ring indexing: one
//! 16x16 tile is 24 words at 3 bpw, and each of 32 lanes issues THREE word loads
//! (`j0`, `j1`, `j2`) out of that same 96-byte tile — 96 lane-loads for 24 words.
//! Bytes reported are UNIQUE bytes (tiles x 96), which is what reaches DRAM,
//! because j0/j1/j2 all land inside the tile and are served by L1.
//!
//! | arm | reads | ALU per code | question it answers |
//! |---|---|---|---|
//! | `seq` | coalesced, 1 word/lane, grid-strided | 1 | what is the DRAM ceiling at this footprint? |
//! | `ring3` | j0, j1, j2 | 1 (convert only) | does the ring pattern alone cap below that? |
//! | `ring1` | j0 only | 1 | is the 3-load amplification the cap? |
//! | `decode` | j0, j1, j2 | the real `decode_state_mcg` | what does the decode cost on top? |
//! | `decode2` | j0, j1, j2, two tiles per iteration | the real decode | does more work per iteration help? |
//!
//! `decode_state_mcg` is transcribed from `crates/kernels/src/linear.metal:119-126`;
//! the example checks the transcription against an independent Rust
//! re-implementation and refuses to report timings if they disagree.
//!
//! # Part 2 — the ALU issue ceiling, with an ILP sweep
//!
//! A latency-bound loop reports latency x chains, not issue capability, so the
//! chain count is swept 1..32 and the plateau is the ceiling. The two classes are
//! measured alone and together, at the SAME chain count, so "do the integer and
//! float pipes overlap?" needs no op counting — only wall time.
//!
//! # Reproduce
//!
//! ```sh
//! export CARGO_TARGET_DIR="$PWD/target"
//! cargo build --release -p kaplumba-engine --example alu_boost
//! scripts/gpuq try -- ./target/release/examples/alu_boost
//! ```

use metal::{Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const GROUPS: u64 = 320;
const TPT: u64 = 256; // 8 simdgroups per threadgroup
const WORDS: u64 = 1 << 26; // 256 MiB = 67.1 M words = 2.797 M tiles of 24 words
const ITERS: u32 = 20_000;

fn tiles() -> u32 {
    (WORDS / 24) as u32
}

/// The real EXL3 mcg decode. Transcribed from `linear.metal:119-126`.
const DECODE_MCG: &str = r#"
static inline float decode_mcg(uint value) {
    value = value * 0xCBAC1FEDu;
    value = 0x3b603b60u ^ (value & 0x8fff8fffu);
    half2 h = as_type<half2>(value);
    return float(h.x + h.y);
}
"#;

/// Shared ring indexing, emitted into every ring arm. Returns nothing: the
/// caller pastes this preamble and then its own body. Identical to the shipping
/// kernel's `exl3_decode8` prologue at ka = 3.
const RING_PREAMBLE: &str = r#"
    constexpr uint KA = 3u;
    constexpr uint RING_WORDS = KA * 8u;
    constexpr uint RING = RING_WORDS * 32u;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;
    const uint lo = (lane * 8u + 1u) * KA + RING - 16u;
    const uint sh = lo % 32u;
    const uint i0 = lo / 32u;
    const uint j0 = i0 >= RING_WORDS ? i0 - RING_WORDS : i0;
    const uint j1 = j0 + 1u >= RING_WORDS ? j0 + 1u - RING_WORDS : j0 + 1u;
    const uint j2 = j1 + 1u >= RING_WORDS ? j1 + 1u - RING_WORDS : j1 + 1u;
"#;

fn arm_seq() -> String {
    r#"
kernel void arm_seq(
    const device uint *packed [[buffer(0)]],
    constant P1 &p [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    // The textbook coalesced stream: one word per lane, grid-strided, with
    // EIGHT independent loads in flight per thread. Without the manual ILP this
    // loop measures load LATENCY, not bandwidth -- it reported 151 GB/s on a box
    // whose own probe reads 238-244 GB/s with the same inputs.
    float acc = 0.0f;
    uint i = gid;
    for (; i + 7u * p.stride < p.words; i += 8u * p.stride) {
        acc += float(packed[i])
             + float(packed[i + p.stride])
             + float(packed[i + 2u * p.stride])
             + float(packed[i + 3u * p.stride])
             + float(packed[i + 4u * p.stride])
             + float(packed[i + 5u * p.stride])
             + float(packed[i + 6u * p.stride])
             + float(packed[i + 7u * p.stride]);
    }
    for (; i < p.words; i += p.stride) acc += float(packed[i]);
    out[gid] = acc;
}
"#
    .to_string()
}

/// What the 8-code body does after the window is built.
#[derive(Clone, Copy, PartialEq)]
enum Body {
    /// 2 extractions, no decode: the cheapest thing that still builds a window.
    WindowOnly,
    /// All 8 extractions, summed as integers: prices the extraction alone.
    ExtractOnly,
    /// The real thing: 8 extractions + 8 `decode_state_mcg`, accumulated in f32.
    Decode,
    /// `decode_state_mcg` but accumulating the half2 in half precision and
    /// converting once. NOT bit-exact -- this is the ceiling probe for removing
    /// the per-code unpack/convert/add (4 ops) and keeping only one HADD2.
    DecodeHalf2Acc,
}

/// `loads`: 3 = j0,j1,j2 ; 1 = j0 only.
fn arm_ring(name: &str, loads: u32, body_kind: Body, tiles_per_iter: u32) -> String {
    let body = match body_kind {
        Body::WindowOnly => {
            r#"            const ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32u - sh)) : 0ul);
            acc += float(uint(a >> 32u) & 0xffffu) + float(uint(a) & 0xffffu);"#
        }
        Body::ExtractOnly => {
            r#"            const ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32u - sh)) : 0ul);
            const ulong c = ((w1 << 32) | w2) << sh;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                const uint r = i * KA + 16u;
                const uint sc = (r <= 64u) ? uint(a >> (64u - r)) : uint(c >> (96u - r));
                codes += int(sc & 0xffffu);
            }
            acc += float(codes);"#
        }
        Body::Decode => {
            r#"            const ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32u - sh)) : 0ul);
            const ulong c = ((w1 << 32) | w2) << sh;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                const uint r = i * KA + 16u;
                const uint sc = (r <= 64u) ? uint(a >> (64u - r)) : uint(c >> (96u - r));
                acc += decode_mcg(sc & 0xffffu);
            }"#
        }
        Body::DecodeHalf2Acc => {
            r#"            const ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32u - sh)) : 0ul);
            const ulong c = ((w1 << 32) | w2) << sh;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                const uint r = i * KA + 16u;
                const uint sc = (r <= 64u) ? uint(a >> (64u - r)) : uint(c >> (96u - r));
                uint v = (sc & 0xffffu) * 0xCBAC1FEDu;
                v = 0x3b603b60u ^ (v & 0x8fff8fffu);
                hacc += as_type<half2>(v);   // one HADD2, no unpack/cvt/add
            }
            acc += float(hacc.x + hacc.y);"#
        }
    };
    let needs_hacc = body_kind == Body::DecodeHalf2Acc;
    let hacc_decl = if needs_hacc {
        "    half2 hacc = half2(0.0h, 0.0h);\n"
    } else {
        ""
    };
    let codes_decl = if body_kind == Body::ExtractOnly {
        "    int codes = 0;\n"
    } else {
        ""
    };
    let decode_decl = if !needs_hacc {
        ""
    } else {
        ""
    };
    let _ = decode_decl;
    // Three separate word loads are needed for the decode body; for the
    // loads-only arms we let the compiler choose, but keep the same count.
    let (l0, l1, l2) = if loads == 3 {
        ("packed[base + j0]", "packed[base + j1]", "packed[base + j2]")
    } else {
        ("packed[base + j0]", "packed[base + j0]", "packed[base + j0]")
    };

    format!(
        r#"
kernel void {name}(
    const device uint *packed [[buffer(0)]],
    constant P1 &p [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{{
{RING_PREAMBLE}
{hacc_decl}{codes_decl}    float acc = 0.0f;
    for (uint t = (tgid * 8u + sg) * {tiles_per_iter}; t < p.tiles; t += p.groups * 8u * {tiles_per_iter}) {{
        #pragma unroll
        for (uint k = 0u; k < {tiles_per_iter}u; ++k) {{
            const uint base = (t + k) * RING_WORDS;
            const ulong w0 = ulong({l0});
            const ulong w1 = ulong({l1});
            const ulong w2 = ulong({l2});
{body}
        }}
    }}
    out[tgid * 8u + sg] = acc + float(lane);
}}
"#
    )
}

/// The SAME tile assignment and the SAME per-thread accumulation order as
/// `arm_decode`, unrolled four deep with the iteration stride. Because each
/// thread still consumes its own tiles in the same order, the floating-point
/// sum is bit-identical -- unlike the `tpi` arms, which repartition the tiles.
/// This is the shape a bit-exact shipping change would take.
fn arm_decode_u4() -> String {
    let decode_body = |indent: &str, tt: &str| {
        format!(
            r#"{indent}const uint base = ({tt}) * RING_WORDS;
{indent}const ulong w0 = ulong(packed[base + j0]);
{indent}const ulong w1 = ulong(packed[base + j1]);
{indent}const ulong w2 = ulong(packed[base + j2]);
{indent}const ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32u - sh)) : 0ul);
{indent}const ulong c = ((w1 << 32) | w2) << sh;
{indent}#pragma unroll
{indent}for (uint i = 0u; i < 8u; ++i) {{
{indent}    const uint r = i * KA + 16u;
{indent}    const uint sc = (r <= 64u) ? uint(a >> (64u - r)) : uint(c >> (96u - r));
{indent}    acc += decode_mcg(sc & 0xffffu);
{indent}}}
"#
        )
    };
    format!(
        r#"
kernel void arm_decode_u4(
    const device uint *packed [[buffer(0)]],
    constant P1 &p [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{{
{RING_PREAMBLE}
    float acc = 0.0f;
    const uint S = p.groups * 8u;
    uint t = tgid * 8u + sg;
    for (; t + 3u * S < p.tiles; t += 4u * S) {{
        #pragma unroll
        for (uint k = 0u; k < 4u; ++k) {{
{body0}        }}
    }}
    for (; t < p.tiles; t += S) {{
{body1}    }}
    out[tgid * 8u + sg] = acc + float(lane);
}}
"#,
        body0 = decode_body("            ", "(t + k * S)"),
        body1 = decode_body("        ", "t"),
    )
}

/// Emit one part-2 kind. Every chain is an independent accumulator in a
/// `#pragma unroll` loop with literal indices, so it stays in registers and the
/// ILP is exactly the chain count.
fn p2_kind(name: &str, nin: usize, nfp: usize) -> String {
    let mut body = String::new();
    for c in 0..nin {
        body.push_str(&format!(
            "            su[{c}] = su[{c}] * 0xCBAC1FEDu;\n\
             su[{c}] = 0x3b603b60u ^ (su[{c}] & 0x8fff8fffu);\n"
        ));
    }
    for c in 0..nfp {
        body.push_str(&format!(
            "            fa[{c}] = fma(fa[{c}], fm, fc);\n\
             fa[{c}] = fma(fa[{c}], fm, fc);\n"
        ));
    }
    let n = nin.max(1);
    let nf = nfp.max(1);
    let ni = nin.max(1);
    let su_decl = if nin == 0 {
        "uint su[1] = { 0u };".to_string()
    } else {
        format!("uint su[{nin}];")
    };
    let su_init = if nin == 0 {
        String::new()
    } else {
        format!(
            "    #pragma unroll\n    for (uint c = 0u; c < {nin}u; ++c) su[c] = 0x12345678u + tid * 2654435761u + c;\n"
        )
    };
    format!(
        r#"
kernel void {name}(
    device float *out [[buffer(0)]],
    constant uint &iters [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{{
    {su_decl}
{su_init}    float fa[{nf}];
    #pragma unroll
    for (uint c = 0u; c < {nf}u; ++c) fa[c] = float(tid) * 0.001f + float(c) + 1.0f;
    const float fm = 1.0009765625f;
    const float fc = 0.0001220703125f;
    float acc = 0.0f;
    for (uint i = 0u; i < iters; ++i) {{
        #pragma unroll
        for (uint c = 0u; c < {n}u; ++c) {{
{body}        }}
    }}
    #pragma unroll
    for (uint c = 0u; c < {nf}u; ++c) acc += fa[c];
    #pragma unroll
    for (uint c = 0u; c < {ni}u; ++c) acc += float(su[c]);
    out[tid] = acc;
}}
"#
    )
}

fn msl() -> String {
    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    s.push_str("struct P1 { uint tiles; uint groups; uint words; uint stride; };\n");
    s.push_str(DECODE_MCG);
    s.push_str(&arm_seq());
    s.push_str(&arm_ring("arm_ring3", 3, Body::WindowOnly, 1));
    s.push_str(&arm_ring("arm_ring1", 1, Body::WindowOnly, 1));
    s.push_str(&arm_ring("arm_extract", 3, Body::ExtractOnly, 1));
    s.push_str(&arm_ring("arm_decode", 3, Body::Decode, 1));
    s.push_str(&arm_ring("arm_decode2", 3, Body::Decode, 2));
    s.push_str(&arm_ring("arm_decode4", 3, Body::Decode, 4));
    s.push_str(&arm_ring("arm_decode8", 3, Body::Decode, 8));
    s.push_str(&arm_decode_u4());
    s.push_str(&arm_ring("arm_decode_h2", 3, Body::DecodeHalf2Acc, 1));
    for (name, ni, nf) in [
        ("p2_int4", 4, 0),
        ("p2_int8", 8, 0),
        ("p2_int16", 16, 0),
        ("p2_int32", 32, 0),
        ("p2_fp4", 0, 4),
        ("p2_fp8", 0, 8),
        ("p2_fp16", 0, 16),
        ("p2_fp32", 0, 32),
        ("p2_mix16", 16, 16),
        ("p2_mix32", 32, 32),
    ] {
        s.push_str(&p2_kind(name, ni, nf));
    }
    s
}

// ---------------------------------------------------------------- half math --
// The decode emits `float(h.x + h.y)`, i.e. a binary16 add widened to f32, so the
// reference needs a real round-to-nearest-even binary16.

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let mut e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut hm = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (hm & 1) == 1) {
            hm += 1;
        }
        return sign | hm;
    }
    let mut hm = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (hm & 1) == 1) {
        hm += 1;
        if hm == 0x400 {
            hm = 0;
            e += 1;
            if e >= 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | hm
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut e: i32 = -1;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        return f32::from_bits(sign | (((127 - 15 + 1 + e) as u32) << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

fn decode_mcg_ref(value: u32) -> f32 {
    let v = value.wrapping_mul(0xCBAC_1FED);
    let w = 0x3b60_3b60u32 ^ (v & 0x8fff_8fff);
    let lo = f16_to_f32((w & 0xffff) as u16);
    let hi = f16_to_f32((w >> 16) as u16);
    // `h.x + h.y` is a binary16 add in MSL, not an f32 one.
    f16_to_f32(f32_to_f16(lo + hi))
}

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(&msl(), &metal::CompileOptions::new())
        .expect("compile generated MSL");

    let tiles = tiles();
    let mut packed_data: Vec<u32> = Vec::with_capacity(WORDS as usize);
    let mut st = 0x1234_5678u32;
    for _ in 0..WORDS {
        st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        packed_data.push(st);
    }
    let packed = device.new_buffer_with_data(
        packed_data.as_ptr() as *const _,
        WORDS * 4,
        MTLResourceOptions::StorageModeShared,
    );
    // SIZED for the widest writer: `arm_seq` stores out[gid] with gid up to
    // GROUPS*TPT-1. Sizing this for GROUPS*8 wrote ~320 KiB past the end of the
    // buffer and faulted the command buffer, which `wait_until_completed`
    // reports as an instant completion -- a bogus 1.7 TB/s "DRAM ceiling".
    let out = device.new_buffer(GROUPS * TPT * 4, MTLResourceOptions::StorageModeShared);
    let p1buf = device.new_buffer(16, MTLResourceOptions::StorageModeShared);

    // Verify the decode transcription BEFORE timing anything.
    {
        let f = lib.get_function("arm_decode", None).expect("arm_decode");
        let st2 = device.new_compute_pipeline_state_with_function(&f).unwrap();
        set_p1(&p1buf, 1, 1, WORDS as u32, (GROUPS * 8 * TPT) as u32);
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&st2);
        e.set_buffer(0, Some(&packed), 0);
        e.set_buffer(1, Some(&p1buf), 0);
        e.set_buffer(2, Some(&out), 0);
        e.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        const KA: usize = 3;
        const RING_WORDS: usize = KA * 8;
        const RING: usize = RING_WORDS * 32;
        let lane = 0usize;
        let lo = (lane * 8 + 1) * KA + RING - 16;
        let sh = (lo % 32) as u32;
        let i0 = lo / 32;
        let j0 = i0 % RING_WORDS;
        let j1 = (j0 + 1) % RING_WORDS;
        let j2 = (j1 + 1) % RING_WORDS;
        let w0 = packed_data[j0] as u64;
        let w1 = packed_data[j1] as u64;
        let w2 = packed_data[j2] as u64;
        let a = (((w0 << 32) | w1) << sh) | if sh > 0 { w2 >> (32 - sh) } else { 0 };
        let mut expect = 0.0f64;
        for i in 0..8usize {
            let r = i * KA + 16;
            let s = (a >> (64 - r)) as u32;
            expect += decode_mcg_ref(s & 0xffff) as f64;
        }
        let got = unsafe { *(out.contents() as *const f32) } as f64;
        let diff = (got - expect).abs();
        println!("  [check] decode lane0 tile0   kernel {got:.9}  rust {expect:.9}  |d| {diff:.3e}");
        assert!(diff < 1e-6, "decode transcription does NOT match: {diff}");
        println!("  [check] decode transcription verified against independent Rust reference");
    }

    // Does the 4-deep unroll preserve the bits? Run both arms over the whole
    // input and compare every output word. Same mapping + same order => the two
    // must agree exactly; if they do not, the unroll is NOT bit-exact and the
    // speedup it reports is not shippable under the no-quality-loss constraint.
    {
        let mut outs: Vec<Vec<u32>> = Vec::new();
        for name in ["arm_decode", "arm_decode_u4"] {
            let f = lib.get_function(name, None).expect(name);
            let st2 = device.new_compute_pipeline_state_with_function(&f).unwrap();
            let cbbuf = device.new_buffer(2560 * 4, MTLResourceOptions::StorageModeShared);
            set_p1(&p1buf, tiles, GROUPS as u32, WORDS as u32, (GROUPS * TPT) as u32);
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&st2);
            e.set_buffer(0, Some(&packed), 0);
            e.set_buffer(1, Some(&p1buf), 0);
            e.set_buffer(2, Some(&cbbuf), 0);
            e.dispatch_thread_groups(MTLSize::new(GROUPS, 1, 1), MTLSize::new(TPT, 1, 1));
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let p = cbbuf.contents() as *const u32;
            outs.push(unsafe { std::slice::from_raw_parts(p, 2560) }.to_vec());
        }
        let differing = outs[0]
            .iter()
            .zip(outs[1].iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "  [check] arm_decode vs arm_decode_u4 over {} words: {differing} differ{}  -> {}",
            2560,
            "",
            if differing == 0 {
                "BIT-EXACT (the unroll is shippable under no-quality-loss)"
            } else {
                "NOT bit-exact"
            }
        );
        assert_eq!(differing, 0, "the 4-deep unroll is not bit-exact");
    }

    // ------------------------------------------------------------ part 1 ---
    println!();
    println!(
        "part 1 — {:.1} MiB unique bytes ({tiles} tiles of 96 B), medians of 7",
        tiles as f64 * 96.0 / 1048576.0
    );
    println!(
        "  {:<12} {:>9} {:>9} {:>9}   {}",
        "arm", "ms", "GB/s", "ns/tile", "what it does"
    );
    let arms = [
        ("arm_seq", "coalesced 1 word/lane, 8-way ILP — the DRAM ceiling"),
        ("arm_ring1", "ring pattern, 1 load/lane/tile, window only"),
        ("arm_ring3", "ring pattern, 3 loads/lane/tile, window only"),
        ("arm_extract", "ring pattern + all 8 extractions, NO decode"),
        ("arm_decode", "ring pattern + REAL decode (the shipping shape)"),
        ("arm_decode2", "real decode, 2 tiles/lane/iteration"),
        ("arm_decode4", "real decode, 4 tiles/lane/iteration"),
        ("arm_decode8", "real decode, 8 tiles/lane/iteration (repartitions)"),
        ("arm_decode_u4", "real decode, 4-deep unroll, SAME mapping (bit-exact)"),
        ("arm_decode_h2", "decode, half2 accumulator (NOT bit-exact: a ceiling)"),
    ];
    let mut gbs = [0.0f64; 10];
    let mut per_arm: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut states: Vec<metal::ComputePipelineState> = Vec::new();
    for (name, _) in &arms {
        let f = lib
            .get_function(name, None)
            .unwrap_or_else(|_| panic!("{name} missing; have {:?}", lib.function_names()));
        states.push(device.new_compute_pipeline_state_with_function(&f).unwrap());
    }
    let threads = (GROUPS * TPT) as u32;
    set_p1(&p1buf, tiles, GROUPS as u32, WORDS as u32, threads);
    // INTERLEAVED rounds: one pass over every arm per round. Two sequential
    // sweeps of this same code drifted 10% apart on `arm_seq` alone
    // (189.7 -> 210.4 GB/s), so a non-interleaved comparison here is worthless.
    const ROUNDS: usize = 9;
    for _round in 0..ROUNDS {
        for (k, _) in arms.iter().enumerate() {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&states[k]);
            e.set_buffer(0, Some(&packed), 0);
            e.set_buffer(1, Some(&p1buf), 0);
            e.set_buffer(2, Some(&out), 0);
            e.dispatch_thread_groups(MTLSize::new(GROUPS, 1, 1), MTLSize::new(TPT, 1, 1));
            e.end_encoding();
            let t = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            per_arm[k].push(t.elapsed().as_secs_f64());
        }
    }
    for (k, (name, note)) in arms.iter().enumerate() {
        let times = &mut per_arm[k];
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = times[times.len() / 2];
        let bytes = if *name == "arm_seq" {
            WORDS as f64 * 4.0
        } else {
            tiles as f64 * 96.0
        };
        gbs[k] = bytes / m / 1e9;
        println!(
            "  {:<12} {:>9.3} {:>9.1} {:>9.3}   {}   [spread {:.3}-{:.3}]",
            name,
            m * 1e3,
            gbs[k],
            m * 1e9 / tiles as f64,
            note,
            times[0] * 1e3,
            times[times.len() - 1] * 1e3
        );
    }
    {
        let s = &per_arm[0];
        let (lo, hi) = (s[0], s[s.len() - 1]);
        println!(
            "  [control] arm_seq across {ROUNDS} interleaved rounds: {:.3}-{:.3} ms = {:.1}% drift",
            lo * 1e3,
            hi * 1e3,
            100.0 * (hi / lo - 1.0)
        );
    }
    println!();
    println!("  DRAM ceiling (seq)                  {:.1} GB/s", gbs[0]);
    println!(
        "  ring pattern alone (ring3)          {:.1} GB/s  = {:.0}% of seq   [pattern costs {:.0}%]",
        gbs[2],
        100.0 * gbs[2] / gbs[0],
        100.0 * (1.0 - gbs[2] / gbs[0])
    );
    println!(
        "  + all 8 extractions (extract)       {:.1} GB/s  = {:.0}% of ring3  [extraction costs {:.0}%]",
        gbs[3],
        100.0 * gbs[3] / gbs[2],
        100.0 * (1.0 - gbs[3] / gbs[2])
    );
    println!(
        "  + the 8 decodes (decode)            {:.1} GB/s  = {:.0}% of extract [decode costs {:.0}%]",
        gbs[4],
        100.0 * gbs[4] / gbs[3],
        100.0 * (1.0 - gbs[4] / gbs[3])
    );
    println!(
        "  decode end-to-end vs coalesced      {:.0}% of seq  ({:.1} -> {:.1} GB/s)",
        100.0 * gbs[4] / gbs[0],
        gbs[0],
        gbs[4]
    );
    println!(
        "  SAME mapping, 4-deep unroll (BIT-EXACT) {:.1} GB/s = {:.3}x decode",
        gbs[8],
        gbs[8] / gbs[4]
    );
    println!(
        "  half2 accumulator (NOT bit-exact)       {:.1} GB/s = {:.3}x decode  [dead]",
        gbs[9],
        gbs[9] / gbs[4]
    );
    println!("  loop ILP (tiles per lane per iteration), NOT the same mapping:");
    for (k, tpi) in [1usize, 2, 4, 8].iter().enumerate() {
        let idx = [4usize, 5, 6, 7][k];
        println!(
            "    tpi={tpi}  {:>7.1} GB/s   = {:.3}x  vs tpi=1",
            gbs[idx],
            gbs[idx] / gbs[4]
        );
    }
    let _ = gbs[1];

    // ------------------------------------------------------------ part 2 ---
    println!();
    println!("part 2 — ALU issue ceiling vs chain count (ILP sweep), medians of 5");
    let threads = GROUPS * TPT;
    let buf = device.new_buffer(threads * 4, MTLResourceOptions::StorageModeShared);
    let itersbuf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe {
        *(itersbuf.contents() as *mut u32) = ITERS;
    }
    let mut res: Vec<(String, f64)> = Vec::new();
    let mut p2_states: Vec<(&str, metal::ComputePipelineState)> = Vec::new();
    let mut p2_times: Vec<Vec<f64>> = Vec::new();
    for name in [
        "p2_int4", "p2_int8", "p2_int16", "p2_int32", "p2_fp4", "p2_fp8", "p2_fp16", "p2_fp32",
        "p2_mix16", "p2_mix32",
    ] {
        let f = lib
            .get_function(name, None)
            .unwrap_or_else(|_| panic!("{name} missing; have {:?}", lib.function_names()));
        p2_states.push((name, device.new_compute_pipeline_state_with_function(&f).unwrap()));
        p2_times.push(Vec::new());
    }
    for _round in 0..7 {
        for k in 0..p2_states.len() {
            let st2 = &p2_states[k].1;
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(st2);
            e.set_buffer(0, Some(&buf), 0);
            e.set_buffer(1, Some(&itersbuf), 0);
            e.dispatch_thread_groups(MTLSize::new(GROUPS, 1, 1), MTLSize::new(TPT, 1, 1));
            e.end_encoding();
            let t = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            p2_times[k].push(t.elapsed().as_secs_f64());
        }
    }
    for (k, (name, _)) in p2_states.iter().enumerate() {
        let name: &str = name;
        let times = &mut p2_times[k];
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = times[times.len() / 2];
        // Warp-instructions per second: one issued instruction moves 32 lanes.
        // Chains are independent, so the instruction count is the patch length.
        let chains: f64 = match name {
            "p2_int4" | "p2_fp4" => 4.0,
            "p2_int8" | "p2_fp8" => 8.0,
            "p2_int16" | "p2_fp16" | "p2_mix16" => 16.0,
            _ => 32.0,
        };
        let instrs_per_iter = match name {
            "p2_int4" | "p2_int8" | "p2_int16" | "p2_int32" => chains * 2.0, // mul + lop3
            "p2_fp4" | "p2_fp8" | "p2_fp16" | "p2_fp32" => chains * 2.0,    // 2 fma
            "p2_mix16" => 16.0 * 2.0 + 16.0 * 2.0,
            _ => 32.0 * 4.0,
        };
        let warps = threads / 32;
        let rate = instrs_per_iter * ITERS as f64 * warps as f64 / m / 1e9;
        res.push((name.to_string(), rate));
        println!(
            "  {name:<10} {:.3} ms   {:.1} G warp-instr/s   [{:.3}-{:.3}]",
            m * 1e3,
            rate,
            times[0] * 1e3,
            times[times.len() - 1] * 1e3
        );
    }
    println!();
    let get = |n: &str| res.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap();
    println!(
        "  int plateau 16 chains {:.1} G warp-instr/s   fp plateau 16 chains {:.1} G",
        get("p2_int16"),
        get("p2_fp16")
    );
    println!(
        "  mixed(16+16) {:.1} G vs max(int16, fp16) {:.1} G -> {:.2}x  (>1 => the int and fp pipes overlap)",
        get("p2_mix16"),
        get("p2_int16").max(get("p2_fp16")),
        get("p2_mix16") / get("p2_int16").max(get("p2_fp16"))
    );
    println!(
        "  if one shared port, mixed(16+16) would be ~int16+fp16 = {:.1} G; measured {:.1} G",
        get("p2_int16") + get("p2_fp16"),
        get("p2_mix16")
    );
}

fn set_p1(buf: &metal::Buffer, tiles: u32, groups: u32, words: u32, stride: u32) {
    unsafe {
        let p = buf.contents() as *mut u32;
        std::ptr::write(p, tiles);
        std::ptr::write(p.add(1), groups);
        std::ptr::write(p.add(2), words);
        std::ptr::write(p.add(3), stride);
    }
}
