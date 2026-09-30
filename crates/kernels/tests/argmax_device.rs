//! A device-side argmax that agrees with the host `argmax_logits`, and proof
//! that the existing device-side one does not.
//!
//! The claim under test (`crates/engine/src/model.rs:3512-3525`,
//! `generate.rs:782-787`): `greedy_key` in `elements.metal:529-536` packs
//! `0xFFFF - index` into 16 bits, which underflows for every index at or past
//! 65,536, so the packed key stops ordering anything and the reduction returns
//! a low token id regardless of the logits.
//!
//! Three things are established here:
//!
//!  1. the underflow, in integer arithmetic, plus the shipping kernel's actual
//!     return value on the real 262,144 shape (it returns token 0);
//!  2. a replacement that picks the SAME token as
//!     [`argmax_logits`](crate::generate) on 65,574 rows, including all 65,536
//!     fp16 bit patterns, ties, +/-0, +/-Inf, NaN, and winners at and past
//!     index 65,536;
//!  3. that the test in (2) CAN fail: it is re-run against a deliberately
//!     broken packing and is required to disagree (AGENTS.md 3.4).
//!
//! STATUS: this file has not been compiled in-tree — `crates/kernels` does not
//! build at the moment (`crates/kernels/src/attention.rs`, another agent's
//! uncommitted work, fails to compile). The identical Metal source, row set and
//! assertions were run as a standalone crate against this same M4 Pro with raw
//! metal-rs; see `spec/decode-19tok-repacked-250gb/implementation_notes.md`.
//! Only the command-buffer plumbing below differs, so run this before trusting
//! it.

use half::f16;
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Host reference, copied verbatim from crates/engine/src/generate.rs:416-427.
// Tie rule: strict `>` from index 0, so the FIRST (lowest) index wins a tie.
// ---------------------------------------------------------------------------
fn argmax_logits(logits: &[u16]) -> u32 {
    let mut best = 0;
    let mut best_val = f16::from_bits(logits[0]).to_f32();
    for index in 1..logits.len() {
        let val = f16::from_bits(logits[index]).to_f32();
        if val > best_val {
            best_val = val;
            best = index;
        }
    }
    best as u32
}

/// The shipping packing, on the host. `elements.metal:529-536`.
fn greedy_key_shipping(bits: u16, index: u32) -> u32 {
    let sortable = if bits & 0x8000 != 0 {
        !u32::from(bits)
    } else {
        u32::from(bits) ^ 0x8000
    };
    (sortable << 16) | (0xFFFFu32.wrapping_sub(index))
}

/// `elements.metal:570`: `0xFFFFu - (*best % 0x10000u)`.
fn greedy_token_shipping(best: u32) -> u32 {
    0xFFFFu32.wrapping_sub(best % 0x10000)
}

/// Elements per atomic word. A 32-bit key holds 16 bits of value and 16 bits
/// of index, so 2^16 indices share one word and a 262,144-entry vocabulary
/// needs 4 words for the host to fold.
const SLOT_SIZE: usize = 0x1_0000;

// ---------------------------------------------------------------------------
// Metal source
// ---------------------------------------------------------------------------

const PROLOGUE: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline uint sortable16(uint bits) {
    return (bits & 0x8000u) != 0u ? ~bits : (bits ^ 0x8000u);
}
"#;

/// Monotone in the fp16 order, with the two places the host's `f32`
/// comparison is not a plain ordering folded away:
///   * NaN never wins (the host's `NaN > best_val` is false, forever), and
///   * -0.0 and +0.0 get the SAME key (the host compares them equal, so the
///     tie rule -- lowest index -- decides).
const VALUE_KEY_FIXED: &str = r#"
static inline uint value_key(uint bits) {
    const uint EXP = (bits >> 10) & 0x1Fu;
    const uint MAN = bits & 0x3FFu;
    if (EXP == 0x1Fu && MAN != 0u) {
        return 0u;                     // below -inf (0x03FF): NaN can never win
    }
    if ((bits & 0x7FFFu) == 0u) {
        return 0x8000u;                // +0.0: -0.0 folds onto it
    }
    return sortable16(bits);
}
"#;

/// The shipping value key: no NaN handling, no zero folding.
const VALUE_KEY_SHIPPING: &str = r#"
static inline uint value_key(uint bits) {
    return sortable16(bits);
}
"#;

/// 16-bit value over a 16-bit index-WITHIN-SLOT. Taking the index modulo the
/// slot width is what makes the field wide enough for its range, so
/// `0xFFFF - index_in_slot` cannot underflow whatever the vocabulary is.
const KEY_FIXED: &str = r#"
static inline uint argmax_key(uint bits, uint index) {
    return (value_key(bits) << 16) | ((0xFFFFu - (index & 0xFFFFu)) & 0xFFFFu);
}
"#;

/// The shipping packing: 16 + 16 with the WHOLE index in the low field, so it
/// underflows past 65,535 and the OR clobbers the value field as well.
const KEY_SHIPPING: &str = r#"
static inline uint argmax_key(uint bits, uint index) {
    return (value_key(bits) << 16) | (0xFFFFu - index);
}
"#;

const KERNELS: &str = r#"
// One params STRUCT, not two scalar `constant uint &` references. With two
// scalars and one 8-byte `set_bytes` at index 2, Metal filled slot 2 and left
// slot 3 zero, so `per_slot` read as 0, the kernel divided by it, and the GPU
// hung with no diagnostic. Every other kernel here passes a struct for exactly
// this reason (SoftcapParams, NormRopeQkvParams, ...).
struct ArgmaxParams {
    uint count;
    uint per_slot;
};

// ONE dispatch, and the only precondition is that the caller partitions
// threadgroups over slots.
//
// The grid is `slots * per_slot` threadgroups of `tgdim` threads. Threadgroup
// `tgid` owns slot `tgid / per_slot` and nothing else, so the strided loop
// never leaves its slot: it walks that slot's contiguous span, which is also
// the coalescing pattern (consecutive lanes read consecutive elements). One
// relaxed 32-bit atomic per SIMDGROUP, into the slot it owns.
//
// The slot partition is not an optimisation, it is what makes the reduction
// correct. An earlier version keyed the atomic off the thread's own index and
// strided over the whole row, so one SIMDGROUP covered two slots and wrote
// only one of them; a second version reduced (key, index) per threadgroup and
// also wrote only one slot per threadgroup. Both silently left slots 2 and 3
// at zero and still agreed with the host on every row whose winner happened to
// be in slot 0 or 1. The all-equal rows are what exposed them.
kernel void argmax_slots(const device half *logits [[buffer(0)]],
                         device atomic_uint *best [[buffer(1)]],
                         constant ArgmaxParams &params [[buffer(2)]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint tgdim [[threads_per_threadgroup]]) {
    const uint per_slot = params.per_slot;
    const uint slot = tgid / per_slot;
    const uint lo = slot * SLOT_SIZE;
    const uint hi = min(lo + SLOT_SIZE, params.count);
    uint local = 0u;
    const uint stride = per_slot * tgdim;
    for (uint i = lo + (tgid % per_slot) * tgdim + lid; i < hi; i += stride) {
        // `i & 0xFFFFu` is `i - lo`: `lo` is a multiple of SLOT_SIZE and the
        // span is at most SLOT_SIZE wide, so the packed index field is the
        // in-slot index and cannot underflow.
        local = max(local, argmax_key(uint(as_type<ushort>(logits[i])), i));
    }
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        local = max(local, simd_shuffle_xor(local, offset));
    }
    // EVERY simdgroup, not just the first: `simd_shuffle_xor` reduces within a
    // simdgroup, so a 256-thread threadgroup holds 8 independent results and a
    // `lid < 32` guard threw seven of them away. That is how an early version
    // of this kernel returned 65,535 for a row whose winner was 9,259, with no
    // error anywhere.
    if (lane == 0u) {
        atomic_fetch_max_explicit(&best[slot], local, memory_order_relaxed);
    }
}

// The shipping kernel, verbatim, for the root-cause measurement.
static inline uint greedy_key_ship(half value, uint index) {
    const uint SIGN = 0x8000u;
    uint bits = uint(as_type<ushort>(value));
    uint sortable = (bits & SIGN) != 0u ? ~bits : (bits ^ SIGN);
    return (sortable << 16) | (0xFFFFu - index);
}

kernel void greedy_pick(const device half *logits [[buffer(0)]],
                        device atomic_uint *best [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint simd_lane [[thread_index_in_simdgroup]]) {
    uint local = 0u;
    for (uint i = simd_lane; i < count; i += 32u) {
        local = max(local, greedy_key_ship(logits[i], i));
    }
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        local = max(local, simd_shuffle_xor(local, offset));
    }
    if (simd_lane == 0u) {
        atomic_fetch_max_explicit(best, local, memory_order_relaxed);
    }
}

kernel void greedy_token(const device atomic_uint *best [[buffer(0)]],
                         device uint *out [[buffer(1)]],
                         constant uint &row [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) {
        return;
    }
    out[row] = 0xFFFFu - (*(const device uint *)best) % 0x10000u;
}
"#;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Packing {
    Fixed,
    Shipping,
}

impl Packing {
    fn source(self) -> String {
        let (value_key, key) = match self {
            Packing::Fixed => (VALUE_KEY_FIXED, KEY_FIXED),
            Packing::Shipping => (VALUE_KEY_SHIPPING, KEY_SHIPPING),
        };
        // The shipping packing has one slot (all 65,536 of them underflowing),
        // the fixed one has ceil(count / 65536). Compiling SLOT_SIZE to a
        // constant lets the compiler turn the multiplies into shifts.
        let slot_size = match self {
            Packing::Fixed => SLOT_SIZE as u32,
            Packing::Shipping => 0xFFFF_FFFF,
        };
        format!("{PROLOGUE}\n{value_key}\n{key}\n#define SLOT_SIZE {slot_size}u\n{KERNELS}")
    }

    /// Words the host has to fold. 1 for the shipping single-word key.
    fn slots(self, count: usize) -> usize {
        match self {
            Packing::Fixed => count.div_ceil(SLOT_SIZE),
            Packing::Shipping => 1,
        }
    }

    /// The host-side fold.
    fn token(self, keys: &[u32]) -> u32 {
        let mut best_slot = 0usize;
        let mut best = keys[0];
        for slot in 1..keys.len() {
            // Strict `>` over the VALUE field only: on a tie the lowest slot
            // wins, and a lower slot holds only lower indices, which is the
            // host's tie rule.
            if (keys[slot] >> 16) > (best >> 16) {
                best = keys[slot];
                best_slot = slot;
            }
        }
        let in_slot = 0xFFFFu32.wrapping_sub(best & 0xFFFF);
        match self {
            Packing::Fixed => (best_slot as u32) * SLOT_SIZE as u32 + in_slot,
            // The shipping kernel has one word and decodes it with
            // `greedy_token`'s expression, which is the index only while the
            // index never exceeded 65,535.
            Packing::Shipping => 0xFFFFu32.wrapping_sub(best % 0x10000),
        }
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Probe {
    device: Device,
    queue: metal::CommandQueue,
    slots: ComputePipelineState,
    pick: ComputePipelineState,
    token: ComputePipelineState,
    groups: u64,
    tg: u64,
}

const GROUPS: u64 = 512;
const TG: u64 = 256;

impl Probe {
    fn new(packing: Packing) -> Probe {
        let device = Device::system_default().expect("Metal device");
        let source = packing.source();
        let library: Library = device
            .new_library_with_source(&source, &CompileOptions::new())
            .unwrap_or_else(|e| panic!("shader compile failed:\n{e}\n---\n{source}"));
        let pipe = |name: &str| -> ComputePipelineState {
            let function = library
                .get_function(name, None)
                .unwrap_or_else(|e| panic!("no function {name}: {e}"));
            device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap_or_else(|e| panic!("no pipeline {name}: {e}"))
        };
        let slots = pipe("argmax_slots");
        let pick = pipe("greedy_pick");
        let token = pipe("greedy_token");
        let queue = device.new_command_queue();
        Probe {
            device,
            queue,
            slots,
            pick,
            token,
            groups: GROUPS,
            tg: TG,
        }
    }

    /// Change the grid without rebuilding the pipelines.
    fn set_grid(&mut self, groups: u64, tg: u64) {
        assert!(tg <= 1024, "threads_per_threadgroup is capped at 1024");
        self.groups = groups;
        self.tg = tg;
    }

    fn upload(&self, row: &[u16]) -> Buffer {
        let buffer = self.device.new_buffer(
            (row.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(row.as_ptr(), buffer.contents() as *mut u16, row.len());
        }
        buffer
    }

    fn zeroed(&self, bytes: u64) -> Buffer {
        let buffer = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(buffer.contents() as *mut u8, 0, bytes as usize) };
        buffer
    }

    /// One dispatch. `per_slot` threadgroups per slot, `groups` in total.
    fn keys(&self, row: &Buffer, n: usize, packing: Packing, per_slot: u64) -> Vec<u32> {
        let slots = packing.slots(n);
        assert_eq!(
            per_slot * slots as u64,
            self.groups,
            "the grid must cover the slots"
        );
        let best = self.zeroed((slots * 4) as u64);
        let mut params = (n as u32).to_le_bytes().to_vec();
        params.extend_from_slice(&(per_slot as u32).to_le_bytes());
        let mut encoder = MetalEncoder::new(&self.queue).expect("encoder");
        encoder
            .dispatch_groups(
                &self.slots,
                &[Some(row), Some(&best)],
                &params,
                self.groups,
                self.tg,
            )
            .expect("argmax dispatch");
        encoder.commit_and_wait().expect("commit");
        let out = unsafe { std::slice::from_raw_parts(best.contents() as *const u32, slots) };
        out.to_vec()
    }

    /// The shipping two-dispatch path, as `encode_greedy_pick` drives it.
    fn token_shipping(&self, row: &Buffer, n: usize) -> u32 {
        let best = self.zeroed(4);
        let out = self.zeroed(4);
        let groups = ((n as u64 / 32).min(4096)).max(1);
        let mut encoder = MetalEncoder::new(&self.queue).expect("encoder");
        encoder
            .dispatch_groups(
                &self.pick,
                &[Some(row), Some(&best)],
                &(n as u32).to_le_bytes(),
                groups,
                32,
            )
            .expect("greedy_pick");
        // `encode_greedy_pick` refuses row == 0, so use the row the engine's
        // caller used and then read slot 0 of the output.
        encoder
            .dispatch_1d(
                &self.token,
                &[Some(&best), Some(&out)],
                &1u32.to_le_bytes(),
                1,
                1,
            )
            .expect("greedy_token");
        encoder.commit_and_wait().expect("commit");
        unsafe { *(out.contents() as *const u32) }
    }

    /// GPU own time for one dispatch, from the GPU's own clock.
    fn gpu_own_ms(&self, row: &Buffer, n: usize, per_slot: u64, iters: usize) -> Vec<f64> {
        let best = self.zeroed((n.div_ceil(SLOT_SIZE) * 4) as u64);
        let mut params = (n as u32).to_le_bytes().to_vec();
        params.extend_from_slice(&(per_slot as u32).to_le_bytes());
        let mut samples = Vec::new();
        for _ in 0..iters {
            unsafe { std::ptr::write_bytes(best.contents() as *mut u8, 0, 4) };
            let mut encoder = MetalEncoder::new(&self.queue).expect("encoder");
            encoder
                .dispatch_groups(
                    &self.slots,
                    &[Some(row), Some(&best)],
                    &params,
                    self.groups,
                    self.tg,
                )
                .expect("dispatch");
            encoder.commit_stamped_and_wait("argmax").expect("commit");
            for stamp in encoder.take_gpu_stamps() {
                samples.push(stamp.own_ms);
            }
        }
        samples
    }
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

const N: usize = 262_144;

struct Case {
    name: String,
    row: Vec<u16>,
    /// `false` where the host and any value-ordered device reduction provably
    /// cannot agree. See `exact` in [`cases`].
    exact: bool,
    /// Run this row on every grid shape. The 65,536 exhaustive rows are for
    /// value-ordering coverage and only need one shape each; they cost a
    /// command-buffer round trip apiece.
    every_grid: bool,
}

fn f(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
}

fn cases() -> Vec<Case> {
    let mut out: Vec<Case> = Vec::new();
    // `exact` is DERIVED, not hand-set: the host latches `best_val = NaN` when
    // the FIRST logit is NaN and then returns 0 unconditionally, which no
    // value-ordered device reduction can reproduce unless index 0 also holds
    // the row's maximum. That is the one and only disagreement, and it only
    // arises on a row the model has already destroyed.
    let mut push = |name: &str, row: Vec<u16>| {
        let exact = !f16::from_bits(row[0]).to_f32().is_nan();
        out.push(Case {
            name: name.to_string(),
            row,
            exact,
            every_grid: true,
        });
    };

    let mut r = vec![f(0.0); N];
    push("all equal, +0.0", r.clone());
    for (i, v) in r.iter_mut().enumerate() {
        if i % 2 == 1 {
            *v = f(-0.0);
        }
    }
    push("all equal, -0.0/+0.0 mix", r.clone());
    r.iter_mut().for_each(|v| *v = f(-7.0));
    push("all equal, -7.0", r.clone());
    r.iter_mut().for_each(|v| *v = f16::NAN.to_bits());
    push("all NaN", r.clone());
    r.iter_mut().for_each(|v| *v = f16::INFINITY.to_bits());
    push("all +Inf", r.clone());
    r.iter_mut().for_each(|v| *v = f16::NEG_INFINITY.to_bits());
    push("all -Inf", r.clone());

    // -0.0 and +0.0 tie in the host's f32 comparison, so the lowest index wins.
    let mut r = vec![f(-1.0); N];
    r[0] = f(-0.0);
    r[1] = f(0.0);
    r[2] = f(-0.0);
    push("-0.0 at 0, +0.0 at 1, -0.0 at 2, rest -1", r);

    // NaN mixed into a normal row: never selected, on either side.
    let mut r = vec![f(-2.0); N];
    r[123] = f16::NAN.to_bits();
    r[65_536] = f16::NAN.to_bits();
    r[262_143] = f16::NEG_INFINITY.to_bits() | 0x0200; // -NaN
    r[200_000] = f(3.5);
    push("NaN at 123/65536, -NaN at 262143, winner 200000", r);

    // NaN at index 0: the host latches best_val = NaN and can never update, so
    // it returns 0 even though 9,259 holds 28.5. Provably different; the
    // derived `exact` flag is what keeps this row out of the equality set.
    let mut r = vec![f(-2.0); N];
    r[0] = f16::NAN.to_bits();
    r[9_259] = f(28.5);
    push("NaN at index 0, winner 9259 (host returns 0)", r);

    // The measured real-checkpoint row: winner 9,259 at 28.5, with most of the
    // vocabulary past 65,536.
    let mut r = vec![f(-5.0); N];
    r[9_259] = f(28.5);
    push("checkpoint shape: winner 9259 at 28.5", r);

    let mut r = vec![f(-5.0); N];
    r[N - 1] = f(28.5);
    push("winner at 262143", r);
    let mut r = vec![f(-5.0); N];
    r[65_536] = f(28.5);
    push("winner at 65536", r);
    let mut r = vec![f(-5.0); N];
    r[65_535] = f(28.5);
    r[65_537] = f(28.5);
    push("tie at 65535 and 65537 -> 65535", r);
    let mut r = vec![f(-5.0); N];
    r[65_535] = f(28.5);
    r[65_537] = f(28.5);
    r[131_071] = f(28.5);
    r[196_608] = f(28.5);
    push("4-way tie across all four slots -> 65535", r);

    for &(count, at) in &[
        (2usize, 4usize),
        (7, 0),
        (7, 1),
        (3, 65_536),
        (5, 262_130),
        (2, 131_072),
    ] {
        let mut r = vec![f(1.0); N];
        for k in 0..count {
            r[at + k] = f(9.0);
        }
        push(&format!("tie x{count} at {at}"), r);
    }

    // One fp16 ulp apart across the 65,536 boundary.
    let mut r = vec![f(-5.0); N];
    r[65_535] = f(28.5);
    r[65_536] = f(28.5 + 0.015625);
    push("one ulp apart across 65536", r);

    for &n in &[
        1usize, 2, 31, 32, 33, 255, 256, 257, 1000, 65_535, 65_536, 65_537, 131_072, 131_073, N,
    ] {
        let mut s = 0x1234_5678_9abc_def0u64 ^ (n as u64);
        let mut r = vec![0u16; n];
        for v in r.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((s >> 33) as f32) / ((1u64 << 31) as f32);
            *v = f(unit * 40.0 - 20.0);
        }
        push(&format!("random n={n}"), r);
    }

    for &n in &[65_537usize, N] {
        let mut s = 0xfeed_face_cafe_0001u64 ^ (n as u64);
        let mut r = vec![0u16; n];
        for v in r.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = f(((s >> 33) % 9) as f32 * 3.0 - 12.0);
        }
        push(&format!("tie-heavy random (9 distinct) n={n}"), r);
    }

    // Exhaustive: every fp16 bit pattern as a 3-element row, so every tie,
    // every +/-0 pair, every Inf and every NaN sits next to its immediate
    // neighbours in both orders.
    for a in 0..=0xFFFFu32 {
        let row = vec![a as u16, ((a + 1) & 0xFFFF) as u16, a as u16];
        let exact = !f16::from_bits(row[0]).to_f32().is_nan();
        out.push(Case {
            name: format!("exhaustive {a:#06x}"),
            row,
            exact,
            every_grid: false,
        });
    }

    out
}

// ---------------------------------------------------------------------------
// 1. Root cause
// ---------------------------------------------------------------------------

#[test]
fn shipping_packing_underflows_past_65_535() {
    // The arithmetic, exactly as `elements.metal:535` evaluates it.
    for index in [
        0u32, 1, 65_535, 65_536, 65_537, 65_538, 100_000, 131_071, 131_072, 196_608, 262_143,
    ] {
        let low = 0xFFFFu32.wrapping_sub(index);
        let sortable = u32::from(0x3C00u16) ^ 0x8000; // fp16 1.0
        let key = (sortable << 16) | low;
        println!(
            "index {index:>6}: 0xFFFF - index = 0x{low:08x} -> key = 0x{key:08x} -> \
             greedy_token decodes {}",
            greedy_token_shipping(key)
        );
        if index < 65_536 {
            assert_eq!(key >> 16, sortable, "index {index}: value field intact");
            assert_eq!(
                key & 0xFFFF,
                0xFFFF - index,
                "index {index}: index field intact"
            );
            assert_eq!(
                greedy_token_shipping(key),
                index,
                "index {index} decodes back to itself"
            );
        } else {
            // The subtraction wraps to 0xFFFF_xxxx, so the OR clobbers the VALUE
            // field as well: every index past 65,535 keys as if its logit were
            // the largest representable one, and beats every real fp16 value
            // (+inf keys to 0xFC00_ffff).
            assert_eq!(
                key,
                0xFFFF_FFFFu32.wrapping_sub(index - 0x1_0000),
                "index {index}: key = 0xFFFFFFFF - (index - 65536)"
            );
            assert!(key > 0xFC00_FFFF, "index {index}: outranks even +inf");
        }
    }
    // Closed form, at every index of the real vocabulary.
    for index in 65_536u32..=262_143 {
        let sortable = u32::from(0x3C00u16) ^ 0x8000;
        let key = (sortable << 16) | (0xFFFFu32.wrapping_sub(index));
        assert_eq!(
            key,
            0xFFFF_FFFFu32.wrapping_sub(index - 0x1_0000),
            "index {index}"
        );
    }
    // Over the whole tail the key DECREASES with the index, so the maximum over
    // any row longer than 65,536 is uniquely index 65,536, and `greedy_token`
    // decodes 0xFFFFFFFF to 0. No tie, so it is deterministic, not racy.
    assert_eq!(greedy_token_shipping(0xFFFF_FFFF), 0);

    // A 65,536-entry vocabulary is still fine, which is why this sat unnoticed.
    let mut row = vec![f(-5.0); 65_536];
    row[40_000] = f(28.5);
    let mut best = 0u32;
    for (i, bits) in row.iter().enumerate() {
        best = best.max(greedy_key_shipping(*bits, i as u32));
    }
    assert_eq!(greedy_token_shipping(best), 40_000);
}

#[test]
fn shipping_kernel_returns_token_0_on_the_real_vocabulary() {
    let probe = Probe::new(Packing::Fixed);
    let mut row = vec![f(-5.0); N];
    row[9_259] = f(28.5);
    let buffer = probe.upload(&row);
    assert_eq!(argmax_logits(&row), 9_259, "the host agrees it is 9,259");
    assert_eq!(
        probe.token_shipping(&buffer, N),
        0,
        "the shipping device pick returns token 0"
    );
}

// ---------------------------------------------------------------------------
// 2. The replacement
// ---------------------------------------------------------------------------

fn check_packing(packing: Packing) -> (Vec<String>, usize, usize) {
    let mut probe = Probe::new(packing);
    let mut failures = Vec::new();
    let mut divergences: Vec<String> = Vec::new();
    let mut rows = 0usize;
    for case in cases() {
        rows += 1;
        let n = case.row.len();
        let host = argmax_logits(&case.row);
        let buffer = probe.upload(&case.row);
        // One shape for the exhaustive rows, all four for the interesting ones,
        // so the answer cannot depend on the partition.
        let slots = packing.slots(n) as u64;
        let shapes: &[(u64, u64)] = if case.every_grid {
            &[(1, 32), (3, 256), (17, 128), (128, 256)]
        } else {
            &[(128, 256)]
        };
        let mut got = Vec::new();
        for &(per_slot, tg) in shapes {
            probe.set_grid(per_slot * slots, tg);
            got.push(packing.token(&probe.keys(&buffer, n, packing, per_slot)));
        }
        probe.set_grid(GROUPS, TG);
        if got.iter().any(|g| *g != host) {
            if case.exact {
                failures.push(format!("{}: host {host}, device {got:?}", case.name));
            } else {
                divergences.push(format!("{}: host {host}, device {got:?}", case.name));
            }
        }
    }
    println!(
        "{rows} rows checked, 4 grid shapes each; {} known divergences, all \
         `logits[0]` NaN (e.g. {})",
        divergences.len(),
        divergences.first().map(String::as_str).unwrap_or("none")
    );
    (failures, rows, divergences.len())
}

#[test]
fn device_argmax_matches_host_argmax() {
    let (failures, rows, divergences) = check_packing(Packing::Fixed);
    assert!(
        failures.is_empty(),
        "device argmax disagreed with argmax_logits on {} of {rows} row(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    // The only permitted disagreement, and it is a row the model has already
    // destroyed: `logits[0]` is NaN, so the host's `best_val` is NaN and it
    // returns 0 without ever comparing anything.
    assert_eq!(
        divergences, 3,
        "only the three NaN-at-index-0 rows may differ"
    );
}

#[test]
fn the_equivalence_test_can_fail() {
    // AGENTS.md 3.4: a coverage claim is not a claim until the failing branch
    // has been seen to fail. The SAME check with the shipping whole-index field
    // must disagree.
    let (failures, rows, _) = check_packing(Packing::Shipping);
    assert!(
        !failures.is_empty(),
        "the 16-bit index field was supposed to break the equivalence; it did \
         not, so the test above cannot detect that bug and proves nothing"
    );
    println!(
        "{} of {rows} row(s) disagree with the host:",
        failures.len()
    );
    for line in failures.iter().take(8) {
        println!("  {line}");
    }
}

// ---------------------------------------------------------------------------
// 3. Cost
// ---------------------------------------------------------------------------

fn med(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
fn device_argmax_is_cheaper_than_the_host_loop() {
    let probe = Probe::new(Packing::Fixed);
    let mut s = 0x5ee5_5eed_1234_0001u64;
    let mut row = vec![0u16; N];
    for v in row.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((s >> 33) as f32) / ((1u64 << 31) as f32);
        *v = f(unit * 10.0 - 5.0);
    }
    row[9_259] = f(28.5);
    let buffer = probe.upload(&row);
    let mut dst = vec![0u16; N];

    // The shipping per-step sequence: model.rs:3132-3136 (the 512 KiB copy out
    // of the shared buffer) then generate.rs:867 (the argmax on the host).
    let mut host = || {
        unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), dst.as_mut_ptr(), N) };
        std::hint::black_box(argmax_logits(std::hint::black_box(&dst)))
    };
    let best: Buffer = probe.zeroed(16);
    let mut device = || {
        unsafe { *(best.contents() as *mut u32) = 0 };
        let mut params = (N as u32).to_le_bytes().to_vec();
        params.extend_from_slice(&((probe.groups / 4) as u32).to_le_bytes());
        let mut encoder = MetalEncoder::new(&probe.queue).expect("encoder");
        encoder
            .dispatch_groups(
                &probe.slots,
                &[Some(&buffer), Some(&best)],
                &params,
                probe.groups,
                probe.tg,
            )
            .expect("dispatch");
        encoder.commit_and_wait().expect("commit");
        let keys = unsafe { std::slice::from_raw_parts(best.contents() as *const u32, 4) };
        std::hint::black_box(Packing::Fixed.token(keys))
    };

    for _ in 0..20 {
        assert_eq!(host(), 9_259);
        assert_eq!(device(), 9_259);
    }
    let iters = 300;
    let (mut host_ms, mut dev_ms) = (Vec::new(), Vec::new());
    for _ in 0..iters {
        let t = Instant::now();
        host();
        host_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        device();
        dev_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let (h, d) = (med(host_ms), med(dev_ms));
    let own = med(probe.gpu_own_ms(&buffer, N, GROUPS / 4, 64));
    println!("host  (512 KiB copy + argmax over {N})  median {h:.4} ms");
    println!("device 1 dispatch + barrier              median {d:.4} ms");
    println!("device kernel own time (GPU clock)      median {own:.4} ms");
    println!(
        "saving                                  {:.4} ms/token",
        h - d
    );
    assert!(h > d, "the device form must be cheaper than the host loop");
}
