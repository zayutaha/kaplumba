//! Where the MRHS (multi-RHS GEMV) prefill kernel's bandwidth goes as a function
//! of the query count M, against the decode GEMV it was derived from.
//!
//! WHY THIS EXISTS (spec/small-batch-prefill-mrhs): end-to-end TTFT for a
//! 14-token turn dropped 1308 -> 392 ms (3.3x) but AC4 wants <= 200 ms. The
//! profile says every projection runs at ~16-19 GB/s of weight bytes while the
//! same tensors stream at ~185 GB/s in decode. This sweep finds which M bucket
//! loses it and whether the loss is per-row (compute/register bound) or fixed
//! (memory bound), on synthetic weights so no checkpoint is needed.
//!
//! Instrument rules copied from gemv_shape.rs because they are what makes a
//! number from this box usable:
//!   * median of interleaved rounds, never min-of-passes;
//!   * cold working set (1.6 GB rotated), so no pass is L2-resident;
//!   * every arm re-measured every round, so clock drift shows as a moving ratio;
//!   * boot clocks warmed before the first measured round.
//!
//! USAGE: cargo run --release --example mrhs_shape -- [--rounds N]

use half::f16;
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearKernels, PrefillEncodeRequest};
use metal::{Device, MTLResourceOptions};

const WORKING_SET_BYTES: u64 = 1_600_000_000;

/// gate_proj at the real checkpoint shape: 5376 -> 21504 at 3 bpw = 43.4 MB.
const IN_F: usize = 5376;
const OUT_F: usize = 21504;
const RATE: u8 = 3;

fn build(seed: u32) -> Vec<u16> {
    let tiles = (IN_F / 16) * (OUT_F / 16);
    let mask = (1u32 << RATE) - 1;
    let mut packed = Vec::with_capacity(tiles * 256 * RATE as usize / 16);
    let mut state = seed | 1;
    let mut states = vec![0u16; 256];
    for _ in 0..tiles {
        for slot in states.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *slot = ((state >> 16) as u16) & (mask as u16);
        }
        packed.extend(pack_integer_states(&states, RATE).unwrap().0);
    }
    packed
}

struct Case {
    device: Device,
    kernels: &'static LinearKernels,
    queue: metal::CommandQueue,
    packed: metal::Buffer,
    stride: u64,
    copies: u64,
    bytes: f64,
    input: metal::Buffer,
    input_signs: metal::Buffer,
    output_signs: metal::Buffer,
}

impl Case {
    fn build() -> Self {
        let device = Device::system_default().expect("Metal device");
        let kernels = LinearKernels::shared().expect("kernels");
        let queue = device.new_command_queue();
        let one = build(0x5eed_0000 ^ (IN_F as u32) ^ (OUT_F as u32));
        let bytes = (one.len() * 2) as u64;
        let copies = (WORKING_SET_BYTES / bytes).max(2);
        let mut big: Vec<u16> = Vec::with_capacity(one.len() * copies as usize);
        for copy in 0..copies {
            let tag = (copy as u16).wrapping_mul(0x9E37) ^ 0x5A5A;
            big.extend(one.iter().map(|value| value ^ tag));
        }
        let packed = device.new_buffer_with_data(
            big.as_ptr() as *const std::ffi::c_void,
            (big.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        drop(big);
        drop(one);
        let signs = |value: f32| {
            device.new_buffer_with_data(
                [f16::from_f32(value).to_bits(); 8].as_ptr() as *const std::ffi::c_void,
                16,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let input = signs(0.0031);
        let input_signs = signs(0.0031);
        let output_signs = signs(-0.0027);
        Self {
            device,
            kernels,
            queue,
            packed,
            stride: bytes,
            copies,
            bytes: bytes as f64,
            input,
            input_signs,
            output_signs,
        }
    }

    /// M rows of activations, distinct per row so a kernel that reads the wrong
    /// row cannot produce a correct-looking number.
    fn activations(&self, m: usize) -> metal::Buffer {
        let values: Vec<u16> = (0..m * IN_F)
            .map(|i| f16::from_f32(((i % 1024) as f32 / 1024.0) - 0.5).to_bits())
            .collect();
        self.device.new_buffer_with_data(
            values.as_ptr() as *const std::ffi::c_void,
            (values.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn run_mrhs(&self, m: usize, perm_mode: u32, passes: usize) -> f64 {
        let input = self.activations(m);
        let transformed = self.device.new_buffer(
            (m * IN_F * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let raw = self
            .device
            .new_buffer((m * OUT_F * 4) as u64, MTLResourceOptions::StorageModeShared);
        let output = self
            .device
            .new_buffer((m * OUT_F * 2) as u64, MTLResourceOptions::StorageModeShared);
        let mut encoder = MetalEncoder::new(&self.queue).unwrap();
        let started = std::time::Instant::now();
        for pass in 0..passes {
            let offset = (pass as u64 % self.copies) * self.stride;
            self.kernels
                .encode_gemv_mrhs(
                    &mut encoder,
                    PrefillEncodeRequest {
                        packed: &self.packed,
                        packed_offset: offset,
                        perm_mode,
                        input: &input,
                        input_signs: &self.input_signs,
                        output_signs: &self.output_signs,
                        transformed: &transformed,
                        raw: &raw,
                        output: &output,
                        queries: m,
                        in_features: IN_F,
                        out_features: OUT_F,
                        rate: Exl3Rate::Integer(RATE),
                        codebook: Exl3Codebook::Mcg,
                    },
                )
                .unwrap();
        }
        encoder.commit_and_wait().unwrap();
        started.elapsed().as_secs_f64() / passes as f64 * 1e3
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let rounds: usize = std::env::args()
        .position(|a| a == "--rounds")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let case = Case::build();
    let passes = ((300e6 / case.bytes).ceil() as usize).clamp(4, 64);

    // Arms: the decode kernel (M=1 reference) and each compiled MRHS bucket, at
    // both weight layouts. `decode` is encode_gemv with one row; the MRHS arms
    // include M=1 through the same dispatch the model uses.
    struct Arm {
        label: String,
        m: usize,
        perm_mode: u32,
        decode: bool,
    }
    let mut arms = vec![Arm {
        label: "decode gemv".to_string(),
        m: 1,
        perm_mode: 1,
        decode: true,
    }];
    for perm in [1u32, 0u32] {
        for m in [1usize, 4, 8, 16, 32] {
            let bucket = if m <= 4 {
                4
            } else if m <= 8 {
                8
            } else if m <= 16 {
                16
            } else {
                32
            };
            arms.push(Arm {
                label: format!("mrhs m{m} (m{bucket}) perm{perm}"),
                m,
                perm_mode: perm,
                decode: false,
            });
        }
    }

    // Boot clocks + pipeline/residency warm-up, once, before any measurement.
    for arm in &arms {
        let _ = case.run_mrhs(arm.m, arm.perm_mode, passes);
    }

    let mut samples = vec![Vec::with_capacity(rounds); arms.len()];
    for round in 0..rounds {
        for (index, arm) in arms.iter().enumerate() {
            samples[index].push(case.run_mrhs(arm.m, arm.perm_mode, passes));
        }
        eprintln!("  round {}/{}", round + 1, rounds);
    }

    println!(
        "\n  {}x{} @ {}bpw = {:.1} MB weights, {rounds} interleaved rounds of {passes} passes",
        IN_F,
        OUT_F,
        RATE,
        case.bytes / 1e6
    );
    println!(
        "    {:<26}{:>10}{:>10}{:>9}{:>10}",
        "arm", "med ms", "GB/s", "spread", "vs decode"
    );
    let base = {
        let mut control = samples[0].clone();
        median(&mut control)
    };
    for (index, arm) in arms.iter().enumerate() {
        let mut s = samples[index].clone();
        let mut sorted = s.clone();
        sorted.sort_by(f64::total_cmp);
        let m = median(&mut s);
        let gbps = case.bytes / 1e6 / 1e3 / (m / 1e3);
        println!(
            "    {:<26}{:>10.4}{:>10.1}{:>8.2}x{:>9.3}x",
            arm.label,
            m,
            gbps,
            sorted[sorted.len() - 1] / sorted[0].max(1e-9),
            base / m,
        );
    }
}
