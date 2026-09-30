//! PROBE HARNESS: load-issue vs decode-ALU partition for
//! `linear_gemv_ishizuki_k3_colmajor_contig4`.
//!
//! Compiles its own Metal library -- `decode_state*` preamble + the **unmodified**
//! `linear_repacked.metal` + `crates/kernels/probe/contig4_load_alu.metal` -- so
//! the production kernel is measured as arm (a) inside the same compiler run as
//! every probe arm. Nothing in `crates/kernels` is modified; the shipping library
//! does not contain these kernels and no production dispatch can reach them.
//!
//! Arms (see the .metal file for why each one is shaped the way it is):
//!   (a) linear_gemv_ishizuki_k3_colmajor_contig4   production, verbatim
//!   (r) probe_contig4_replica                       fidelity gate: must be
//!                                                   bitwise equal to (a)
//!   (b) probe_contig4_noload                        NO weight loads, full decode
//!   (c) probe_contig4_nodecode                      weight loads, no decode
//!   (d) probe_contig4_l1weight                      weight loads, 129 KB window
//!   (e) probe_contig4_alufloor                      NO loads, hoistable decode
//!
//! Usage: `probe_contig4_load_alu [--rounds N] [--iters N] [--bytes N]`

use half::f16;
use kaplumba_engine::exl3::{pack_integer_states, reorder_tensor_to_colmajor};
use kaplumba_kernels::exl3::KernelError;
use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::time::Instant;

/// The `decode_state*` family, transcribed verbatim from
/// `crates/kernels/src/linear.metal:115-147`. `linear_repacked.metal` needs
/// `decode_state_mcg`; `linear.metal` itself is 10.4k lines of mostly generated
/// code and is not worth compiling into a probe library.
const PREAMBLE: &str = r#"
// `linear_repacked.metal` has no header of its own: `LinearKernels::new` gets it
// by concatenating it onto `linear.metal`, which opens with these two lines. The
// probe library reproduces that, because the errors are not cosmetic -- without
// `using namespace metal` every `simd_shuffle_xor` / `mem_flags` in the file is
// an undeclared identifier.
#include <metal_stdlib>
using namespace metal;

static float decode_state_plain(uint value) {
    value = value * 89226354u + 64248484u;
    value = 0x3b603b60u ^ (value & 0x8fff8fffu);
    return float(as_type<half>((ushort)value) + as_type<half>((ushort)(value >> 16)));
}

static inline float decode_state_mcg(uint value) {
    value = value * 0xCBAC1FEDu;
    value = 0x3b603b60u ^ (value & 0x8fff8fffu);
    half2 h = as_type<half2>(value);
    return float(h.x + h.y);
}

static float decode_state_mul1(uint value) {
    value = value * 0x83DCD12Du;
    uint sum = 0x6400u + (value & 0xffu) + ((value >> 8) & 0xffu) + ((value >> 16) & 0xffu) + ((value >> 24) & 0xffu);
    half byte_value = as_type<half>((ushort)sum);
    half inverse = as_type<half>((ushort)0x1eeeu);
    half bias = as_type<half>((ushort)0xc931u);
    return float(half(float(byte_value) * float(inverse) + float(bias)));
}

static float decode_state(uint state, uint codebook) {
    if (codebook == 0u) { return decode_state_plain(state); }
    if (codebook == 1u) { return decode_state_mcg(state); }
    return decode_state_mul1(state);
}
"#;

const PRODUCTION: &str = include_str!("../../kernels/src/linear_repacked.metal");
const PROBE: &str = include_str!("../../kernels/probe/contig4_load_alu.metal");

const PRODUCTION_KERNEL: &str = "linear_gemv_ishizuki_k3_colmajor_contig4";

/// (name, label). The order is the order arms are timed in, interleaved, so it is
/// also the order they appear in every table this prints.
const ARMS: [(&str, &str); 8] = [
    (PRODUCTION_KERNEL, "(a) production"),
    ("probe_contig4_replica", "(r) replica (control)"),
    ("probe_contig4_noload", "(b) NO loads, full decode"),
    ("probe_contig4_alufloor", "(e) NO loads, hoisted decode"),
    ("probe_contig4_noshift", "(f) (b) + constant shift amt"),
    ("probe_contig4_shiftonly", "(g) (b) + NO decode body"),
    ("probe_contig4_nodecode", "(c) loads kept, NO decode"),
    ("probe_contig4_l1weight", "(d) loads kept, 129KB window"),
];

// --- host mirror of decode_state_mcg, for the CPU cross-check ---------------
fn host_decode_mcg(value: u32) -> f32 {
    let value = value.wrapping_mul(0xCBAC1FED);
    let value = 0x3b603b60 ^ (value & 0x8fff_8fff);
    let hx = f16::from_bits((value & 0xffff) as u16);
    let hy = f16::from_bits((value >> 16) as u16);
    (hx + hy).to_f32()
}

fn probe_word(seed: u32, tile: u32) -> u32 {
    seed ^ tile.wrapping_mul(0x9E37_79B9)
}

struct Ctx {
    device: Device,
    queue: metal::CommandQueue,
    pipelines: Vec<(String, ComputePipelineState)>,
}

fn build_library() -> Result<Ctx, KernelError> {
    let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
    let mut src = String::with_capacity(PREAMBLE.len() + PRODUCTION.len() + PROBE.len() + 2);
    src.push_str(PREAMBLE);
    src.push('\n');
    src.push_str(PRODUCTION);
    src.push('\n');
    src.push_str(PROBE);
    let library = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .map_err(KernelError::Shader)?;
    let queue = device.new_command_queue();
    let mut pipelines = Vec::new();
    for (name, _) in ARMS {
        let function = library
            .get_function(name, None)
            .map_err(KernelError::Shader)?;
        let state = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(KernelError::Pipeline)?;
        pipelines.push((name.to_string(), state));
    }
    Ok(Ctx {
        device,
        queue,
        pipelines,
    })
}

fn pipeline<'a>(ctx: &'a Ctx, name: &str) -> &'a ComputePipelineState {
    ctx.pipelines
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, s)| s)
        .unwrap_or_else(|| panic!("no pipeline {name}"))
}

/// Dispatch one GEMV. Mirrors `LinearKernels::encode_ishizuki_gemv`
/// (`linear.rs:1285`): grid = `tiles_n` threadgroups of 256, buffer(3) = shape.
fn dispatch(
    ctx: &Ctx,
    name: &str,
    weights: &Buffer,
    weight_offset: u64,
    x: &Buffer,
    y: &Buffer,
    probe: [u32; 2],
    in_f: usize,
    out_f: usize,
    enc: &ComputeCommandEncoderRef,
) {
    let groups = (out_f / 16) as u64;
    // { K, N, seed_a, seed_b }: production's first two entries, then the probe
    // seeds, so the params block is 16 bytes for every arm including (a).
    let params: [u32; 4] = [in_f as u32, out_f as u32, probe[0], probe[1]];
    let mut shape_bytes = [0u8; 16];
    for (i, value) in params.iter().enumerate() {
        shape_bytes[i * 4..(i + 1) * 4].copy_from_slice(&value.to_le_bytes());
    }
    enc.set_compute_pipeline_state(pipeline(ctx, name));
    enc.set_buffer(0, Some(weights), weight_offset);
    enc.set_buffer(1, Some(x), 0);
    enc.set_buffer(2, Some(y), 0);
    enc.set_bytes(3, 16, shape_bytes.as_ptr() as *const std::ffi::c_void);
    enc.dispatch_threads(MTLSize::new(groups * 256, 1, 1), MTLSize::new(256, 1, 1));
}

fn shared_buffer(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
}

fn build_packed(in_f: usize, out_f: usize, bits: u8, seed: u32) -> Vec<u16> {
    let tile_cols = out_f / 16;
    let tiles = (in_f / 16) * tile_cols;
    let mut packed = Vec::with_capacity(tiles * 16 * bits as usize);
    let mut s = seed;
    for _ in 0..tiles {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let states: Vec<u16> = (0..256)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s & ((1u32 << bits) - 1)) as u16
            })
            .collect();
        let (w, _) = pack_integer_states(&states, bits).unwrap();
        packed.extend(w);
    }
    packed
}

/// Fill `dst` with `copies` perturbed copies of `src`, exactly as
/// `bench_colmajor.rs:52-62` does, so the DRAM behaviour matches the 187.7 GB/s
/// reference measurement.
fn replicate_into(dst: &mut [u16], src: &[u16], copies: usize) {
    for c in 0..copies {
        let bump = (c as u16).wrapping_mul(0x9E37);
        let base = c * src.len();
        for (i, v) in src.iter().enumerate() {
            dst[base + i] = v ^ bump;
        }
    }
}

fn sin_input(in_f: usize) -> Vec<u16> {
    (0..in_f)
        .map(|i| f16::from_f32((i as f32 * 0.01).sin()).to_bits())
        .collect()
}

// ---------------------------------------------------------------------------
// CPU reference for arm (b) on a tiny shape: K=512 (tiles_k=32) x N=16.
// grid = 1 threadgroup of 256 threads => lane = lid%32, sg = lid/32, tn = 0.
// `step` = SG*4 = 32 and `k_unroll_end` = 32, so the unrolled loop runs exactly
// once per simdgroup at tk = sg*4 and the remainder loop never runs.
// ---------------------------------------------------------------------------
fn cpu_noload(x: &[f16], shape: (usize, usize), seeds: (u32, u32)) -> Vec<f32> {
    const KA: u32 = 3;
    const RING: u32 = 256 * 3;
    const SG: usize = 8;
    let (_k, n) = shape;
    let (sa, sb) = seeds;
    // per (lane, sg) accumulators, reduced exactly as the kernel reduces them
    let mut acc = vec![[0.0f32; 2]; SG * 32]; // [sg*32+lane][0]=acc0 [1]=acc1
    for sg in 0..SG {
        for lane in 0..32usize {
            let r0 = (lane % 4) * 2;
            let lo0 = ((lane * 8 + 1) as u32) * 3 + RING - 16;
            let sh0 = lo0 % 32;
            let mut a0 = 0.0f32;
            let mut a1 = 0.0f32;
            for q in 0..4u32 {
                let tile = (sg as u32) * 4 + q;
                let wa = probe_word(sa, tile);
                let wb = probe_word(sb, tile);
                let a = ((wa as u64) << 32 | wb as u64) << sh0;
                let mut v = [0.0f32; 8];
                for i in 0..8usize {
                    let rad = (i as u32) * KA + 16;
                    v[i] = host_decode_mcg(((a >> (64 - rad)) as u32) & 0xffff);
                }
                let base = (tile as usize) * 16 + r0;
                let p = [
                    x[base].to_f32(),
                    x[base + 1].to_f32(),
                    x[base + 8].to_f32(),
                    x[base + 9].to_f32(),
                ];
                a0 += p[0] * v[0] + p[1] * v[1] + p[2] * v[2] + p[3] * v[3];
                a1 += p[0] * v[4] + p[1] * v[5] + p[2] * v[6] + p[3] * v[7];
            }
            acc[sg * 32 + lane] = [a0, a1];
        }
    }
    // simd_shuffle_xor 1 then 2: lane l ends up holding sum over lanes (l & !3)..+3
    let mut red = vec![[0.0f32; 16]; SG];
    for sg in 0..SG {
        for lane in 0..32usize {
            if lane % 4 != 0 {
                continue;
            }
            let group = lane / 4;
            for c in 0..4usize {
                let o = acc[sg * 32 + group * 4 + c];
                red[sg][group] += o[0];
                red[sg][group + 8] += o[1];
            }
        }
    }
    let mut y = vec![0.0f32; n];
    for lid in 0..16usize {
        let mut sum = 0.0f32;
        for g in 0..SG {
            sum += red[g][lid];
        }
        y[lid] = sum;
    }
    y
}

fn run_once(
    ctx: &Ctx,
    name: &str,
    weights: &Buffer,
    weight_offset: u64,
    x: &Buffer,
    y: &Buffer,
    probe: [u32; 2],
    in_f: usize,
    out_f: usize,
) {
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    dispatch(
        ctx,
        name,
        weights,
        weight_offset,
        x,
        y,
        probe,
        in_f,
        out_f,
        enc,
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

fn read_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = arg(&args, "--rounds", 7);
    let iters: usize = arg(&args, "--iters", 20);
    let target_bytes: usize = arg(&args, "--bytes", 3_000_000_000);

    let ctx = build_library().expect("compile probe library");
    println!("probe library compiled: {} kernels", ctx.pipelines.len());

    // ---------------- validation 0: CPU cross-check on a tiny shape ----------
    //
    // Arm (b) is garbage by construction -- it never reads `packed` -- so it has
    // no reference to be checked against. What it *can* be checked against is the
    // CPU model of exactly the semantics it claims: every decode fed from
    // `probe_word(seed, tile)` for a loop-varying `tile`, accumulated in the
    // kernel's own order, reduced through the kernel's own two-stage shuffle. If
    // the compiler had hoisted the decode, this would disagree by O(1) rather than
    // O(1e-7), and the harness panics.
    {
        let (in_f, out_f) = (512usize, 16usize);
        let x_host: Vec<f16> = sin_input(in_f)
            .into_iter()
            .map(|b| f16::from_bits(b))
            .collect();
        let x_bits: Vec<u16> = x_host.iter().map(|v| v.to_bits()).collect();
        let xb = ctx.device.new_buffer_with_data(
            x_bits.as_ptr() as *const _,
            (x_bits.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let yb = shared_buffer(&ctx.device, out_f * 4);
        // A tiny weight buffer: arm (b) never reads it, but (a)/(r)/(c) do, and
        // for the CPU check only (b) matters.
        let wb = shared_buffer(&ctx.device, 4096);
        let seeds: [(u32, u32); 2] = [(0x1111_1111, 0x2222_2222), (0x9ABC_DEF0, 0x0FED_CBA9)];
        for (sa, sb) in seeds {
            run_once(
                &ctx,
                "probe_contig4_noload",
                &wb,
                0,
                &xb,
                &yb,
                [sa, sb],
                in_f,
                out_f,
            );
            let got = read_f32(&yb, out_f);
            let want = cpu_noload(&x_host, (in_f, out_f), (sa, sb));
            let finite = got.iter().all(|v| v.is_finite());
            let max_rel = got
                .iter()
                .zip(want.iter())
                .map(|(g, w)| {
                    let scale = w.abs().max(1e-6);
                    (g - w).abs() / scale
                })
                .fold(0.0f32, f32::max);
            // Informational: f32 accumulation order differs between the GPU and
            // this model, so exact equality is not expected. What matters is that
            // the values agree to fp32 rounding AND that they are seed-dependent --
            // see the DCE GATE, which is the part that can invalidate the probe.
            let exact = got == want;
            println!(
                "  cpu-check noload(seed={sa:#x},{sb:#x}): finite={finite} \
                 max_rel_err={max_rel:.3e} exact_f32_order={exact} \
                 got[0]={:.6} want[0]={:.6}",
                got[0], want[0]
            );
            if !finite {
                panic!("arm (b) produced non-finite output -- the decode was eliminated");
            }
            if max_rel > 1e-3 {
                panic!(
                    "arm (b) does not match the CPU model of `loads removed, ALU kept` \
                     (max_rel_err {max_rel:.3e}) -- the probe is measuring something else"
                );
            }
        }
    }

    // ---------------- benchmark -------------------------------------------
    for (name, in_f, out_f) in [
        ("gate_up  5376x43008", 5376usize, 43008usize),
        ("down     21504x5376", 21504, 5376),
    ] {
        let packed_orig = build_packed(in_f, out_f, 3, 7);
        let packed = reorder_tensor_to_colmajor(&packed_orig, in_f, out_f).expect("colmajor");
        drop(packed_orig);
        let weight_bytes = (packed.len() * 2) as f64;
        let copies = ((target_bytes / (packed.len() * 2)).max(2)) as usize;
        let total_words = packed.len() * copies;
        let wbuf = shared_buffer(&ctx.device, total_words * 2);
        {
            let dst =
                unsafe { std::slice::from_raw_parts_mut(wbuf.contents() as *mut u16, total_words) };
            replicate_into(dst, &packed, copies);
        }
        let stride = (packed.len() * 2) as u64;
        drop(packed);
        println!(
            "\n=== {name}: working set {:.2} GB ({copies} copies, {:.1} MB each), \
             weights {:.1} MB, grid {} threadgroups x 256",
            total_words as f64 * 2.0 / 1e9,
            stride as f64 / 1e6,
            weight_bytes / 1e6,
            out_f / 16,
        );

        let x_bits = sin_input(in_f);
        let xb = ctx.device.new_buffer_with_data(
            x_bits.as_ptr() as *const _,
            (x_bits.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let yb = shared_buffer(&ctx.device, out_f * 4);
        let seeds: [u32; 2] = [0x1111_1111, 0x2222_2222];
        let seeds2: [u32; 2] = [0x9ABC_DEF0, 0x0FED_CBA9];

        // -------- fidelity gate: (r) must be bitwise equal to (a) ----------
        run_once(
            &ctx,
            PRODUCTION_KERNEL,
            &wbuf,
            0,
            &xb,
            &yb,
            seeds,
            in_f,
            out_f,
        );
        let y_prod = read_f32(&yb, out_f);
        run_once(
            &ctx,
            "probe_contig4_replica",
            &wbuf,
            0,
            &xb,
            &yb,
            seeds,
            in_f,
            out_f,
        );
        let y_rep = read_f32(&yb, out_f);
        let replica_equal = y_prod == y_rep;
        let prod_finite = y_prod.iter().all(|v| v.is_finite());
        println!(
            "  FIDELITY: replica bitwise-equal to production = {replica_equal}, \
             production output finite = {prod_finite}"
        );
        if !replica_equal || !prod_finite {
            panic!("probe body is not a faithful copy of production -- every arm is void");
        }
        // The (r)-vs-(a) *timing* gap is reported, not asserted: the arms are
        // attributed against (r), so a replica that schedules differently from
        // production does not corrupt the partition. It does mean the absolute
        // (a) row is not a lower bound on the probe bodies' cost.

        // -------- DCE gate: perturbing the seeds must change (b)'s output --
        run_once(
            &ctx,
            "probe_contig4_noload",
            &wbuf,
            0,
            &xb,
            &yb,
            seeds,
            in_f,
            out_f,
        );
        let b_seed1 = read_f32(&yb, out_f);
        run_once(
            &ctx,
            "probe_contig4_noload",
            &wbuf,
            0,
            &xb,
            &yb,
            seeds2,
            in_f,
            out_f,
        );
        let b_seed2 = read_f32(&yb, out_f);
        let b_differs = b_seed1 != b_seed2;
        let b_differs_from_prod = b_seed1 != y_prod;
        let b_nonzero = b_seed1.iter().any(|v| *v != 0.0);
        println!(
            "  DCE GATE: noload(seed A) != noload(seed B) = {b_differs}, \
             != production = {b_differs_from_prod}, nonzero = {b_nonzero}"
        );
        if !b_differs || !b_nonzero {
            panic!("arm (b)'s decode was eliminated -- the probe is void");
        }

        // -------- warmup ---------------------------------------------------
        for (arm, _) in ARMS {
            for w in 0..2u64 {
                run_once(&ctx, arm, &wbuf, w * stride, &xb, &yb, seeds, in_f, out_f);
            }
        }

        // -------- interleaved A/B, median + spread ------------------------
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
        for round in 0..rounds {
            for (ai, (arm, _)) in ARMS.iter().enumerate() {
                let cb = ctx.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                for w in 0..iters as u64 {
                    let off = ((round as u64 * iters as u64 + w) % copies as u64) * stride;
                    dispatch(&ctx, arm, &wbuf, off, &xb, &yb, seeds, in_f, out_f, enc);
                }
                enc.end_encoding();
                let t = Instant::now();
                cb.commit();
                cb.wait_until_completed();
                let secs = t.elapsed().as_secs_f64() / iters as f64;
                samples[ai].push(secs * 1e3);
            }
        }

        // Estimator is the MINIMUM over rounds, not the median. Other agents share
        // this box and `gpuq` only excludes the *engine*'s flock, not a benchmark
        // harness like this one, so rounds get stolen. Contention can only add
        // time to a GPU kernel, never remove it, so min is the estimator that
        // estimates the uncontended kernel. The median and max are printed beside
        // it precisely so the reader can see how bad the interference was; when
        // max/median is ~1.0x the run was clean.
        println!(
            "\n  {:<28} {:>9} {:>9} {:>9} {:>9} {:>10} {:>8}",
            "arm", "min ms", "median ms", "max ms", "GB/s(min)", "med/min", "vs (a)"
        );
        let best = |i: usize| samples[i].iter().cloned().fold(f64::MAX, f64::min);
        let base = best(0);
        let mut out = String::new();
        for (ai, (_, label)) in ARMS.iter().enumerate() {
            let lo = best(ai);
            let m = median(&samples[ai]);
            let hi = samples[ai].iter().cloned().fold(0.0f64, f64::max);
            let gbps = weight_bytes / (lo * 1e-3) / 1e9;
            let rel = (lo - base) / base * 100.0;
            let mratio = m / lo;
            out.push_str(&format!(
                "  {:<28} {lo:>9.4} {m:>9.4} {hi:>9.4} {gbps:>10.1} {mratio:>9.2} {rel:>+7.1}%\n",
                label
            ));
        }
        print!("{out}");

        // -------- the partition ------------------------------------------
        // Attributed against (r), not (a): (r) is the byte-identical body, so any
        // difference between it and (b)/(c)/(d)/(e) is the probe delta and nothing
        // else. Using (a) would fold the replica's own scheduling difference into
        // every number.
        let t_a = best(0);
        let t_r = best(1);
        let t_b = best(2);
        let t_e = best(3);
        let t_f = best(4);
        let t_g = best(5);
        let t_c = best(6);
        let t_d = best(7);
        let pct = |v: f64| v / t_r * 100.0;
        println!("\n  --- partition, attributed against the replica T(r) = {t_r:.4} ms ---");
        println!(
            "  T(a) production                  {t_a:.4} ms  ({:+.1}% vs T(r))",
            pct(t_a - t_r)
        );
        println!(
            "  T(e) loads gone, decode hoisted  {t_e:.4} ms  ({:+.1}% vs T(r))",
            pct(t_e - t_r)
        );
        println!(
            "  T(b) loads gone, decode kept     {t_b:.4} ms  ({:+.1}% vs T(r))",
            pct(t_b - t_r)
        );
        println!(
            "  T(f) (b), constant shift amt     {t_f:.4} ms  ({:+.1}% vs T(r))",
            pct(t_f - t_r)
        );
        println!(
            "  T(g) (b), no decode body         {t_g:.4} ms  ({:+.1}% vs T(r))",
            pct(t_g - t_r)
        );
        println!(
            "  T(c) loads kept, no decode       {t_c:.4} ms  ({:+.1}% vs T(r))",
            pct(t_c - t_r)
        );
        println!(
            "  T(d) loads kept, 129KB window    {t_d:.4} ms  ({:+.1}% vs T(r))",
            pct(t_d - t_r)
        );
        println!(
            "\n  decode total = T(b) - T(e)       {:+.4} ms  ({:+.1}% of T(r))",
            t_b - t_e,
            pct(t_b - t_e)
        );
        println!(
            "    of which 64-bit barrel shift = T(b) - T(f)   {:+.4} ms  ({:+.1}%)",
            t_b - t_f,
            pct(t_b - t_f)
        );
        println!(
            "    of which decode_state_mcg    = T(f) - T(e)   {:+.4} ms  ({:+.1}%)",
            t_f - t_e,
            pct(t_f - t_e)
        );
        println!(
            "    cross-check (g): shift only   = T(b) - T(g)   {:+.4} ms  ({:+.1}%)",
            t_b - t_g,
            pct(t_b - t_g)
        );
        println!(
            "                 decode only     = T(g) - T(e)   {:+.4} ms  ({:+.1}%)",
            t_g - t_e,
            pct(t_g - t_e)
        );
        println!(
            "  load  total = T(r) - T(b)       {:+.4} ms  ({:+.1}% of T(r))",
            t_r - t_b,
            pct(t_r - t_b)
        );
        println!(
            "    cross-check (c): loads kept, decode gone = T(c) - T(r) {:+.4} ms ({:+.1}%)",
            t_c - t_r,
            pct(t_c - t_r)
        );
        println!(
            "    cross-check (d): DRAM/L2 miss only = T(r) - T(d) {:+.4} ms ({:+.1}%)",
            t_r - t_d,
            pct(t_r - t_d)
        );
        println!(
            "  residual = T(b) - T(e) - (T(r) - T(b)) = {:+.4} ms ({:+.1}%)",
            (t_b - t_e) - (t_r - t_b),
            pct((t_b - t_e) - (t_r - t_b))
        );
        println!(
            "  => partition of T(r): decode {:+.1}% / load {:+.1}% / neither {:+.1}%",
            pct(t_b - t_e),
            pct(t_r - t_b),
            pct(t_e)
        );
        let worst = samples
            .iter()
            .map(|s| median(s) / s.iter().cloned().fold(f64::MAX, f64::min))
            .fold(0.0f64, f64::max);
        println!(
            "\n  ESTIMATOR: min over {rounds} interleaved rounds. Worst median/min \
             ratio across arms = {worst:.2}x (>1 means another process stole the \
             GPU during the run; gpuq only excludes the engine's flock, not a \
             benchmark harness). Interleaved A B A B, so a stolen round inflates \
             every arm it touched rather than biasing one."
        );
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
