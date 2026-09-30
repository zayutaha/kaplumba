//! T14.1 — does halving the load amplification actually move the bandwidth?
//!
//! The 3 bpw layer GEMV sustains **187.7 GB/s** while the 6 bpw `lm_head`
//! sustains **250.5 GB/s**, despite moving identical useful bytes. The fitted
//! explanation is request/byte amplification: a 3 bpw tile is 24 `u32` words and
//! the live kernel issues 64 lane-loads for them (2.667x), where 6 bpw is 48
//! words and 96 loads (2.000x). Hence
//!
//! ```text
//! predicted 3 bpw = 250.5 x 2.000/2.667 = 187.9 GB/s
//! measured                            = 187.7 GB/s        (0.1% fit)
//! ```
//!
//! That fit has **two points and both are already measured**, so it cannot
//! predict a third regime. `D = 32` (8 lanes per tile, 4 words each) would put
//! amplification at 1.333 and, if the law holds, land on the DRAM ceiling.
//! This tool is the cheapest way to find out whether it does, *before* writing
//! the real kernel — which is a 4-tiles-per-simdgroup restructure, not a 60-line
//! edit.
//!
//! ## What is measured
//!
//! Three arms, interleaved round by round, cold (`MADV_DONTNEED` before every
//! round, the arm `bench_residency` established as the trusted one):
//!
//! | arm | kernel | requests/tile | bytes/tile | byte amp |
//! |---|---|---:|---:|---:|
//! | reference | `..._colmajor_contig4` | 64 | 256 | 2.667 |
//! | vector | `..._d32_probe` | 22 | 128 | 1.333 |
//! | scalar | `..._d32_probe_sc4` | 32 | 128 | 1.333 |
//!
//! **The vector and scalar arms fetch identical bytes and differ only in
//! request count.** That is deliberate and it is the actual experiment. If the
//! ~187.7 GB/s plateau is a *bytes* wall the two arms measure the same; if it is
//! a *requests* wall the scalar arm is 32/22 slower. The existing data cannot
//! tell these apart, because at 3 bpw and 6 bpw the two coincide exactly
//! (64 loads x 4 B = 2.667 x 96, and 96 x 4 = 2.000 x 192).
//!
//! ## The probe kernels are wrong on purpose
//!
//! `..._d32_probe*` write deliberately meaningless output — they measure the
//! load path, not the arithmetic. See the header in `linear_repacked.metal`. A
//! probe that returns plausible numbers is a probe that has been wired into the
//! decode path by accident, which is why `LinearKernels::is_probe_name` keeps
//! them out of every rate's dispatch slot.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --example probe_d32 -- [rounds] [shape...]
//! ```
//!
//! Only the 3 bpw colmajor shapes are eligible (the probes decode 3 bpw only).
//! Runs against a scratch mmap, not the model, so it must not overlap a model
//! load — check `scripts/gpuq status` first.

use half::f16;
use kaplumba_engine::bench::{summarize, Layout, SHAPES};
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{tile_amplification, LinearEncodeRequest, LinearKernels};
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use std::time::Instant;

const REFERENCE: &str = "linear_gemv_ishizuki_k3_colmajor_contig4";
const VECTOR: &str = "linear_gemv_ishizuki_k3_colmajor_d32_probe";
const SCALAR: &str = "linear_gemv_ishizuki_k3_colmajor_d32_probe_sc4";

const ROUND_TARGET_MS: f64 = 40.0;
const MIN_ITERS: u64 = 4;
const MAX_ITERS: u64 = 256;

/// AC7's gate. The law predicts 250.5 GB/s for `D = 32` (DRAM-capped), i.e.
/// 250.5/187.7 = 1.334x the reference. Demanding only 1.15x keeps the work worth
/// doing if the extrapolation is partly right, and rejects it if the plateau is
/// caused by something the amplification law does not describe.
const GO_NO_GO_RATIO: f64 = 1.15;

struct FileBacked {
    _file: std::fs::File,
    path: std::path::PathBuf,
    mapping: *mut libc::c_void,
    length: usize,
}

impl FileBacked {
    fn create(path: &std::path::Path, data: &[u16]) -> std::io::Result<Self> {
        use std::io::Write;
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        {
            let mut file = std::fs::File::create(path)?;
            file.write_all(&bytes)?;
            file.flush()?;
        }
        // Read-only descriptor: `mmap(PROT_READ, MAP_PRIVATE)` fails with EACCES
        // on one still open O_RDWR on this box. The engine does the same
        // (`ShardReader::open` opens for `read_at`).
        let file = std::fs::File::open(path)?;
        let length = data.len() * 2;
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                std::os::unix::io::AsRawFd::as_raw_fd(&file),
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            _file: file,
            path: path.to_path_buf(),
            mapping,
            length,
        })
    }

    /// Drop every page of the mapping, so the next round reads from DRAM.
    fn evict(&self) {
        unsafe { libc::madvise(self.mapping, self.length, libc::MADV_DONTNEED) };
    }
}

impl Drop for FileBacked {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.mapping, self.length) };
        let _ = std::fs::remove_file(&self.path);
    }
}

struct Ctx {
    queue: CommandQueue,
    input: Buffer,
    input_signs: Buffer,
    output_signs: Buffer,
    transformed: Buffer,
    raw: Buffer,
    output: Buffer,
    in_features: usize,
    out_features: usize,
}

fn ones(device: &Device, n: usize) -> Buffer {
    let bits = f16::from_f32(1.0).to_bits();
    device.new_buffer_with_data(
        vec![bits; n].as_ptr() as *const std::ffi::c_void,
        (n * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn build(bits: u8, in_f: usize, out_f: usize, seed: u32) -> Vec<u16> {
    let tiles = (in_f / 16) * (out_f / 16);
    let mut packed = Vec::with_capacity(tiles * 16 * bits as usize);
    let mut s = seed;
    for _ in 0..tiles {
        let mut states = [0u16; 256];
        for st in &mut states {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *st = (s & ((1u32 << bits) - 1)) as u16;
        }
        let (w, _) = pack_integer_states(&states, bits).expect("pack");
        packed.extend(w);
    }
    packed
}

/// Bytes the *weights* are, which is what achieved bandwidth is computed
/// against. Identical for all three arms, so the ratio between them is a ratio
/// of efficiency and nothing else.
fn weight_bytes(in_f: usize, out_f: usize, bits: u8) -> u64 {
    let tiles = (in_f / 16) * (out_f / 16);
    (tiles * 16 * bits as usize * 2) as u64
}

fn dispatch(
    kernels: &LinearKernels,
    ctx: &Ctx,
    kernel: &str,
    packed: &Buffer,
    iters: u64,
    fb: &FileBacked,
) -> f64 {
    let bytes = weight_bytes(ctx.in_features, ctx.out_features, 3);
    fb.evict();
    let mut encoder = MetalEncoder::new(&ctx.queue).expect("encoder");
    let started = Instant::now();
    for _ in 0..iters {
        let request = LinearEncodeRequest {
            packed,
            packed_offset: 0,
            perm_mode: 1,
            input: &ctx.input,
            input_signs: &ctx.input_signs,
            output_signs: &ctx.output_signs,
            transformed: &ctx.transformed,
            raw: &ctx.raw,
            output: &ctx.output,
            in_features: ctx.in_features,
            out_features: ctx.out_features,
            rate: Exl3Rate::Integer(3),
            codebook: Exl3Codebook::Mcg,
            pretransformed: false,
        };
        if LinearKernels::is_probe_name(kernel) {
            kernels
                .encode_d32_probe(&mut encoder, kernel, &request)
                .expect("probe dispatch");
        } else {
            kernels
                .encode_gemv_named(&mut encoder, kernel, &request)
                .expect("reference dispatch");
        }
    }
    encoder.commit_and_wait().expect("wait");
    bytes as f64 / (started.elapsed().as_secs_f64() / iters as f64) / 1e9
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rounds: usize = args.first().and_then(|a| a.parse().ok()).unwrap_or(7);

    // Only 3 bpw colmajor shapes: the probes decode 3 bpw only, and a row-major
    // shape would change the tile layout out from under the comparison.
    let shapes: Vec<_> = SHAPES
        .iter()
        .filter(|s| s.bits == 3 && s.layout == Layout::Colmajor)
        .collect();
    let selected: Vec<_> = if args.len() > 1 {
        shapes
            .iter()
            .filter(|s| args[1..].iter().any(|a| a == s.name))
            .copied()
            .collect()
    } else {
        shapes.clone()
    };
    if selected.is_empty() {
        eprintln!("no matching 3 bpw colmajor shapes; names available:");
        for s in &shapes {
            eprintln!("  {}", s.name);
        }
        std::process::exit(2);
    }

    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device");
        std::process::exit(1);
    };
    let Ok(kernels) = LinearKernels::new() else {
        eprintln!("LinearKernels unavailable");
        std::process::exit(1);
    };
    let queue = device.new_command_queue().to_owned();

    // The model, computed rather than quoted, so a reader can check the
    // arithmetic the whole decision rests on.
    let live = tile_amplification(3, 8);
    let d32 = tile_amplification(3, 32);
    println!("model (derived in linear.rs::tile_amplification, unit-tested):");
    println!(
        "  D= 8 live : {} lanes x {} words = {:2} requests, {:3} B requested / {:3} B useful = {:.3}x",
        live.lanes,
        live.words_per_lane,
        live.scalar_lane_loads,
        live.requested_bytes,
        live.useful_bytes,
        live.byte_amplification(),
    );
    println!(
        "  D=32 probe: {} lanes x {} words = {:2} requests (vec) / {:2} (scalar), {:3} B requested / {:3} B useful = {:.3}x",
        d32.lanes,
        d32.words_per_lane,
        d32.ideal_lane_loads,
        d32.scalar_lane_loads,
        d32.requested_bytes,
        d32.useful_bytes,
        d32.byte_amplification(),
    );
    let ceiling = 250.5_f64;
    println!(
        "  rate law  : {:.3} x {:.3}/{:.3} = {:.1} GB/s predicted for D=32 (DRAM-capped at {ceiling:.1})",
        ceiling,
        tile_amplification(6, 8).byte_amplification(),
        d32.byte_amplification(),
        ceiling * tile_amplification(6, 8).byte_amplification() / d32.byte_amplification(),
    );
    println!(
        "  AC7 gate  : D=32 vector arm must reach {GO_NO_GO_RATIO:.2}x the reference median\n",
    );

    let mut go = true;
    for (index, shape) in selected.iter().enumerate() {
        let (in_f, out_f) = (shape.in_features, shape.out_features);
        println!(
            "=== {} {}x{} ({} MB) ===",
            shape.name,
            in_f,
            out_f,
            weight_bytes(in_f, out_f, 3) / 1_000_000,
        );
        let packed = build(3, in_f, out_f, 0x5eed_0000 + index as u32);
        // Shape names are `sliding/gate`, so they cannot go into a filename
        // as-is: the slash makes a path whose parent directory does not exist.
        let flat: String = shape
            .name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let path = std::env::temp_dir().join(format!("probe_d32_{flat}.bin"));
        let fb = FileBacked::create(&path, &packed).expect("scratch mmap");
        let view = device.new_buffer_with_bytes_no_copy(
            fb.mapping as *const std::ffi::c_void,
            fb.length as u64,
            // `newBufferWithBytes:...` rejects `StorageModePrivate` outright --
            // the engine binds its mmap'd weights as Shared (weights.rs:1039)
            // and so does bench_residency.
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let ctx = Ctx {
            queue: queue.clone(),
            input: ones(&device, in_f),
            input_signs: ones(&device, in_f / 8),
            output_signs: ones(&device, out_f / 8),
            transformed: ones(&device, in_f),
            raw: ones(&device, out_f),
            output: ones(&device, out_f * 2),
            in_features: in_f,
            out_features: out_f,
        };

        // Calibrate iteration count off the reference so all three arms time the
        // same amount of work.
        let iters = (0..)
            .find(|_| false)
            .unwrap_or_else(|| {
                let mut n = MIN_ITERS;
                loop {
                    let mut enc = MetalEncoder::new(&ctx.queue).expect("encoder");
                    kernels
                        .encode_gemv_named(
                            &mut enc,
                            REFERENCE,
                            &LinearEncodeRequest {
                                packed: &view,
                                packed_offset: 0,
                                perm_mode: 1,
                                input: &ctx.input,
                                input_signs: &ctx.input_signs,
                                output_signs: &ctx.output_signs,
                                transformed: &ctx.transformed,
                                raw: &ctx.raw,
                                output: &ctx.output,
                                in_features: in_f,
                                out_features: out_f,
                                rate: Exl3Rate::Integer(3),
                                codebook: Exl3Codebook::Mcg,
                                pretransformed: false,
                            },
                        )
                        .expect("calibrate");
                    let started = Instant::now();
                    enc.commit_and_wait().expect("wait");
                    if started.elapsed().as_secs_f64() * 1000.0 * n as f64 >= ROUND_TARGET_MS
                        || n >= MAX_ITERS
                    {
                        return n;
                    }
                    n += 1;
                }
            });

        // Smoke check: one dispatch of each probe arm, before committing to
        // `iters * rounds` of them. A kernel with a register-pressure problem
        // spills to local memory and can be 100x slower than the reference,
        // which looks identical to a hang from the outside. Found that way once
        // already (parked in `commitAndWait` for minutes), so the check is
        // structural rather than a matter of patience.
        let ref_one = dispatch(&kernels, &ctx, REFERENCE, &view, 1, &fb);
        let vec_one = dispatch(&kernels, &ctx, VECTOR, &view, 1, &fb);
        let scl_one = dispatch(&kernels, &ctx, SCALAR, &view, 1, &fb);
        println!(
            "  single-dispatch smoke: reference {ref_one:.1}  vector {vec_one:.1}  scalar {scl_one:.1} GB/s"
        );
        if vec_one < ref_one / 8.0 || scl_one < ref_one / 8.0 {
            println!(
                "  ABORT: a probe arm is >8x slower than the reference on a single dispatch.\n  \
                 That is not a bandwidth result, it is the kernel being pathological\n  \
                 (register spill, or the grid is wrong). Fix it before timing."
            );
            std::process::exit(3);
        }

        let mut reference = Vec::new();
        let mut vector = Vec::new();
        let mut scalar = Vec::new();
        for _ in 0..rounds {
            // Interleaved A/B/C so drift hits all three arms equally.
            reference.push(dispatch(&kernels, &ctx, REFERENCE, &view, iters, &fb));
            vector.push(dispatch(&kernels, &ctx, VECTOR, &view, iters, &fb));
            scalar.push(dispatch(&kernels, &ctx, SCALAR, &view, iters, &fb));
        }

        let r = summarize(&reference).expect("reference samples");
        let v = summarize(&vector).expect("vector samples");
        let s = summarize(&scalar).expect("scalar samples");
        println!(
            "  iters/round {iters}   rounds {rounds}   (GB/s, useful weight bytes)\n  \
             {:<26} {:>7.1}  [{:.1} .. {:.1}]  spread {:.3}\n  \
             {:<26} {:>7.1}  [{:.1} .. {:.1}]  spread {:.3}   ratio {:.3}x\n  \
             {:<26} {:>7.1}  [{:.1} .. {:.1}]  spread {:.3}   ratio {:.3}x",
            "reference (contig4, D=8)",
            r.median,
            r.worst,
            r.best,
            r.spread,
            "probe vector (D=32, 22 req)",
            v.median,
            v.worst,
            v.best,
            v.spread,
            v.median / r.median,
            "probe scalar (D=32, 32 req)",
            s.median,
            s.worst,
            s.best,
            s.spread,
            s.median / r.median,
        );
        let ratio = v.median / r.median;
        let repeatable = r.is_repeatable(0.10) && v.is_repeatable(0.10);
        println!(
            "  -> {:.3}x   {}   requests-vs-bytes: {}",
            ratio,
            if ratio >= GO_NO_GO_RATIO {
                "PASS"
            } else {
                "FAIL (AC7: cancel T14.2)"
            },
            if (v.median - s.median).abs() / v.median > 0.05 {
                format!("REQUESTS matter (scalar {:.3}x of vector)", s.median / v.median)
            } else {
                format!("BYTES matter (scalar {:.3}x of vector)", s.median / v.median)
            },
        );
        if !repeatable {
            println!("  -> WARNING: spread > 10%; this run cannot resolve a 1.15x effect");
        }
        if ratio < GO_NO_GO_RATIO || !repeatable {
            go = false;
        }
        println!();
    }

    println!(
        "AC7: {}",
        if go {
            "PASS -- the extrapolation holds; T14.2 may proceed"
        } else {
            "FAIL -- the amplification law does not extrapolate to D=32. \
             Record it as a two-point coincidence and move to T15."
        }
    );
    std::process::exit(if go { 0 } else { 1 });
}
