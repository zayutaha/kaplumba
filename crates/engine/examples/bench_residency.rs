//! Does the GEMV bandwidth survive contact with file-backed weights?
//!
//! `bench_gemv` allocates its own `StorageModeShared` Metal buffer, so the whole
//! working set is physically resident and contiguous. The engine does not do
//! that: `weights.rs:983` mmaps the safetensors shard `PROT_READ` /
//! `MAP_PRIVATE` and `weights.rs:1036` binds it with
//! `new_buffer_with_bytes_no_copy`. On a 24 GiB box that means the 11-12 GB of
//! weights live in the **file cache**, not in resident memory, and the kernel
//! reads through page faults.
//!
//! That difference is not academic. Measured on the same binary:
//!
//! ```text
//! peak memory footprint      12,156,560,792   (11.3 GiB)
//! peak_resident_pages_bytes   4,319,641,600   (4.0 GiB)
//! page faults                        236,743
//! decode_tok_s                            1.200   <- eviction collapse
//! ```
//!
//! A microbenchmark whose buffers are all resident measures a machine state the
//! engine never runs in. This tool runs the *same* kernels and shapes over
//! weights allocated three ways, so the residency effect is isolated:
//!
//! | arm | backing | what it models |
//! |---|---|---|
//! | `resident` | `new_buffer_with_data` | the previous benchmark: all resident |
//! | `file` | mmap + `new_buffer_with_bytes_no_copy` | **the engine's actual path** |
//! | `file_cold` | mmap, `MADV_DONTNEED` before each round | worst case: evict first |
//!
//! If the 8x-unroll kernel's advantage is real it should appear in all three. If
//! it only appears in `resident`, the kernel is faster at issuing loads but the
//! engine is stalled on page faults, and no amount of unrolling will help.
//!
//! Usage: `bench_residency [working_set_gb] [rounds] [shapes...]`
//!
//! Allocates ~2x working_set_gb per arm and writes a scratch file under
//! `$TMPDIR`. Do not run it while a model is resident.

use half::f16;
use kaplumba_engine::bench::{Layout, SHAPES};
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Buffer, Device, MTLResourceOptions};
use std::time::Instant;

const ROUND_TARGET_MS: f64 = 40.0;
const MIN_ITERS: u64 = 4;
const MAX_ITERS: u64 = 256;

/// A scratch file plus its mmap, mirroring `ShardReader::open` + `no_copy_view`.
struct FileBacked {
    _file: std::fs::File,
    path: std::path::PathBuf,
    mapping: *mut libc::c_void,
    length: usize,
}

impl FileBacked {
    /// Write `data` to a scratch file, then mmap it exactly as the engine does.
    fn create(path: &std::path::Path, data: &[u16]) -> std::io::Result<Self> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)
            .map_err(|e| std::io::Error::new(e.kind(), format!("create {}: {e}", path.display())))?;
        // u16 little-endian, matching the checkpoint's on-disk byte order.
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        file.write_all(&bytes)?;
        file.flush()?;
        drop(file);
        // Reopen read-only. Measured on this box: `mmap(PROT_READ, MAP_PRIVATE)`
        // fails with EACCES on a descriptor still open O_RDWR, and succeeds on a
        // read-only one. The engine does the same thing (its File is opened for
        // `read_at`, not for writing), which is why it mmaps at all.
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
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!(
                    "mmap {} len={length} PROT_READ/MAP_PRIVATE failed: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                ),
            ));
        }
        Ok(Self {
            _file: file,
            path: path.to_path_buf(),
            mapping,
            length,
        })
    }

    /// Drop the mapping's pages so the next touch faults them back in.
    ///
    /// This is what makes `file_cold` an honest worst case rather than a
    /// second copy of `file`. On Darwin `MADV_FREE` only marks pages, and
    /// `MADV_DONTNEED` reclaims them outright; the latter is what we want.
    fn evict(&self) {
        unsafe {
            libc::madvise(self.mapping, self.length, libc::MADV_DONTNEED);
        }
    }

    /// Ask the kernel to fault the whole mapping in ahead of time, i.e. the
    /// best case for the file-backed path.
    fn prefault(&self) {
        unsafe {
            libc::madvise(self.mapping, self.length, libc::MADV_WILLNEED);
            // Touch one byte per page so WILLNEED is not merely advisory.
            let base = self.mapping as *const u8;
            let mut acc = 0u64;
            for i in (0..self.length).step_by(4096) {
                acc = acc.wrapping_add(unsafe { *base.add(i) } as u64);
            }
            std::hint::black_box(acc);
        }
    }

    fn bind(&self, device: &Device) -> Buffer {
        // Same call the engine makes, so the buffer's storage mode and page
        // backing are identical.
        device.new_buffer_with_bytes_no_copy(
            self.mapping as *const std::ffi::c_void,
            self.length as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

impl Drop for FileBacked {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mapping, self.length);
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

struct Ctx {
    queue: metal::CommandQueue,
    input: Buffer,
    input_signs: Buffer,
    output_signs: Buffer,
    transformed: Buffer,
    raw: Buffer,
    output: Buffer,
    in_features: usize,
    out_features: usize,
    rate: Exl3Rate,
}

fn upload_exact(device: &Device, v: &[u16]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn ones(device: &Device, n: usize) -> Buffer {
    upload_exact(device, &vec![f16::from_f32(1.0).to_bits(); n])
}

fn build(shape_bits: u8, in_f: usize, out_f: usize, seed: u32) -> Vec<u16> {
    let tiles = (in_f / 16) * (out_f / 16);
    let mut packed = Vec::with_capacity(tiles * 16 * shape_bits as usize);
    let mut s = seed;
    for _ in 0..tiles {
        let mut states = [0u16; 256];
        for st in &mut states {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *st = (s & ((1u32 << shape_bits) - 1)) as u16;
        }
        let (w, _) = pack_integer_states(&states, shape_bits).expect("pack");
        packed.extend(w);
    }
    packed
}

/// One timed round, GB/s. `evict_first` runs before the timing window only.
fn round_gbps(
    kernels: &LinearKernels,
    ctx: &Ctx,
    kernel: &str,
    buf: &Buffer,
    iters: u64,
    evict_first: Option<&FileBacked>,
) -> f64 {
    let bytes = buf.length();
    if let Some(fb) = evict_first {
        fb.evict();
    }
    let mut encoder = MetalEncoder::new(&ctx.queue).expect("encoder");
    let started = Instant::now();
    for w in 0..iters {
        kernels
            .encode_gemv_named(
                &mut encoder,
                kernel,
                &LinearEncodeRequest {
                    packed: buf,
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
                    rate: ctx.rate,
                    codebook: Exl3Codebook::Mcg,
                    pretransformed: false,
                },
            )
            .expect("encode");
        // Cycle a byte offset so repeated dispatches do not all hit one page.
        let _ = w;
    }
    encoder.commit_and_wait().expect("wait");
    bytes as f64 / (started.elapsed().as_secs_f64() / iters as f64) / 1e9
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Bytes per gibibyte: 2^30, not 10^9.
///
/// Every "GiB" label in this file used to be produced by dividing by `1e9`,
/// which understates by **7.37 %** -- a 12.16 GB footprint printed as
/// "12.16 GiB" when it is 11.32 GiB. The tool's own doc comment above already
/// used the right divisor (`12,156,560,792 (11.3 GiB)` = /2^30) while its output
/// did not. See `gib` and the unit test below.
const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// Bytes to gibibytes. Only ever call this on a value that is truly bytes.
fn gib(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GIB
}

/// The peak *resident page* count for this process, in bytes. **Not a
/// footprint.**
///
/// The function used to be called `footprint_bytes` and its two call sites
/// printed it as `phys_footprint`, which is exactly the misreading
/// AGENTS.md 10.3 warns about: `getrusage().ru_maxrss` counts resident pages
/// only and misses the Metal buffer pages entirely. Measured on the same run at
/// 614 tokens, `ru_maxrss` read 6.23 GiB against a `phys_footprint` of
/// 11.69 GiB -- a 5.5 GiB gap -- so a run of this tool labelled its output
/// "phys_footprint" understated real memory usage by almost half.
///
/// It was renamed `peak_resident_pages_bytes` in `metrics.rs` (`ee14f62`)
/// precisely so it could not be misread again. AGENTS.md 10.3 also records that
/// a hand-rolled `task_vm_info` FFI has been attempted twice and rejected
/// twice: `libc` exports neither the struct nor the offset, and a guessed offset
/// yields a plausible wrong number, which is the worst failure mode for the
/// figure the memory budget is judged on. So this tool prints the page count,
/// labelled as such, and leaves the real footprint to
/// `/usr/bin/time -l` (`tests/memory_proof.rs`).
fn peak_resident_pages_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe {
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
    }
    // ru_maxrss on Darwin is bytes.
    usage.ru_maxrss as u64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let wanted: Vec<String> = args[2..].to_vec();

    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue().to_owned();
    let kernels = LinearKernels::new().expect("LinearKernels");
    // std::env::temp_dir() resolves to a per-user sandbox that is not writable
    // for mmap purposes here, so put the scratch file beside the model the
    // engine actually reads, where the page cache behaviour is the same.
    let scratch_dir: std::path::PathBuf = std::env::var("KAPLUMBA_SCRATCH_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            std::path::PathBuf::from(format!("{home}/.kaplumba/scratch"))
        });
    std::fs::create_dir_all(&scratch_dir).expect("create scratch dir");

    let shapes: Vec<&kaplumba_engine::bench::Shape> = SHAPES
        .iter()
        .filter(|s| wanted.is_empty() || wanted.iter().any(|w| s.name.starts_with(w)))
        .filter(|s| s.layout == Layout::Colmajor)
        .collect();

    println!("=== bench_residency: kernel bandwidth vs weight backing ===");
    println!(
        "{rounds} rounds per (shape, kernel, backing); \
         peak resident pages now {:.2} GiB (ru_maxrss, NOT phys_footprint)\n",
        gib(peak_resident_pages_bytes())
    );

    for (idx, shape) in shapes.iter().enumerate() {
        println!(
            "--- {}  {}x{} @{}bpw  ({:.1} MB, {} tile rows) ---",
            shape.name,
            shape.in_features,
            shape.out_features,
            shape.bits,
            shape.weight_bytes() as f64 / 1e6,
            shape.tile_rows()
        );

        let colmajor = kaplumba_engine::exl3::reorder_tensor_to_colmajor(
            &build(shape.bits, shape.in_features, shape.out_features, 7 + idx as u32),
            shape.in_features,
            shape.out_features,
        )
        .expect("colmajor");

        let ctx = Ctx {
            queue: queue.clone(),
            input: {
                let v: Vec<u16> = (0..shape.in_features)
                    .map(|i| f16::from_f32((i as f32 * 0.017).sin()).to_bits())
                    .collect();
                upload_exact(&device, &v)
            },
            input_signs: ones(&device, shape.in_features),
            output_signs: ones(&device, shape.out_features),
            transformed: device
                .new_buffer((shape.in_features * 2) as u64, MTLResourceOptions::StorageModeShared),
            raw: device.new_buffer(
                (shape.out_features * 4 * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            output: device
                .new_buffer((shape.out_features * 2) as u64, MTLResourceOptions::StorageModeShared),
            in_features: shape.in_features,
            out_features: shape.out_features,
            rate: Exl3Rate::Integer(shape.bits),
        };

        let path = scratch_dir.join(format!(
            "residency_{}_{}.bin",
            shape.name.replace(['/', ' '], "_"),
            std::process::id()
        ));
        let file_backed = FileBacked::create(&path, &colmajor).expect("scratch file");
        let resident = upload_exact(&device, &colmajor);
        let file_buf = file_backed.bind(&device);

        // Prefault the mapping so the "file warm" arm is not secretly cold.
        file_backed.prefault();

        let arm_names = ["resident", "file", "file_cold"];
        let mut results: Vec<(String, Vec<f64>)> = Vec::new();

        // `_contig4` is the kernel the engine actually dispatches, per
        // `KAPLUMBA_TRACE_GEMV=1`, so it is the one worth measuring. The `_u8`
        // variant this instrument originally compared against did not survive the
        // branch reset and measured 1.01-1.03x cold when it did -- see
        // implementation_notes.md; it is not worth carrying.
        for kernel in [
            "linear_gemv_ishizuki_k3_colmajor",
            "linear_gemv_ishizuki_k3_colmajor_contig4",
        ] {
            // Calibrate iters once, from the resident arm.
            let secs_per_iter = {
                let g = round_gbps(&kernels, &ctx, kernel, &resident, MIN_ITERS, None);
                (resident.length() as f64 / g / 1e9) as f64
            };
            let iters =
                (((ROUND_TARGET_MS * 1e-3) / secs_per_iter).round() as u64).clamp(MIN_ITERS, MAX_ITERS);
            let _ = round_gbps(&kernels, &ctx, kernel, &resident, MIN_ITERS, None); // warmup

            for (arm_i, arm) in arm_names.iter().copied().enumerate() {
                let (buf, evict) = match arm {
                    "resident" => (&resident, None),
                    "file" => (&file_buf, None),
                    _ => (&file_buf, Some(&file_backed)),
                };
                let mut samples = Vec::new();
                for _ in 0..rounds {
                    samples.push(round_gbps(&kernels, &ctx, kernel, buf, iters, evict));
                }
                let m = median(&samples);
                let best = samples.iter().cloned().fold(f64::MIN, f64::max);
                let worst = samples.iter().cloned().fold(f64::MAX, f64::min);
                println!(
                    "  {:<18} {:<8} median {:>7.1} GB/s  best {:>7.1}  worst {:>7.1}  spread {:.3}",
                    kernel.rsplit('_').next().unwrap_or(kernel),
                    arm,
                    m,
                    best,
                    worst,
                    worst / best
                );
                results.push((
                    format!("{} {arm}", kernel.rsplit('_').next().unwrap_or(kernel)),
                    samples,
                ));
                let _ = arm_i;
            }
        }

        // The question this tool exists to answer.
        let get = |k: &str| {
            results
                .iter()
                .find(|(n, _)| n.starts_with(k))
                .map(|(_, s)| median(s))
                .unwrap_or(0.0)
        };
        println!(
            "  => contig4/colmajor by backing:  resident {:.2}x   file {:.2}x   file_cold {:.2}x",
            get("contig4 resident") / get("colmajor resident"),
            get("contig4 file") / get("colmajor file"),
            get("contig4 file_cold") / get("colmajor file_cold"),
        );
        println!(
            "  => file/resident at 4x: {:.2}x\n",
            get("colmajor file") / get("colmajor resident")
        );
    }

    println!(
        "final peak resident pages {:.2} GiB (ru_maxrss; the real phys_footprint \
         needs /usr/bin/time -l)",
        gib(peak_resident_pages_bytes())
    );
}

/// The unit conversion, as a test, because the bug this replaced was a silent
/// one: `bytes / 1e9` printed into a format string labelled `GiB`.
///
/// Run with `cargo test --package kaplumba-engine --example bench_residency`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gibibyte_is_two_to_the_thirty_not_a_billion() {
        assert_eq!(BYTES_PER_GIB, 1_073_741_824.0);
        assert_eq!(gib(1_073_741_824), 1.0);
        assert_eq!(gib(1_000_000_000), 1_000_000_000.0 / 1_073_741_824.0);
        assert!(
            gib(1_000_000_000) < 1.0,
            "a decimal gigabyte is not a gibibyte"
        );
    }

    /// The tool's own documented measurement, from the doc comment at the top:
    /// 12,156,560,792 B is 11.3 GiB, not 12.2. With a `1e9` divisor this
    /// asserted value would print as "12.16 GiB" -- a 7.37 % overstatement of
    /// GiB, i.e. an understatement of the memory actually used.
    #[test]
    fn the_documented_footprint_converts_to_the_documented_gib() {
        assert!((gib(12_156_560_792) - 11.3).abs() < 0.05, "{}", gib(12_156_560_792));
        // And the 10.2 % gap that dividing by 1e9 introduces.
        let understated = 12_156_560_792.0 / 1e9;
        assert!(
            understated - gib(12_156_560_792) > 0.8,
            "expected ~0.84 GiB of divergence between 1e9 and 2^30, got {understated} vs {}",
            gib(12_156_560_792),
        );
    }
}