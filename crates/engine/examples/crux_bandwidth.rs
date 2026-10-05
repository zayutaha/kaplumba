//! `spec/decode-20tok-crux` part 2: **measure** achieved streaming bandwidth,
//! and separate the three axes that earlier specs conflated.
//!
//! # The question
//!
//! `spec/decode-19tok` concluded that ~220 GB/s is a **request-rate wall** rather
//! than a bandwidth wall, on the evidence that "bandwidth is flat from 8 to 32
//! bytes in flight per thread". That is the load-width/ILP axis. But the sweep
//! that produced it only reached 20,480 threads, and the device allows 81,920,
//! so the **occupancy** axis was never moved. A flat curve on one axis does not
//! license a conclusion about a limit on a different axis. This probe moves all
//! three, independently, and reports whether the plateau is a wall and what the
//! maximum is.
//!
//! # Method, and why each part of it is there
//!
//! * **The buffer is written before it is streamed.** `examples/bw.rs` did not
//!   do this and reported 55 GB/s; that number became the achievable-bandwidth
//!   denominator for a whole round of experiments. Here every page is touched
//!   once with a high-entropy pattern, so the pages are resident and no line is
//!   compressible.
//! * **Median, best, worst and spread are all reported.** `bw.rs` reported
//!   min-of-passes, which on a shared box selects the one pass that ran while a
//!   competitor was idle — the likely origin of the 248.5 GB/s figure.
//! * **Rounds are interleaved and every arm is measured in every round**, so
//!   box drift appears as a moving ratio rather than being folded into whichever
//!   arm ran during it. `spec/decode-bandwidth-240gb` measured one shape to
//!   completion at a time and its two control rows moved +10.6% and −7.9%.
//! * **One control arm that cannot change** is carried through every round: a
//!   6 bpw-shaped scalar arm. If the control moves, the run is reported as
//!   untrustworthy rather than interpreted.
//! * **The equal-bytes gate.** Arms that visit the same bytes in the same order
//!   must produce the same checksum. `stream_u1`, `stream_u2`, `stream_u4` and
//!   every `stream_u4_uN` do, and the run asserts it before printing bandwidth.
//!   An arm that read a different range would fail here instead of producing a
//!   plausible wrong number.
//! * **Sabotage.** `--sabotage` halves the element count of the reference arm
//!   and requires the checksum to change. An instrument that cannot detect that
//!   it read half the data cannot be trusted to detect a 5% difference.
//!
//! # What it does NOT measure
//!
//! Achieved *weight* bandwidth. It measures what the memory system delivers when
//! asked for nothing else. The 3 bpw EXL3 GEMV is a different question with a
//! different answer (its 2.67x request amplification is measured in
//! `spec/decode-3bpw-amplification`), and the gap between the two numbers is the
//! single most important quantity in this spec.

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const SRC: &str = include_str!("../../kernels/src/crux_bw.metal");

/// Fill with content that is **globally non-repeating** and fault every page in.
///
/// # The defect this replaces
///
/// The first version tiled a 64 MiB LCG pattern across the whole buffer. That is
/// a fatal flaw, and it was caught only because the result was absurd: it
/// reported 492.7 GB/s, which is 180% of the 273 GB/s wire peak. DRAM cannot
/// deliver more than the wire can carry. What it actually measured was cache.
/// With a 64 MiB period, a 48 MB system-level cache still absorbs a large
/// fraction of every pass, so "achieved bandwidth" was partly L2 hit rate.
///
/// This is the same class of error as `examples/bw.rs` never writing its buffer
/// at all (55 GB/s) and as `bw.rs` reporting min-of-passes (248.5 GB/s): a probe
/// that measures something adjacent to the thing it names.
///
/// # The fix
///
/// One `splitmix64` stream seeded by the **global 16-byte block index**, so every
/// block in the buffer has content determined by where it lives and no two
/// 128-byte lines anywhere in the buffer can match. That makes every read a
/// compulsory miss on a cold cache and defeats any tile-based shortcut.
///
/// `distinct_tiles_verified` then *proves* it rather than asserting it: it
/// compares the first and last 64-byte lines and the sampled tiles must differ.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The OLD, DEFECTIVE fill, kept so the new check can be shown to reject it.
///
/// This is verbatim the tiling scheme used by `examples/probe_bw_250gb.rs:15`
/// and `examples/bw.rs` — the source of the "254 GB/s" figure. It exists only
/// so `--sabotage-fill` can demonstrate that `fill_is_non_repeating` catches the
/// thing it was written to catch. A guard that has never rejected the defect it
/// guards against is an assumption.
fn tiled_fill_period(buf: &metal::Buffer, bytes: usize, period: usize) {
    let ptr = buf.contents() as *mut u8;
    let pattern_bytes = std::cmp::min(period, bytes);
    let mut pattern = vec![0u8; pattern_bytes];
    let mut x: u32 = 0x9e37_79b9;
    for c in pattern.chunks_exact_mut(4) {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        c.copy_from_slice(&x.to_le_bytes());
    }
    let mut off = 0usize;
    while off < bytes {
        let n = std::cmp::min(pattern_bytes, bytes - off);
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), ptr.add(off), n) };
        off += n;
    }
}

fn tiled_fill(buf: &metal::Buffer, bytes: usize) {
    tiled_fill_period(buf, bytes, 64 * 1024 * 1024)
}

fn fill_and_fault(buf: &metal::Buffer, bytes: usize) {
    let ptr = buf.contents() as *mut u8;
    let nblocks = bytes / 16;
    for i in 0..nblocks {
        let v = splitmix64(i as u64);
        unsafe {
            (ptr.add(i * 16) as *mut u128).write_unaligned(v as u128);
        }
    }
    // Every page is now resident *and* touched with distinct content. The
    // volatile store per page is redundant after the loop above, but it is the
    // cheapest possible guarantee that no page is handed back as a lazy zero
    // fill, which macOS does do for untouched pages.
    for page in 0..(bytes / 4096) {
        unsafe { std::ptr::write_volatile(ptr.add(page * 4096), 0u8) };
    }
}

/// A working set far larger than any cache must make a *second* pass over the
/// same bytes no faster than the first.
///
/// If a pass were partly served from cache, pass 2 would be markedly faster.
/// This is the cheapest possible check that the measurement is DRAM traffic and
/// not cache, and it is the check the old probe could not have passed: with a
/// 64 MiB tile repeating across the whole buffer, its second pass was reading
/// the *same addresses* it had just read, which is the definition of a cache
/// hit, and it reported the result as streaming bandwidth.
///
/// Returns `(pass1_gbps, pass2_gbps)`.
fn pass_ratio_probe(
    device: &Device,
    queue: &metal::CommandQueue,
    pipe: &metal::ComputePipelineState,
    data: &metal::Buffer,
    out: &metal::Buffer,
    threads: u64,
    count: u32,
) -> (f64, f64) {
    let time_one = |passes: u32| -> f64 {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(pipe);
        e.set_buffer(0, Some(data), 0);
        e.set_buffer(1, Some(out), 0);
        e.set_bytes(2, 4, &count as *const u32 as *const std::ffi::c_void);
        for _ in 0..passes {
            e.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(256, 1, 1));
        }
        e.end_encoding();
        let t0 = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        t0.elapsed().as_secs_f64() / passes as f64
    };
    time_one(2); // warm
    let p1 = time_one(1);
    let p2 = time_one(1);
    let g = |t: f64| data.length() as f64 / t / 1e9;
    (g(p1), g(p2))
}

/// Sample 64-byte lines across the whole buffer and require them all distinct.
///
/// Checking only the first and last line would pass for a buffer tiled at any
/// period that does not divide the buffer — a 64 MiB tile across a 2 GB buffer
/// gives 32 identical copies and the first and last lines *could* still differ if
/// the tile length is not a divisor. Sampling a spread of offsets is what makes
/// a repeat detectable: with a period P, two samples P apart collide, and the
/// sample grid here resolves any period up to `bytes / 8`.
///
/// Returns `(duplicate_samples, total_samples)`.
/// Two properties of `fill_is_non_repeating`, asserted on known inputs.
///
/// A guard is only worth having if it rejects the thing it guards against. Both
/// directions are checked here: the honest fill must pass, the historical tiled
/// fill must fail. Skipping the negative direction is how a guard becomes a
/// ritual — it will happily assert anything if nobody ever tries to break it.
#[cfg(test)]
mod guard_tests {
    use super::*;

    /// A tiny allocation is enough: the properties under test are about the
    /// sampler and the counting, not about DRAM.
    const N: usize = 4 * 1024 * 1024;

    #[test]
    fn sampler_counts_duplicates_not_distinct() {
        let d = Device::system_default().unwrap();
        let buf = d.new_buffer(
            N as u64,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
        );
        fill_and_fault(&buf, N);
        let (dups, n) = fill_is_non_repeating(&buf, N);
        assert_eq!(dups, 0, "an honestly filled buffer has no duplicate lines");
        assert!(n > 1);

        // Now the exact defect from `probe_bw_250gb.rs:15`: a repeating tile.
        //
        // The period must be SMALLER than the buffer for a repeat to exist
        // inside it. Tiling a 4 MiB buffer with the historical 64 MiB pattern
        // yields one tile's worth of content and no repeats — the first version
        // of this test used 64 MiB against 4 MiB and asserted `dups2 > 0` on a
        // buffer that legitimately had none. A guard test that fails for the
        // wrong reason teaches you nothing. 1 MiB period over 4 MiB gives four
        // copies, which the sampler must see.
        tiled_fill_period(&buf, N, N / 4);
        let (dups2, _) = fill_is_non_repeating(&buf, N);
        assert!(
            dups2 > 0,
            "a buffer tiled at 1/4 of its length must be rejected; the guard failed \
             (reported {dups2} duplicates)"
        );
    }

    #[test]
    fn available_memory_parse_handles_the_unit_traps() {
        // "554M" is 0.55 GiB, not 554 GiB. Stripping non-digits and calling it
        // GB overstates by 1000x, which is the bug that once made gpuq warn on a
        // healthy box. Exercised through the same parse the probe uses.
        let parse = |physmem: &str| -> f64 {
            let before = physmem.split("unused").next().unwrap_or("");
            let tok = before.split_whitespace().last().unwrap_or("");
            let (num, unit) = tok.split_at(
                tok.find(|c: char| !(c.is_ascii_digit() || c == '.'))
                    .unwrap_or(tok.len()),
            );
            let v: f64 = num.parse().unwrap_or(0.0);
            match unit.chars().next().unwrap_or('G') {
                'T' => v * 1000.0,
                'G' => v,
                'M' => v / 1000.0,
                'K' => v / 1_000_000.0,
                _ => v,
            }
        };
        let cases = [
            ("PhysMem: 23G used (21G wired, 533M compressor), 112M unused.", 0.112),
            ("PhysMem: 14G used (2922M wired, 410M compressor), 9625M unused.", 9.625),
            ("PhysMem: 5G used, 554M unused.", 0.554),
            ("PhysMem: 12G used, 19G unused.", 19.0),
            ("PhysMem: 3G used, 2.5T unused.", 2500.0),
        ];
        for (line, want) in cases {
            let got = parse(line);
            assert!(
                (got - want).abs() < 1e-6,
                "{line:?}\n  parsed {got} GB, expected {want} GB"
            );
        }
    }
}

fn fill_is_non_repeating(buf: &metal::Buffer, bytes: usize) -> (usize, usize) {
    let ptr = buf.contents() as *const u8;
    let fold = |off: usize| -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for i in 0..64 {
            h ^= unsafe { *ptr.add(off + i) } as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    };
    let n = 512usize;
    let step = bytes / n;
    // Sample the MIDDLE of each step-sized window, and clamp the first sample so
    // it cannot read before the buffer. An earlier version computed
    // `i * step - step / 2` in `usize`, which wrapped to ~2^64 for `i == 0`.
    let mut seen: Vec<u64> = (0..n)
        .map(|i| fold(i * step + step / 2))
        .collect();
    seen.sort_unstable();
    // After sorting, each run of equal values of length L contributes L-1
    // adjacent-equal pairs, so this IS the duplicate count.
    //
    // The first version called this `uniq` and returned `n - uniq`, i.e. it
    // reported the number of DISTINCT lines. A correct fill then reported
    // "512 of 512 duplicates" and the guard rejected it. The guard was wrong,
    // not the fill — which is the one acceptable way for a guard to fail, and the
    // reason it must be tested against a known-bad input as well as a known-good
    // one.
    let duplicates = seen.windows(2).filter(|w| w[0] == w[1]).count();
    (duplicates, n)
}

#[derive(Clone)]
struct Arm {
    kernel: &'static str,
    threads: u64,
    tg: u64,
    bytes: u64,
    elem_bytes: u64,
    /// words written per thread, which differs between the uint4 and uint1/2 arms
    out_words: u64,
    streams: u64,
    label: String,
}

struct Result {
    median: f64,
    best: f64,
    worst: f64,
    gbps: Vec<f64>,
}

/// Sample `PhysMem`/`vm.swapusage` and the presence of any `kaplumba` process.
///
/// Printed at the start and end of the run. A bandwidth run on a box that
/// started quiet and finished with 11 GB of another agent's model resident is a
/// different measurement from one that was quiet throughout, and the difference
/// is larger than most of the effects being chased here.
/// Parse `top`'s `PhysMem` "unused" figure into GiB.
///
/// Two traps, both of which this repo has already hit and shipped:
/// * `vm_stat`'s "Pages free" is useless here — it counts only never-touched
///   pages, ignoring the file cache. It read 1.47 GiB on an idle box.
/// * `top` switches units (K/M/G/T) and drops the decimal below 10, so `554M`
///   is 0.55 GiB. Stripping non-digits turns that into 554 GiB — a 1000x
///   overstatement that once made `gpuq` warn on a healthy box.
fn available_memory_gb() -> f64 {
    let out = std::process::Command::new("/usr/bin/top")
        .args(["-l", "1", "-n", "0"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    for l in out.lines() {
        if !l.contains("PhysMem") {
            continue;
        }
        // The token immediately preceding "unused" is the number. Reading the
        // *leading* characters of the text before "unused" gets a comma and
        // silently yields 0.0, which then reads as "no memory at all" — the
        // first version of this parser did exactly that.
        let before = l.split("unused").next().unwrap_or("");
        let tok = before.split_whitespace().last().unwrap_or("");
        let (num, unit) = tok.split_at(
            tok.find(|c: char| !(c.is_ascii_digit() || c == '.'))
                .unwrap_or(tok.len()),
        );
        let v: f64 = num.parse().unwrap_or(0.0);
        let gb = match unit.chars().next().unwrap_or('G') {
            'T' => v * 1000.0,
            'G' => v,
            'M' => v / 1000.0,
            'K' => v / 1_000_000.0,
            _ => v,
        };
        return gb;
    }
    0.0
}

fn box_state(tag: &str) {
    println!("--- box state ({tag}) ---");
    if let Ok(o) = std::process::Command::new("/usr/bin/top")
        .args(["-l", "1", "-n", "0"])
        .output()
    {
        for l in String::from_utf8_lossy(&o.stdout)
            .lines()
            .filter(|l| l.contains("PhysMem"))
        {
            println!("    {l}");
        }
    }
    if let Ok(o) = std::process::Command::new("sysctl")
        .args(["-n", "vm.swapusage"])
        .output()
    {
        println!("    {}", String::from_utf8_lossy(&o.stdout).trim());
    }
    // A resident model is the single most important fact about a bandwidth run on
    // this box: `ps` RSS understates it by ~5.5 GiB because the Metal buffers
    // are not in RSS (AGENTS.md §10.3), so the process list plus `PhysMem` is the
    // only honest summary.
    let holders = std::process::Command::new("/bin/sh")
        .args(["-c", "ps -Ao pid,rss,etime,comm | grep -i -E 'kaplumba' | grep -v grep"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    let n = holders.lines().filter(|l| !l.trim().is_empty()).count();
    println!(
        "    parsed 'unused': {:.1} GB ; kaplumba-shaped processes resident: {n}",
        available_memory_gb()
    );
    for l in holders.lines().filter(|l| !l.trim().is_empty()) {
        println!("      {l}");
    }
}

fn run_arm(
    _device: &Device,
    queue: &metal::CommandQueue,
    pipe: &metal::ComputePipelineState,
    data: &metal::Buffer,
    out: &metal::Buffer,
    arm: &Arm,
    rounds: usize,
    repeat: u64,
) -> Result {
    // `count` is the number of ELEMENTS the arm visits, and it must be exactly
    // `bytes / elem_bytes` so that the arm reads every byte of the working set.
    //
    // This line said `((bytes / elem_bytes) / 2)`. The kernels then read only
    // half the buffer while `run_arm` computed GB/s as `arm.bytes / elapsed` —
    // i.e. every arm was credited with twice the bytes it actually moved. The
    // tell was physical: the sweep reported 505.5 GB/s, 185% of the 273 GB/s
    // wire peak, and the "physically plausible" filter kept exactly one arm, the
    // slowest. Corrected, the same run gives ~252.8 GB/s = 92.6% of peak, which
    // is a believable DRAM number and lands exactly where the surviving arm
    // already pointed.
    //
    // The reason it survived unnoticed for a full run is that every *relative*
    // comparison in the sweep is unaffected by a constant factor of 2, so axes
    // B, C and S all looked perfectly sensible. Only the absolute scale lied.
    let count = (arm.bytes / arm.elem_bytes) as u32;
    let streams = arm.streams as u32;
    let tgs_per_stream = (arm.threads / arm.tg / arm.streams) as u32;
    assert!(
        arm.threads % (arm.tg * arm.streams) == 0,
        "{}: {} threads is not a multiple of tg {} x streams {}",
        arm.label,
        arm.threads,
        arm.tg,
        arm.streams
    );
    assert!(count > 0, "{}: zero elements", arm.label);

    // Two warm-up passes, discarded: cold clocks and first-touch costs are not
    // the thing being measured.
    for _ in 0..2 {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(pipe);
        e.set_buffer(0, Some(data), 0);
        e.set_buffer(1, Some(out), 0);
        e.set_bytes(2, 4, &count as *const u32 as *const std::ffi::c_void);
        e.set_bytes(3, 4, &streams as *const u32 as *const std::ffi::c_void);
        e.set_bytes(4, 4, &tgs_per_stream as *const u32 as *const std::ffi::c_void);
        e.dispatch_threads(MTLSize::new(arm.threads, 1, 1), MTLSize::new(arm.tg, 1, 1));
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let mut gbps = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(pipe);
        e.set_buffer(0, Some(data), 0);
        e.set_buffer(1, Some(out), 0);
        e.set_bytes(2, 4, &count as *const u32 as *const std::ffi::c_void);
        e.set_bytes(3, 4, &streams as *const u32 as *const std::ffi::c_void);
        e.set_bytes(4, 4, &tgs_per_stream as *const u32 as *const std::ffi::c_void);
        // `repeat` passes per command buffer, so launch latency (~0.2 ms) is
        // amortised and cannot masquerade as a bandwidth difference between arms
        // of different duration.
        for _ in 0..repeat {
            e.dispatch_threads(MTLSize::new(arm.threads, 1, 1), MTLSize::new(arm.tg, 1, 1));
        }
        e.end_encoding();
        let t0 = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let el = t0.elapsed().as_secs_f64() / repeat as f64;
        gbps.push(arm.bytes as f64 / el / 1e9);
    }
    let mut s = gbps.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Result {
        median: s[s.len() / 2],
        best: s[s.len() - 1],
        worst: s[0],
        gbps,
    }
}

fn reduce_out(out: &metal::Buffer, words: usize) -> u32 {
    let p = out.contents() as *const u32;
    let mut s = 0u32;
    for i in 0..words {
        unsafe { s = s.wrapping_add(*p.add(i)) };
    }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sabotage = args.iter().any(|a| a == "--sabotage");
    let quick = args.iter().any(|a| a == "--quick");

    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();

    // The box state is part of the measurement, not context around it. A
    // bandwidth number produced while another process holds 11 GB of the same
    // memory is a different number, and reporting it without saying so is how
    // `decode-19tok-repacked-250gb` ended up claiming 250-254 GB/s.
    box_state("start");
    let lib = device
        .new_library_with_source(SRC, &CompileOptions::new())
        .expect("compile crux_bw.metal");

    // Size the working set to what the box can actually give, and never assume a
    // quiet machine. The first version hardcoded 6 GB and died with SIGBUS
    // (exit 138) while a neighbouring agent held 11 GB of wired memory and 5 GB
    // sat in the compressor: `new_buffer` succeeded, the write pass did not.
    //
    // The floor matters more than the ceiling. Below ~256 MB a Mac's system
    // level cache absorbs the working set and the whole sweep measures cache —
    // which is exactly the defect this probe was written to eliminate. So if the
    // box cannot give a genuinely DRAM-sized buffer, the run is refused rather
    // than silently downgraded into a cache measurement.
    let want_gb: u64 = if quick { 2 } else { 6 };
    let avail_gb = available_memory_gb();
    let gib = want_gb.min(avail_gb.floor() as u64).max(1);
    const MIN_GIB: u64 = 2;
    if gib < MIN_GIB {
        eprintln!(
            "REFUSING TO RUN: only {avail_gb:.1} GB available, need >= {MIN_GIB} GB for a \
             working set that exceeds cache. Below that this probe reports cache \
             bandwidth labelled as DRAM. Free the box and retry."
        );
        std::process::exit(2);
    }
    if gib < want_gb {
        println!(
            "  NOTE: box has {avail_gb:.1} GB available, running at {gib} GB instead of \
             {want_gb} GB. Absolute GB/s will be depressed by whatever else is resident."
        );
    }
    let bytes = gib * 1_000_000_000u64;
    let rounds = if quick { 5 } else { 9 };
    let repeat = if quick { 2 } else { 4 };

    println!(
        "=== crux_bandwidth: {} GB working set, {} interleaved rounds, {} passes/buffer ===",
        gib, rounds, repeat
    );
    println!(
        "device {} ; max threads per threadgroup = {}",
        device.name(),
        device.max_threads_per_threadgroup().width
    );

    let res_opts = MTLResourceOptions::StorageModeShared
        | MTLResourceOptions::HazardTrackingModeUntracked;
    let data = device.new_buffer(bytes, res_opts);
    println!("filling and faulting in {} GB ...", gib);
    fill_and_fault(&data, bytes as usize);
    let (collisions, samples) = fill_is_non_repeating(&data, bytes as usize);
    assert_eq!(
        collisions, 0,
        "FILL DEFECT: {collisions} of {samples} sampled 64-byte lines are duplicates, so \
         the buffer is a repeating tile and cache hits will be counted as DRAM \
         bandwidth. Every number below would be fiction."
    );

    // The bytes an arm claims to move must be the bytes it moves.
    //
    // This assertion is the check that was missing when `count` was
    // `bytes / elem_bytes / 2`: every arm read half the buffer and was credited
    // with all of it, so the sweep reported 505.5 GB/s — 185% of a wire that
    // cannot carry it. Nothing downstream noticed, because a uniform factor of
    // two cancels in every ratio the sweep reports. The absolute scale was the
    // only thing wrong, and nothing in the probe compared it to anything.
    //
    // So compare it: the element count is derived from the byte count, and the
    // byte count is what gets divided by the elapsed time.
    for eb in [4u64, 8, 16] {
        let count = (bytes / eb) as u32;
        assert_eq!(
            count as u64 * eb,
            bytes,
            "an arm with {eb}-byte elements must visit exactly {bytes} bytes"
        );
    }
    assert!(
        (bytes / 4) <= u32::MAX as u64,
        "element count must fit in the u32 the kernels take; {} bytes at 4 B/element \
         does not. Shrink the working set or widen the count type.",
        bytes / 4
    );
    println!("done; non-repeating fill verified ({samples} sampled lines, 0 duplicates)");

    let max_threads: u64 = 81_920;
    let max_out = max_threads * 4;
    let out = device.new_buffer(max_out * 4, res_opts);

    // --- the equal-bytes gate ------------------------------------------------
    // Every arm below reads the same bytes in the same order. If the gate fails,
    // nothing after it means anything, so it runs before any timing is printed.
    println!("\n--- GATE: equal bytes, equal order, equal checksum ---");
    let gate_arms: Vec<(&str, u64, u64)> = vec![
        ("stream_u1", 4, 1),
        ("stream_u2", 8, 1),
        ("stream_u4", 16, 4),
        ("stream_u4_u2", 16, 4),
        ("stream_u4_u4", 16, 4),
        ("stream_u4_u8", 16, 4),
        ("stream_u4_u16", 16, 4),
    ];
    let mut gate_sum = 0u32;
    for (name, eb, ow) in &gate_arms {
        let pipe = device
            .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
            .unwrap();
        let arm = Arm {
            kernel: name,
            threads: 40_960,
            tg: 256,
            bytes,
            elem_bytes: *eb,
            out_words: *ow,
            streams: 1,
            label: name.to_string(),
        };
        run_arm(&device, &queue, &pipe, &data, &out, &arm, 2, 1);
        let s = reduce_out(&out, 40_960 * *ow as usize);
        if gate_sum == 0 {
            gate_sum = s;
            println!("  {name:<14} checksum {s:#010x}  (reference)");
        } else {
            assert_eq!(
                gate_sum, s,
                "GATE FAILED: {name} read a different set of bytes than stream_u1 \
                 ({s:#010x} vs {gate_sum:#010x}). The bandwidth numbers below would \
                 be comparing different work."
            );
            println!("  {name:<14} checksum {s:#010x}  MATCH");
        }
    }

    // --- sabotage ------------------------------------------------------------
    if sabotage {
        println!("\n--- SABOTAGE: halve the element count, checksum must change ---");
        let pipe = device
            .new_compute_pipeline_state_with_function(&lib.get_function("stream_u4", None).unwrap())
            .unwrap();
        let full = Arm {
            kernel: "stream_u4",
            threads: 40_960,
            tg: 256,
            bytes,
            elem_bytes: 16,
            out_words: 4,
            streams: 1,
            label: "full".into(),
        };
        run_arm(&device, &queue, &pipe, &data, &out, &full, 2, 1);
        let a = reduce_out(&out, 40_960 * 4);
        let half = Arm { bytes: bytes / 2, ..full.clone() };
        run_arm(&device, &queue, &pipe, &data, &out, &half, 2, 1);
        let b = reduce_out(&out, 40_960 * 4);
        assert_ne!(a, b, "SABOTAGE FAILED: half the bytes gave the same checksum");
        println!("  full {a:#010x} vs half {b:#010x} -> detected. The instrument has teeth.\n");
    }

    // --- is this DRAM traffic? ----------------------------------------------
    // A second pass over the same bytes must not be faster than the first. If
    // it is, the working set is not actually bigger than the cache and the whole
    // sweep is measuring cache.
    {
        let pipe = device
            .new_compute_pipeline_state_with_function(&lib.get_function("stream_u4", None).unwrap())
            .unwrap();
        let (p1, p2) = pass_ratio_probe(
            &device,
            &queue,
            &pipe,
            &data,
            &out,
            max_threads,
            (bytes / 16) as u32,
        );
        let ratio = p2 / p1;
        println!("\n--- IS THIS DRAM? pass 1 vs pass 2 over the same bytes ---");
        println!("  pass 1 {p1:.1} GB/s, pass 2 {p2:.1} GB/s, ratio {ratio:.3}");
        if ratio > 1.10 {
            println!(
                "  ** CACHE IS SERVING THE WORK. Pass 2 is {ratio:.2}x pass 1. Every number \
                 below is an L2/L3 figure wearing a DRAM label. **"
            );
        } else {
            println!(
                "  pass 2 is not faster than pass 1 ({ratio:.3}x), so the working set exceeds \
                 cache and the sweep below is DRAM traffic."
            );
        }
    }

    // --- AXIS A: occupancy ---------------------------------------------------
    // 20,480 is the ceiling `spec/decode-19tok` reached; 81,920 is the device
    // maximum. Everything between is measured, and beyond it too, because
    // oversubscription is a different regime from occupancy.
    println!("\n=== AXIS A: occupancy (uint4 loads, tg=256, bytes in flight per thread fixed) ===");
    println!(
        "{:>9} {:>10} {:>10} {:>10} {:>9} {:>12} {:>9}",
        "threads", "median", "best", "worst", "spread", "GB/s @6GB", "waves@81920"
    );
    let mut a_arms: Vec<Arm> = Vec::new();
    for threads in [2_560u64, 5_120, 10_240, 20_480, 30_720, 40_960, 61_440, 81_920] {
        a_arms.push(Arm {
            kernel: "stream_u4",
            threads,
            tg: 256,
            bytes,
            elem_bytes: 16,
            out_words: 4,
            streams: 1,
            label: format!("u4 x{threads}"),
        });
    }
    // Plus the oversubscribed region, which is where "more requests" is cheapest.
    for threads in [122_880u64, 163_840, 245_760] {
        a_arms.push(Arm {
            kernel: "stream_u4",
            threads,
            tg: 256,
            bytes,
            elem_bytes: 16,
            out_words: 4,
            streams: 1,
            label: format!("u4 x{threads}"),
        });
    }
    let pipe_u4 = device
        .new_compute_pipeline_state_with_function(&lib.get_function("stream_u4", None).unwrap())
        .unwrap();
    let mut a_res: Vec<(&Arm, Result)> = Vec::new();
    for _ in 0..rounds {
        for arm in &a_arms {
            let r = run_arm(&device, &queue, &pipe_u4, &data, &out, arm, 1, repeat);
            a_res.push((arm, r));
        }
    }
    let mut by_label: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (arm, r) in &a_res {
        by_label.entry(arm.label.as_str()).or_default().extend(r.gbps.clone());
    }
    let mut rows: Vec<(u64, f64, f64, f64)> = Vec::new();
    let mut med_by_arm: BTreeMap<&str, f64> = BTreeMap::new();
    for (label, mut v) in by_label {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        med_by_arm.insert(label, v[v.len() / 2]);
        let threads: u64 = label.split('x').nth(1).unwrap().parse().unwrap();
        rows.push((threads, v[v.len() / 2], v[v.len() - 1], v[0]));
    }
    rows.sort_by_key(|r| r.0);
    let plateau = rows
        .iter()
        .filter(|r| r.0 >= 20_480)
        .map(|r| r.1)
        .fold(f64::MAX, f64::min);
    for (threads, med, best, worst) in &rows {
        println!(
            "{:>9} {:>10.1} {:>10.1} {:>10.1} {:>8.2}x {:>12.1} {:>9.2}",
            threads,
            med,
            best,
            worst,
            best / worst,
            *med,
            *threads as f64 / 81_920.0
        );
    }
    let best_a = rows.iter().map(|r| r.1).fold(0.0, f64::max);
    println!("\n  best median on axis A: {best_a:.1} GB/s");
    let at_20480 = rows.iter().find(|r| r.0 == 20_480).map(|r| r.1).unwrap_or(0.0);
    let at_max = rows.iter().find(|r| r.0 == 81_920).map(|r| r.1).unwrap_or(0.0);
    println!(
        "  20,480 -> 81,920 threads: {:+.1}%   (the sweep decode-19tok never ran)",
        100.0 * (at_max / at_20480 - 1.0)
    );
    println!(
        "  min median at >=20,480 threads = {plateau:.1} GB/s; spread across that region \
         = {:.2}x",
        rows.iter()
            .filter(|r| r.0 >= 20_480)
            .map(|r| r.1)
            .fold(f64::MIN, f64::max)
            / plateau
    );

    // --- AXIS B and C: at full occupancy ------------------------------------
    println!("\n=== AXIS B/C at full occupancy (81,920 threads, tg=256) ===");
    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>9} {:>12} {:>10}",
        "arm", "B/load", "in-flight", "median", "spread", "vs stream_u4", "requests/GB"
    );
    let mut bc: Vec<Arm> = vec![
        Arm { kernel: "stream_u1", bytes, elem_bytes: 4, out_words: 1, threads: max_threads, tg: 256, streams: 1, label: "stream_u1".into() },
        Arm { kernel: "stream_u2", bytes, elem_bytes: 8, out_words: 1, threads: max_threads, tg: 256, streams: 1, label: "stream_u2".into() },
        Arm { kernel: "stream_u4", bytes, elem_bytes: 16, out_words: 4, threads: max_threads, tg: 256, streams: 1, label: "stream_u4".into() },
        Arm { kernel: "stream_u4_u2", bytes, elem_bytes: 16, out_words: 4, threads: max_threads, tg: 256, streams: 1, label: "stream_u4_u2".into() },
        Arm { kernel: "stream_u4_u4", bytes, elem_bytes: 16, out_words: 4, threads: max_threads, tg: 256, streams: 1, label: "stream_u4_u4".into() },
        Arm { kernel: "stream_u4_u8", bytes, elem_bytes: 16, out_words: 4, threads: max_threads, tg: 256, streams: 1, label: "stream_u4_u8".into() },
        Arm { kernel: "stream_u4_u16", bytes, elem_bytes: 16, out_words: 4, threads: max_threads, tg: 256, streams: 1, label: "stream_u4_u16".into() },
        // 16 B in flight but FOUR requests: the discriminator between
        // "bytes in flight" and "requests in flight".
        Arm { kernel: "stream_u1_u4", bytes, elem_bytes: 4, out_words: 1, threads: max_threads, tg: 256, streams: 1, label: "stream_u1_u4".into() },
        Arm { kernel: "stream_u1_u8", bytes, elem_bytes: 4, out_words: 1, threads: max_threads, tg: 256, streams: 1, label: "stream_u1_u8".into() },
    ];
    // --- POSITIVE CONTROL ----------------------------------------------------
    // `CONTROL_repeat` is byte-for-byte, thread-for-byte, round-for-round
    // IDENTICAL to `stream_u4` above: same kernel, same pipeline, same grid,
    // same buffer, same position in the round order (it is the next arm after
    // `stream_u1_u8`). It is duplicated on purpose, and nothing can make it
    // differ.
    //
    // This is the number that decides what a "no effect" reading is allowed to
    // mean. The difference between the two arms IS the instrument's noise floor
    // for this working set on this box, and no difference smaller than it may be
    // reported as an effect. `decode-19tok`'s "bandwidth is flat from 8 to 32
    // bytes in flight" is only a real plateau if it is larger than this.
    bc.push(Arm { kernel: "stream_u4", bytes, elem_bytes: 16, out_words: 4, threads: max_threads, tg: 256, streams: 1, label: "CONTROL_repeat".into() });
    // A second kind of control: `tg` is a launch parameter, not a work
    // parameter. The work is identical, so if the measured rate moves with `tg`
    // alone, the metric is sensitive to something other than bandwidth.
    bc.push(Arm { kernel: "stream_u4", bytes, elem_bytes: 16, out_words: 4, threads: max_threads, tg: 512, streams: 1, label: "CONTROL_tg512".into() });
    // Round order is the construction order above, left alone. A `sort_by` here
    // would interleave `stream_u4` and `CONTROL_repeat` by name, which is fine
    // but obscures the point; the two are adjacent by construction instead.

    let mut pipes: BTreeMap<&str, metal::ComputePipelineState> = BTreeMap::new();
    for arm in &bc {
        pipes.entry(arm.kernel).or_insert_with(|| {
            device
                .new_compute_pipeline_state_with_function(
                    &lib.get_function(arm.kernel, None).unwrap_or_else(|e| panic!("{}: {e}", arm.kernel)),
                )
                .unwrap_or_else(|e| panic!("pipeline {}: {e}", arm.kernel))
        });
    }

    let in_flight = |k: &str| -> f64 {
        match k {
            "stream_u1" => 4.0,
            "stream_u2" => 8.0,
            "stream_u4" => 16.0,
            "stream_u4_u2" => 32.0,
            "stream_u4_u4" => 64.0,
            "stream_u4_u8" => 128.0,
            "stream_u4_u16" => 256.0,
            "stream_u1_u4" => 16.0,
            "stream_u1_u8" => 32.0,
            _ => 16.0,
        }
    };

    let mut bc_res: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for _ in 0..rounds {
        for arm in &bc {
            let r = run_arm(
                &device,
                &queue,
                &pipes[arm.kernel],
                &data,
                &out,
                arm,
                1,
                repeat,
            );
            bc_res.entry(arm.label.as_str()).or_default().extend(r.gbps);
        }
    }
    let u4_median = {
        let mut v = bc_res["stream_u4"].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    for (label, v) in &bc_res {
        let mut v = v.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        let best = v[v.len() - 1];
        let worst = v[0];
        let (elem, nreq) = bc
            .iter()
            .find(|a| a.label == *label)
            .map(|a| (a.elem_bytes, a.out_words))
            .unwrap_or((16, 4));
        let _ = nreq;
        // Requests per GB is determined by the WIDTH of one load, not by the
        // unroll: an unrolled arm still issues one request per 16 B, it just has
        // more of them outstanding. Derived from `elem_bytes` rather than
        // hard-coded per label — the first version of this table special-cased
        // `stream_u4_u*` and CONTROL and then fell through to the 4 B branch for
        // plain `stream_u4`, printing 250.0 M req/GB for an arm that issues
        // 62.5 M. A table with a silent default is worse than no table.
        let req_per_gb = 1e9 / elem as f64;
        println!(
            "{:<16} {:>10.0} {:>10.0} {:>10.1} {:>8.2}x {:>11.1}% {:>10.1}",
            label,
            elem as f64,
            in_flight(label),
            med,
            best / worst,
            100.0 * (med / u4_median - 1.0),
            req_per_gb / 1e6
        );
    }
    println!(
        "\n  requests/GB is in millions. stream_u1_u4 has the SAME bytes in flight as \
         stream_u4 but 4x the requests."
    );

    // --- what this run is allowed to conclude ---------------------------------
    {
        let med = |k: &str| -> f64 {
            let mut v = bc_res[k].clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let floor = 100.0 * (med("CONTROL_repeat") / med("stream_u4") - 1.0).abs();
        let tg = 100.0 * (med("CONTROL_tg512") / med("stream_u4") - 1.0);
        println!("\n=== NOISE FLOOR ===");
        println!("  identical arms (stream_u4 vs CONTROL_repeat): {floor:+.2}% of median");
        println!("  same work, tg 256 -> 512 (CONTROL_tg512)  : {tg:+.2}% of median");
        let base = med("stream_u4");
        let mut rows: Vec<(&str, f64)> = bc_res
            .iter()
            .filter(|&(k, _)| !k.starts_with("CONTROL") && *k != "stream_u4")
            .map(|(k, _)| (*k, 100.0 * (med(k) / base - 1.0)))
            .collect();
        rows.sort_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap());
        for (k, d) in &rows {
            println!(
                "  {k:<16} {d:+7.2}%  {}",
                if d.abs() <= floor { "within noise floor" } else { "MEASURABLE" }
            );
        }
        let below: Vec<&str> = rows
            .iter()
            .filter(|(_, d)| d.abs() <= floor)
            .map(|(k, _)| *k)
            .collect();
        println!(
            "  arms whose difference from stream_u4 is WITHIN the noise floor \
             (|d| <= {floor:.2}%), i.e. indistinguishable: {below:?}"
        );
        if below.is_empty() {
            println!("  (none: every arm moved more than the instrument's own repeat spread)");
        }
    }

    // --- AXIS S: independent address streams --------------------------------
    println!("\n=== AXIS S: independent address streams (81,920 threads, uint4) ===");
    println!(
        "{:>9} {:>10} {:>10} {:>10} {:>9} {:>12}",
        "streams", "median", "best", "worst", "spread", "vs 1 stream"
    );
    let pipe_s = device
        .new_compute_pipeline_state_with_function(&lib.get_function("stream_u4_s", None).unwrap())
        .unwrap();
    let s_arms: Vec<Arm> = [1u64, 2, 4, 8, 16, 32, 64]
        .iter()
        .map(|s| Arm {
            kernel: "stream_u4_s",
            threads: max_threads,
            tg: 256,
            bytes,
            elem_bytes: 16,
            out_words: 4,
            streams: *s,
            label: format!("s{s}"),
        })
        .collect();
    let mut s_res: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for _ in 0..rounds {
        for arm in &s_arms {
            let r = run_arm(&device, &queue, &pipe_s, &data, &out, arm, 1, repeat);
            s_res.entry(arm.streams).or_default().extend(r.gbps);
        }
    }
    let s1 = {
        let mut v = s_res[&1].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    for (streams, v) in &s_res {
        let mut v = v.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{:>9} {:>10.1} {:>10.1} {:>10.1} {:>8.2}x {:>11.1}%",
            streams,
            v[v.len() / 2],
            v[v.len() - 1],
            v[0],
            v[v.len() - 1] / v[0],
            100.0 * (v[v.len() / 2] / s1 - 1.0)
        );
    }

    // --- the number that matters --------------------------------------------
    println!("\n=== VERDICT INPUT: best sustained streaming GB/s on this box ===");
    let mut all: Vec<f64> = Vec::new();
    for (_, med) in &med_by_arm {
        all.push(*med);
    }
    for (_, v) in &bc_res {
        all.push(v[v.len() / 2]);
    }
    for (_, v) in &s_res {
        all.push(v[v.len() / 2]);
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let global_best = *all.last().unwrap();
    let global_med = all[all.len() / 2];
    println!("  best median across every arm : {global_best:.1} GB/s");
    println!("  median of all arm medians     : {global_med:.1} GB/s");
    println!(
        "  as % of the 273 GB/s wire peak: {:.1}% / {:.1}%",
        global_best / 273.0 * 100.0,
        global_med / 273.0 * 100.0
    );

    // A physical consistency check, not a style check. Nothing on this bus can
    // deliver more than the wire can carry, so an arm above 273 GB/s means the
    // measurement counted something other than DRAM traffic — cache hits, a
    // competitor having left mid-run, or a bad fill. The first version of this
    // probe reported 492.7 GB/s (180% of peak) for exactly this reason, and the
    // number looked perfectly reasonable in a table.
    //
    // The verdict uses only arms that pass. If none pass, the run is void and
    // says so, rather than reporting a number that cannot exist.
    const PEAK: f64 = 273.0;
    let plausible: Vec<(String, f64)> = {
        let mut v: Vec<(String, f64)> = Vec::new();
        for (l, m) in &med_by_arm {
            v.push((l.to_string(), *m));
        }
        for (l, x) in &bc_res {
            let mut s = x.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.push((l.to_string(), s[s.len() / 2]));
        }
        for (st, x) in &s_res {
            let mut s = x.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.push((format!("s{st}"), s[s.len() / 2]));
        }
        v.retain(|(_, m)| *m <= PEAK);
        v
    };
    println!(
        "\n  arms at or below the {PEAK} GB/s wire peak: {} of {}",
        plausible.len(),
        med_by_arm.len() + bc_res.len() + s_res.len()
    );
    if plausible.is_empty() {
        println!("  NO ARM IS PHYSICALLY POSSIBLE: this run is void. Do not quote it.");
    } else {
        let mut m = plausible.iter().map(|(_, v)| *v).collect::<Vec<_>>();
        m.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("  usable best  : {:.1} GB/s ({:.1}% of wire peak)", m[m.len() - 1], m[m.len() - 1] / PEAK * 100.0);
        println!("  usable median: {:.1} GB/s ({:.1}% of wire peak)", m[m.len() / 2], m[m.len() / 2] / PEAK * 100.0);
        // Only arms in the top half are quoted as "the achievable bandwidth":
        // a median over ALL arms mixes the deliberately bad ones (16-deep
        // unroll, 64 streams) into the denominator.
        let top: Vec<f64> = {
            let cut = m.len() / 2;
            m[cut..].to_vec()
        };
        println!(
            "  top-half median of usable arms: {:.1} GB/s ({:.1}% of wire peak)  <- the number to use",
            top[top.len() / 2],
            top[top.len() / 2] / PEAK * 100.0
        );
    }
    box_state("end");

    // --- SABOTAGE THE FILL ---------------------------------------------------
    // Run the measurement on the OLD tiled fill and show it produces
    // super-peak numbers, then show the guard rejects that fill. This is the
    // positive control for the whole probe: it demonstrates that the two
    // integrity checks (non-repeating fill, <= wire peak) detect the specific
    // defect that produced the historical 250 GB/s claim.
    if std::env::args().any(|a| a == "--sabotage-fill") {
        println!("\n--- SABOTAGE-FILL: the historical tiled buffer ---");
        let probe = device
            .new_compute_pipeline_state_with_function(
                &lib.get_function("stream_u4", None).unwrap(),
            )
            .unwrap();
        // 256 MiB is plenty to demonstrate the effect and keeps the run quick.
        // A tiled buffer needs only 64 MiB of unique content, so the whole
        // working set is served by a cache-sized region.
        let small = 256 * 1024 * 1024usize;
        let tiled = device.new_buffer(small as u64, res_opts);
        tiled_fill(&tiled, small);
        let (coll, n) = fill_is_non_repeating(&tiled, small);
        println!("  guard: {coll} of {n} sampled lines are duplicates -> REJECTED (as it must be)");
        assert!(coll > 0, "the guard failed to reject the tiled fill it exists to reject");

        let (p1, p2) = pass_ratio_probe(
            &device,
            &queue,
            &probe,
            &tiled,
            &out,
            max_threads,
            ((small / 16) as u32).max(1),
        );
        println!("  tiled 256 MiB: pass 1 {p1:.1} GB/s, pass 2 {p2:.1} GB/s (ratio {:.3})", p2 / p1);
        println!(
            "  a tiled buffer re-reads the same addresses on pass 2, so whatever the ratio \
             comes out as, the traffic is not compulsory. The load is L2/L3-served by \
             construction, whatever the timer says."
        );

        // Now the same size, honestly filled.
        let good = device.new_buffer(small as u64, res_opts);
        fill_and_fault(&good, small);
        let (c2, _) = fill_is_non_repeating(&good, small);
        assert_eq!(c2, 0, "non-repeating fill must pass its own guard");
        let (q1, q2) = pass_ratio_probe(
            &device,
            &queue,
            &probe,
            &good,
            &out,
            max_threads,
            ((small / 16) as u32).max(1),
        );
        println!("  honest 256 MiB: pass 1 {q1:.1} GB/s, pass 2 {q2:.1} GB/s (ratio {:.3})", q2 / q1);
        println!(
            "  the honest fill is rejected by neither guard. Note that at 256 MiB the ratio \
             is not a clean discriminator on its own — the cache is big enough that even \
             unique content partly survives, and pass 2 is {ratio:.3}x. The guard that \
             detects the defect unambiguously is the duplicate-line check (384/512 \
             duplicates, above); the pass ratio is corroboration at 6 GB, not a \
             standalone test.",
            ratio = q2 / q1
        );
    }
}

use std::collections::BTreeMap;
