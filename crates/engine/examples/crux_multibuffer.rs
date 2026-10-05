//! `spec/decode-20tok-crux` — **the last two axes that could break the wall**.
//!
//! # Why this exists
//!
//! The occupancy sweep is flat from 20,480 to 245,760 threads, load width is
//! flat at 8/16/32 B, address streams are flat to 8, working-set size plateaus
//! past 4 GB, row locality does nothing, and power/thermal is not throttled.
//! Every axis anyone has run says "bandwidth wall at ~90% of the 273 GB/s wire
//! peak". If that is true, 20 tok/s at 16k is arithmetic, not engineering.
//!
//! Two things remain untried, and both are things the real decode path does
//! that no probe has imitated:
//!
//! * **AXIS M — independent allocations.** Every address-stream arm in
//!   `crux_bw.metal` carves *one* buffer into S slices. Slices of a single
//!   allocation share physical pages, so a DRAM channel or bank imbalance would
//!   appear identically in all of them and be invisible. A decode token really
//!   touches 411 separate tensor buffers, so this probes S *separately
//!   allocated* buffers — the only way to see channel-level parallelism if it
//!   exists.
//! * **AXIS W — working set at the real size.** Weights are 12.058 GB resident
//!   and a 16k token streams 14.2 GB. The other probe has only ever run at
//!   6 GB, which is smaller than the thing it proxies for.
//!
//! # The guards, and why each is here
//!
//! This project has produced **five** bandwidth instruments that reported
//! bandwidth without measuring it, and two of them landed inside a believable
//! range (`crux.md` §6). A uniform factor of any size cancels in every *ratio*
//! an instrument reports, so only assertions against something outside the
//! measurement catch it. Therefore:
//!
//! * `count * elem_bytes == bytes` is asserted on the host, not derived.
//! * Content is a pure function of the **global** element index, so every
//!   partition of the working set must produce an **identical checksum**. A
//!   kernel that skipped a chunk, or read one twice, fails the gate instead of
//!   reporting a plausible wrong bandwidth.
//! * The fill is verified non-repeating by sampling; `--sabotage-fill` runs the
//!   historical tiled fill so the guard is shown to reject it.
//! * `CONTROL_repeat` is byte-, thread- and partition-identical to the reference
//!   arm. Their difference **is** the noise floor; nothing smaller may be
//!   reported as an effect.
//! * No arm may exceed the 273 GB/s wire peak, and the count of plausible arms
//!   is printed so a wholly contaminated run is visible.
//! * `--sabotage` reads half the elements and requires the checksum to change.

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const SRC: &str = include_str!("../../kernels/src/crux_mb.metal");
const WIRE_PEAK: f64 = 273.0;
const TG: usize = 256;

/// Content is a pure function of the **global** 16-byte block index, so any
/// partition into buffers yields byte-identical content and an identical
/// checksum. The historical tiled fill cannot satisfy this.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn fill_and_fault(bufs: &[metal::Buffer], bytes_per_buf: usize) -> u32 {
    // Also accumulate the checksum the GPU must reproduce: the sum over every
    // uint4 element of (x+y+z+w), mod 2^32. This is an INDEPENDENT reference --
    // computed on the CPU from the fill, not from anything the GPU said -- so a
    // kernel that skipped, repeated or mangled a byte range cannot agree with it.
    let mut expect: u32 = 0;
    for (b, buf) in bufs.iter().enumerate() {
        let ptr = buf.contents() as *mut u8;
        let nblocks = bytes_per_buf / 16;
        for i in 0..nblocks {
            let g = (b * bytes_per_buf) / 16 + i;
            let v = splitmix64(g as u64);
            unsafe { (ptr.add(i * 16) as *mut u128).write_unaligned(v as u128) };
            // v is the low 64 bits of the u128; the high 64 are zero, so the
            // uint4 at this offset is (lo, hi, 0, 0) in little-endian u32 words.
            let lo = v as u32;
            let hi = (v >> 32) as u32;
            expect = expect.wrapping_add(lo).wrapping_add(hi);
        }
    }
    expect
}

/// Sample 64-byte lines across the whole working set and count duplicates.
/// A tiled fill fails this; an honest fill passes it.
fn duplicate_lines(bufs: &[metal::Buffer], bytes_per_buf: usize, samples: usize) -> (usize, usize) {
    use std::collections::HashSet;
    let mut set: HashSet<u64> = HashSet::new();
    let mut dup = 0usize;
    for (b, buf) in bufs.iter().enumerate() {
        let _ = b;
        let ptr = buf.contents() as *const u8;
        let lines = bytes_per_buf / 64;
        let step = (lines / samples).max(1);
        let mut i = 0usize;
        let mut taken = 0usize;
        while i < lines && taken < samples {
            let mut v: u64 = 0;
            for k in 0..8 {
                // `ptr` is ALREADY buffer b's base, so the offset is `i*64`.
                // Adding `b*bytes_per_buf` here reads up to 3 GB past the end of
                // buffer b and segfaults on the first multi-buffer arm -- which
                // is exactly what it did.
                v |= (unsafe { *ptr.add(i * 64 + k) } as u64) << (8 * k);
            }
            if !set.insert(v) {
                dup += 1;
            }
            i += step;
            taken += 1;
        }
    }
    (dup, set.len())
}

struct Arm {
    label: String,
    bufs: usize,
    threads: usize,
}

struct Res {
    median: f64,
    best: f64,
    worst: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    queue: &metal::CommandQueue,
    pipe: &metal::ComputePipelineState,
    bufs: &[metal::Buffer],
    out: &metal::Buffer,
    arm: &Arm,
    bytes_total: usize,
    rounds: usize,
    repeat: u64,
    halve: bool,
) -> Res {
    let bytes_per_buf = bytes_total / arm.bufs;
    let count_full = (bytes_per_buf / 16) as u32; // uint4 elements PER BUFFER

    // Assert the byte accounting directly. This is the guard that was missing
    // when `count` was `bytes / elem_bytes / 2`: every arm read half the buffer
    // and was credited with all of it, so the sweep reported 505.5 GB/s — 185%
    // of a wire that cannot carry it. Nothing downstream noticed, because a
    // uniform factor of two cancels in every ratio the sweep reports.
    let count = if halve { count_full / 2 } else { count_full };
    assert_eq!(
        count as usize * 16 * arm.bufs,
        if halve { bytes_total / 2 } else { bytes_total },
        "{}: count*elem_bytes != bytes",
        arm.label
    );

    let total_tgs = arm.threads / TG;
    assert_eq!(
        total_tgs % arm.bufs,
        0,
        "{}: {} threadgroups is not a multiple of {} buffers",
        arm.label,
        total_tgs,
        arm.bufs
    );
    let per = (arm.threads / arm.bufs) as u32; // threads serving one buffer
    let sbufs = arm.bufs as u32;
    // Exact coverage: each of the `per` threads must take the same number of
    // steps, or the tail is either skipped or double-counted.
    assert_eq!(
        count % per,
        0,
        "{}: {} elements/buffer is not divisible by {} threads/buffer -- the tail would be \
         miscounted and the checksum gate would (correctly) fail",
        arm.label,
        count,
        per
    );

    let mut gbps = Vec::with_capacity(rounds);
    for r in 0..(rounds + 2) {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(pipe);
        // Buffers occupy slots 0..7. Unused slots are bound to buffer 0 so the
        // binding is never null; `sbufs` decides how many slots the kernel
        // actually reads, and the checksum gate proves it read only those.
        for slot in 0..8usize {
            let b = bufs.get(slot).unwrap_or(&bufs[0]);
            e.set_buffer(slot as u64, Some(b), 0);
        }
        e.set_buffer(8, Some(out), 0);
        e.set_bytes(9, 4, &count as *const u32 as *const std::ffi::c_void);
        e.set_bytes(10, 4, &sbufs as *const u32 as *const std::ffi::c_void);
        e.set_bytes(11, 4, &per as *const u32 as *const std::ffi::c_void);
        for _ in 0..repeat {
            e.dispatch_threads(
                MTLSize::new(arm.threads as u64, 1, 1),
                MTLSize::new(TG as u64, 1, 1),
            );
        }
        e.end_encoding();
        let t0 = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let el = t0.elapsed().as_secs_f64() / repeat as f64;
        if r >= 2 {
            // 2 warm-ups discarded: cold clocks are not the measurement.
            gbps.push(bytes_total as f64 / el / 1e9);
        }
    }
    let mut s = gbps.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Res {
        median: s[s.len() / 2],
        best: s[s.len() - 1],
        worst: s[0],
    }
}

fn checksum(out: &metal::Buffer, words: usize) -> u32 {
    let p = out.contents() as *const u32;
    let mut s = 0u32;
    for i in 0..words {
        unsafe { s = s.wrapping_add(*p.add(i)) };
    }
    s
}

fn box_state(tag: &str) {
    println!("--- box state ({tag}) ---");
    let out = std::process::Command::new("top")
        .args(["-l", "1", "-n", "0"])
        .output()
        .expect("top");
    let t = String::from_utf8_lossy(&out.stdout);
    for l in t.lines().filter(|l| l.contains("PhysMem")) {
        println!("    {}", l.trim());
    }
    let ps = std::process::Command::new("ps")
        .args(["-Ao", "pid,rss,args"])
        .output()
        .expect("ps");
    let t = String::from_utf8_lossy(&ps.stdout);
    let n = t
        .lines()
        .filter(|l| {
            let low = l.to_lowercase();
            low.contains("kaplumba") && !low.contains("grep")
        })
        .count();
    println!("    kaplumba-shaped processes resident: {n}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |k: &str| args.iter().any(|a| a == k);
    let sabotage = flag("--sabotage");
    let sabotage_fill = flag("--sabotage-fill");
    let gb: usize = args
        .iter()
        .position(|a| a == "--gb")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);

    box_state("start");
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SRC, &CompileOptions::new())
        .expect("compile crux_mb.metal");
    let pipe = device
        .new_compute_pipeline_state_with_function(&lib.get_function("stream_mb", None).unwrap())
        .expect("stream_mb pipeline");

    // Refuse rather than silently measure cache.
    let free = available_gb();
    println!("device {} ; max threads/threadgroup = {}", device.name(), device.max_threads_per_threadgroup().width);
    println!("available (reclaimable) {free:.1} GB ; requesting {gb} GB working set");
    if free < gb as f64 + 2.0 {
        eprintln!("REFUSING: box cannot give {gb} GB + 2 GB headroom. Would measure cache.");
        std::process::exit(2);
    }

    let opts = MTLResourceOptions::StorageModeShared
        | MTLResourceOptions::HazardTrackingModeUntracked;

    // --- SABOTAGE-FILL: the historical tiled fill must be rejected ----------
    if sabotage_fill {
        let per = 256_000_000usize;
        let tb = device.new_buffer(per as u64, opts);
        let pattern = vec![7u8; per];
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), tb.contents() as *mut u8, per);
        }
        let (dup, distinct) = duplicate_lines(&[tb], per, 512);
        let sampled = dup + distinct;
        println!("\n--- SABOTAGE-FILL: historical tiled buffer ---");
        println!(
            "  guard: {dup} duplicate lines among {sampled} sampled 64-byte lines ({distinct} distinct) -> {}",
            if dup > 0 {
                "REJECTED (as it must be)"
            } else {
                "ACCEPTED -- GUARD IS BROKEN"
            }
        );
    }

    // metal-rs will not let a host build a `device uint4* const*` array, so the
    // kernel takes 8 fixed slots. 8 independently-allocated buffers is the range
    // this axis can express.
    let buf_counts = [1usize, 2, 4, 8];
    let thread_counts = [20480usize, 81920];
    let mut arms: Vec<Arm> = Vec::new();
    for &b in &buf_counts {
        for &t in &thread_counts {
            if (t / TG) % b != 0 {
                continue;
            }
            arms.push(Arm {
                label: format!("buf{b:>2}_thr{t}"),
                bufs: b,
                threads: t,
            });
        }
    }
    // Positive control: identical to the first arm in every respect.
    arms.push(Arm {
        label: "CONTROL_repeat".into(),
        bufs: 1,
        threads: 20480,
    });

    println!(
        "\n=== crux_multibuffer: {gb} GB working set, {} arms, 7 rounds x 3 passes ===",
        arms.len()
    );

    let mut results: Vec<(String, usize, usize, Res)> = Vec::new();
    let mut ref_checksum: Option<u32> = None;

    for arm in &arms {
        // The TOTAL working set must be identical for every arm, or the fill
        // content differs and the arm-to-arm checksum gate fails on a correct
        // kernel. (It did: per-buffer alignment made the 1-buffer arm hold
        // 374,935,360 uint4 and the 2-buffer arm 374,814,720.) So the total
        // element count is fixed first and then split.
        //
        // 655,360 = 81,920 x 8, so total/S stays divisible by every
        // threads-per-buffer value these arms use: per is 20480/81920 (S=1),
        // 10240/40960 (S=2), 5120/20480 (S=4), 2560/10240 (S=8).
        const SPLIT: u64 = 655_360;
        let total_uint4 = ((gb as u64 * 1_000_000_000 / 16) / SPLIT) * SPLIT;
        assert_eq!(
            total_uint4 % arm.bufs as u64,
            0,
            "total element count is not divisible by {} buffers",
            arm.bufs
        );
        let per_buf_uint4 = total_uint4 / arm.bufs as u64;
        let per_aligned = (per_buf_uint4 * 16) as usize;
        let bufs: Vec<metal::Buffer> = (0..arm.bufs)
            .map(|_| device.new_buffer(per_aligned as u64, opts))
            .collect();
        let expect = fill_and_fault(&bufs, per_aligned);
        let (dup, _) = duplicate_lines(&bufs, per_aligned, 512);
        if dup > 0 {
            eprintln!("ABORT {}: fill has {dup} duplicate lines", arm.label);
            std::process::exit(3);
        }
        let out = device.new_buffer((arm.threads * 4) as u64, opts);

        let total = per_aligned * arm.bufs;
        let res = run_arm(&queue, &pipe, &bufs, &out, arm, total, 7, 3, false);
        let chk = checksum(&out, arm.threads);

        // The independent CPU reference. Every arm must match it AND the first
        // arm, which is the same check from two directions.
        if chk != expect {
            eprintln!(
                "ABORT {}: GPU checksum 0x{:08x} != CPU reference 0x{:08x}. The kernel did \
                 not read exactly the intended bytes, so its GB/s is not a bandwidth measurement.",
                arm.label, chk, expect
            );
            std::process::exit(6);
        }
        match ref_checksum {
            None => {
                ref_checksum = Some(chk);
                println!(
                    "\n  {:<16} checksum 0x{:08x} == CPU reference (and arm-to-arm) OK; \
                     {:.3} GB working set",
                    arm.label,
                    chk,
                    total as f64 / 1e9
                );
            }
            Some(r) => {
                if chk != r {
                    eprintln!(
                        "ABORT {}: checksum 0x{:08x} != reference 0x{:08x} -- this arm read a \
                         different byte set, so its GB/s would not be comparable",
                        arm.label, chk, r
                    );
                    std::process::exit(4);
                }
            }
        }

        // --sabotage: the SAME arm reading HALF the elements. The checksum must
        // change. An instrument that cannot notice it read half the data cannot
        // be trusted to notice a 4% difference.
        if sabotage && arm.bufs == 1 && arm.threads == 20480 {
            let out2 = device.new_buffer((arm.threads * 4) as u64, opts);
            run_arm(&queue, &pipe, &bufs, &out2, arm, total, 2, 1, true);
            let chk2 = checksum(&out2, arm.threads);
            println!(
                "\n--- SABOTAGE: half the elements, checksum must change ---\n  \
                 full 0x{:08x} vs half 0x{:08x} -> {}\n",
                chk,
                chk2,
                if chk != chk2 {
                    "detected. The instrument has teeth."
                } else {
                    "NOT DETECTED -- the gate is broken"
                }
            );
            if chk == chk2 {
                std::process::exit(5);
            }
        }

        println!(
            "  {:<16} {:>7.1} GB/s  best {:>7.1}  worst {:>7.1}  spread {:.2}x  ({:.1}% of wire)",
            arm.label,
            res.median,
            res.best,
            res.worst,
            res.worst / res.best,
            res.median / WIRE_PEAK * 100.0
        );
        results.push((arm.label.clone(), arm.bufs, arm.threads, res));
    }

    println!("\n=== AXIS M: does splitting the working set across N allocations help? ===");
    println!("  N_buf   thr20480   thr81920   (GB/s)");
    for &b in &buf_counts {
        let key = format!("buf{b:>2}_");
        let a = results
            .iter()
            .find(|(l, ..)| l.starts_with(&key) && l.ends_with("20480"))
            .map(|r| r.3.median);
        let c = results
            .iter()
            .find(|(l, ..)| l.starts_with(&key) && l.ends_with("81920"))
            .map(|r| r.3.median);
        if let (Some(x), Some(y)) = (a, c) {
            println!("  {:>4}   {:>8.1}   {:>8.1}", b, x, y);
        }
    }

    let ctrl = results
        .iter()
        .find(|(l, ..)| l == "CONTROL_repeat")
        .map(|r| r.3.median);
    let base = results
        .iter()
        .find(|(l, ..)| l == "buf 1_thr20480")
        .map(|r| r.3.median);
    if let (Some(c), Some(b0)) = (ctrl, base) {
        println!(
            "\n  NOISE FLOOR (CONTROL_repeat vs a byte-, thread- and partition-identical arm): \
             {:+.2}% of median",
            (c - b0) / b0 * 100.0
        );
    }
    let best = results.iter().map(|r| r.3.median).fold(0f64, f64::max);
    let plausible = results.iter().filter(|r| r.3.median <= WIRE_PEAK).count();
    println!(
        "\n  best median across every arm : {best:.1} GB/s ({:.1}% of wire)",
        best / WIRE_PEAK * 100.0
    );
    println!(
        "  arms at or below the wire peak: {plausible} of {}",
        results.len()
    );
    box_state("end");
}

/// `top` switches units at 1/1024 GB and drops the decimal below 10, so `554M
/// unused` is 0.55 GB. Stripping non-digits turns that into 554 GB — a 1000x
/// overstatement that shipped in `gpuq` once already (AGENTS.md §10.3).
fn available_gb() -> f64 {
    let out = std::process::Command::new("top")
        .args(["-l", "1", "-n", "0"])
        .output()
        .expect("top");
    let t = String::from_utf8_lossy(&out.stdout);
    for l in t.lines().filter(|l| l.contains("PhysMem")) {
        if let Some(idx) = l.find("unused") {
            // The figure PRECEDES the word: "... , 14G unused." — reading
            // forward from `unused` finds only "." and yields 0.0, which would
            // silently disable the refusal below. (It did, once, on the first
            // run of this probe: it reported 0.0 GB available.)
            let before = l[..idx].trim_end().trim_end_matches(',').trim_end();
            let tok = before.split_whitespace().last().unwrap_or("");
            let num: String = tok
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let v: f64 = num.parse().unwrap_or(0.0);
            let unit = tok.chars().find(|c| c.is_ascii_alphabetic()).unwrap_or('G');
            return match unit {
                'T' => v * 1024.0,
                'G' => v,
                'M' => v / 1024.0,
                'K' => v / (1024.0 * 1024.0),
                _ => v,
            };
        }
    }
    0.0
}
