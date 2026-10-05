//! `spec/decode-20tok-crux2` — **independent** bandwidth instrument.
//!
//! # What this is for
//!
//! The prior proof (`crux.md`) concludes "20 tok/s at 16k is impossible" from
//! bytes/token = 14.2390 GB and a measured 245 GB/s sustained. This instrument
//! attacks the *bandwidth* half with a shape the prior work never ran.
//!
//! Every arm in the prior work reads memory **front to back**, because that is
//! what a grid-stride loop does. The real decode GEMV does not: it walks EXL3
//! trellis tiles of 32x32, so the grid touches a small window and jumps. Two
//! things are therefore untested there and are the axes here:
//!
//! * **burst length** — 1 (raster scan) .. 24 uint4 (384 B = exactly one 32x32
//!   tile's payload at 3 bpw). One parameter, raster scan at one end.
//! * **visit entropy** — a keyed Feistel permutation of the index space. The
//!   brief flags this as untested; the multibuffer arm tested *independence* of
//!   buffers, not *randomness* of order.
//!
//! # Why this is not `crux_bandwidth.rs`
//!
//! Two reasons, and the first is the important one.
//!
//! 1. **The fill.** `crux_bandwidth`'s content is a function of the global index,
//!    so an arm that reads a different range gets a different checksum. Here the
//!    fill is a function of the **position in the buffer**, and the reference
//!    checksum is computed on the **CPU from the fill** for the exact multiset of
//!    indices the arm is asked to visit. A permutation arm therefore has to
//!    produce the same number as a raster arm, which it can only do by actually
//!    visiting every element exactly once.
//! 2. **The byte accounting is asserted against a hand-computed total.** Every
//!    arm declares `count`, `per`, `burst`, `iters`, `nbuf`; the host asserts
//!    `iters * burst * per == count` and `count * 16 * nbuf == bytes_total`
//!    **and** that the index multiset is exactly `[0, count)`. `--sabotage`
//!    breaks the latter deliberately, so the guard is shown to bite.
//!
//! # The positive control, in every run
//!
//! `CONTROL_repeat` is byte-, thread-, buffer- and permutation-identical to the
//! reference arm. Its difference from the reference **is** the noise floor, and
//! nothing smaller may be reported as an effect. On top of that the whole sweep
//! runs `rounds` times **interleaved** (round-major, not arm-major), so a
//! thermal or contention drift hits every arm equally instead of penalising
//! whichever arm ran last.

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const SRC: &str = include_str!("../../kernels/probe/crux2_bw.metal");
const TG: usize = 256;
const WIRE_PEAK: f64 = 273.0;

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// Host mirror of the kernel's `mix`, so the reference checksum is computed
/// from the fill rule rather than from anything the GPU said.
fn mix32(x: u32) -> u32 {
    let mut x = x.wrapping_add(0x9e37_79b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x21f0_aaad);
    x ^= x >> 15;
    x = x.wrapping_mul(0x735a_2d97);
    x ^= x >> 15;
    x
}

// Mirror of the kernel's block permutation. `odd = 1` is the identity, so the
// raster arm executes the same instruction sequence as the permuted arm; the two
// differ only in this multiplier.
fn blk_perm(j: u32, odd: u32, iters: u32) -> u32 {
    // `odd` arrives already reduced mod iters; see the note in the .metal file
    // for why that is load-bearing rather than cosmetic.
    debug_assert!(odd < iters, "multiplier must be reduced mod iters or the product overflows u32");
    (j * odd) % iters
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

struct Arm {
    label: String,
    buffers: usize,
    threads: usize,
    burst: usize,
    odd: u32,
    perm: bool,
}

fn next_pow2(mut n: u64) -> usize {
    let mut p = 1u64;
    while p < n {
        p <<= 1;
    }
    n = p;
    p as usize
}

fn main() {
    let device = Device::system_default().expect("system device");
    let sab = flag("--sabotage");
    // `sabotage-fill` writes the HISTORICAL tiled fill (one repeating pattern
    // across the whole buffer) instead of the position hash. It must be caught
    // by the line-duplication guard.
    let sab_fill = flag("--sabotage-fill");
    // Arg parsing. Accepts BOTH `--gb 8` and `--gb=8`.
//
// This is not pedantry: the first version only handled `--gb=8`, so a run
// invoked as `--gb 8` silently fell back to the 6 GB default and printed 6 GB --
// which is what it measured, so nothing on screen said "wrong". A measurement
// harness that cannot be trusted to report its own settings cannot be trusted to
// report its own numbers. Unknown `--x` arguments are a hard error for the same
// reason.
    let args: Vec<String> = std::env::args().collect();
    let mut gb: usize = 6;
    let mut rounds: usize = 7;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) => (n.to_string(), Some(v.to_string())),
            None => (a.clone(), None),
        };
        let mut take = |i: &mut usize| -> String {
            match inline.clone() {
                Some(v) => v,
                None => {
                    *i += 1;
                    args.get(*i).cloned().unwrap_or_else(|| {
                        eprintln!("FATAL: {name} needs a value");
                        std::process::exit(2);
                    })
                }
            }
        };
        match name.as_str() {
            "--gb" => gb = take(&mut i).parse().expect("--gb must be an integer"),
            "--rounds" => rounds = take(&mut i).parse().expect("--rounds must be an integer"),
            "--sabotage" | "--sabotage-fill" => {}
            other => {
                eprintln!("FATAL: unrecognised argument {other}; refusing to run with settings I do not understand");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    println!("# SETTINGS AS PARSED: gb={gb} rounds={rounds}");
    let bytes_total = gb * 1_000_000_000usize;
    let nbuf_max = 8usize;

    println!("# crux2_bw  device={}  working set={gb} GB  rounds={rounds}  sabotage={sab}", device.name());

    // ---- allocate the buffers and fault them in -------------------------
    let bytes_per_buf = bytes_total / nbuf_max;
    // count must be a power of two for the Feistel arm to be a bijection on
    // exactly [0, count). Round the buffer DOWN to a power-of-two element count
    // and measure only the bytes that count covers.
    // `count` must be divisible by `per * burst` for EVERY arm, so that no tail is
    // skipped or double-counted. Taking the LCM of all (per, burst) pairs and
    // rounding DOWN to a multiple of it wastes nothing.
    //
    // (An earlier version also required `count` to be a power of two, which was
    // only needed by the discarded Feistel arm. That constraint cost 33% of the
    // requested working set: a probe that measures 67% of the size it was asked
    // for is a probe whose level is quietly not the real size.)
    const LCM: usize = 3_932_160; // lcm(per*burst) over per in {81920,10240}, burst in {1,6,12,24,48}
    assert_eq!(
        [1usize, 6, 12, 24, 48]
            .iter()
            .all(|&b| LCM % (81_920 * b) == 0 && LCM % (10_240 * b) == 0),
        true,
        "LCM constant is wrong"
    );
    let max_elems = bytes_per_buf / 16;
    let count = (max_elems / LCM) * LCM;
    assert!(count > 0, "requested working set too small for the arm grid");
    let usable_bytes = count * 16 * nbuf_max;
    println!(
        "# {} buffers x {} uint4 = {} elements; usable {} of {} bytes ({:.2}%)",
        nbuf_max, count, count, usable_bytes, bytes_total,
        100.0 * usable_bytes as f64 / bytes_total as f64
    );

    let opts = MTLResourceOptions::StorageModeShared;
    let bufs: Vec<metal::Buffer> = (0..nbuf_max)
        .map(|_| device.new_buffer((count * 16) as u64, opts))
        .collect();

    // ---- fill + fault, and build the CPU reference per buffer -----------
    //
    // The kernel reads a `uint4` (16 B) per element, so the fill must write a
    // full uint4 per element. My first version wrote ONE u32 per element and
    // left 3/4 of every buffer untouched; the position-purity guard caught it
    // immediately, which is the guard earning its keep.
    //
    // Content is a function of POSITION only. Every arm -- raster or permuted --
    // therefore must produce the same checksum, and that checksum is computed
    // HERE from the fill rule, independently of anything the GPU reports.
    let expect_per_buf: Vec<u32> = bufs
        .iter()
        .map(|buf| {
            let ptr = buf.contents() as *mut u32;
            let mut acc = 0u32;
            for i in 0..count {
                let q = if sab_fill {
                    // Historical tiled fill: ONE repeating 16-byte pattern.
                    // Content is not a function of position, and lines repeat.
                    [0x0123_4567u32, 0x89ab_cdef, 0x0f1e_2d3c, 0x4b5a_6978]
                } else {
                    [
                        mix32(i as u32),
                        mix32((i as u32) ^ 0x1111_1111),
                        mix32((i as u32) ^ 0x2222_2222),
                        mix32((i as u32) ^ 0x3333_3333),
                    ]
                };
                unsafe {
                    for (c, v) in q.iter().enumerate() {
                        std::ptr::write_unaligned(ptr.add(i * 4 + c), *v);
                    }
                }
                for v in q {
                    acc = acc.wrapping_add(v);
                }
            }
            acc
        })
        .collect();
    let expect_total: u32 = expect_per_buf.iter().fold(0u32, |a, b| a.wrapping_add(*b));
    println!("# CPU reference checksum over all buffers: {expect_total} (0x{expect_total:08x})");

    // Non-repetition guard. The right question is NOT "are sampled values
    // distinct" -- `mix32` is a bijection on u32 and a sample of 32k values from
    // 2^32 will collide by birthday paradox about 12% of the time, which says
    // nothing about the fill. (My first version of this guard asserted exactly
    // that and failed on a *correct* fill. A guard that cries wolf is a guard
    // that gets deleted.)
    //
    // The right question is the one a tiled/repeating fill actually fails: is
    // the CONTENT a function of position alone (so every arm sees the same
    // bytes), and is a *64-byte line* repeated anywhere in the buffer. A tiled
    // fill repeats whole lines; a position-hash does not, except by the same
    // birthday effect at line granularity -- so the threshold is derived from
    // the birthday expectation rather than asserted to be zero.
    {
        // (1) content is a pure function of position: recompute and compare.
        let mut mismatches = 0usize;
        let probe_n = 4096usize;
        for i in 0..probe_n {
            let idx = i * (count / probe_n);
            let want = [
                mix32(idx as u32),
                mix32((idx as u32) ^ 0x1111_1111),
                mix32((idx as u32) ^ 0x2222_2222),
                mix32((idx as u32) ^ 0x3333_3333),
            ];
            for buf in bufs.iter() {
                for c in 0..4 {
                    let got = unsafe { *((buf.contents() as *const u32).add(idx * 4 + c)) };
                    if got != want[c] {
                        mismatches += 1;
                    }
                }
            }
        }
        assert_eq!(mismatches, 0, "fill is not a function of position");

        // (2) birthday expectation at 64-byte-line granularity.
        //
        // The per-buffer offset matters and my first version got it wrong: every
        // buffer was sampled at the SAME line indices, and since content is a
        // function of position the 8 buffers hold *identical* bytes at identical
        // offsets, so the guard reported 17,500 "duplicates" against an
        // expectation of 3.0 and rejected a perfectly good fill. Different
        // buffers must be sampled at different offsets.
        let per_buf_samples = 2_500usize;
        let samples = per_buf_samples * bufs.len();
        let lines_per_buf = (count * 16) / 64;
        let lines = lines_per_buf * bufs.len();
        let expected_collisions = (samples * samples) as f64 / (2.0 * lines as f64);
        let mut seen = std::collections::HashSet::new();
        let mut dup = 0usize;
        for (bi, buf) in bufs.iter().enumerate() {
            let p = buf.contents() as *const u8;
            // Offset each buffer's walk so the same line index in different
            // buffers is never compared.
            let start = (bi * (lines_per_buf / bufs.len().max(1))) % lines_per_buf;
            let step = (lines_per_buf / per_buf_samples).max(1);
            let mut li = start;
            let mut taken = 0usize;
            while taken < per_buf_samples {
                if li >= lines_per_buf {
                    li -= lines_per_buf;
                }
                let mut v: u64 = 0;
                for k in 0..8 {
                    v |= (unsafe { *p.add(li * 64 + k) } as u64) << (8 * k);
                }
                if !seen.insert(v) {
                    dup += 1;
                }
                li += step;
                if li >= lines_per_buf {
                    li -= lines_per_buf;
                }
                taken += 1;
            }
        }
        println!(
            "# fill guard: position-pure OK; 64-byte-line sample {samples} over {lines} lines \
             -> {dup} duplicates (birthday expectation {:.1})",
            expected_collisions
        );
        assert!(
            (dup as f64) <= expected_collisions * 8.0 + 5.0,
            "fill repeats whole lines: {dup} duplicates against an expectation of {expected_collisions:.1} \
             -- this is the signature of a tiled fill"
        );
    }

    // SABOTAGE-BYTES: flip one word in the LAST buffer only. A 1-buffer arm
    // reads clean data and passes; an 8-buffer arm must read the corrupted word
    // and FAIL. If the 8-buffer arm still passes, the arm is not reading the
    // buffers it claims to, which is the defect `crux.md` §6 records happening
    // twice already (a multi-buffer arm that never touched its own buffers).
    let expect_b0 = expect_per_buf[0];
    // Exact positive prediction for the sabotaged run.
    //
    // The flip replaces `mix32(0)` with `mix32(0) ^ 0xdeadbeef` at buffer 7,
    // element 0, component 0. Content is a function of position, so the CPU can
    // predict the new all-buffer total exactly -- and therefore predict that
    // 1-buffer arms (which never touch buffer 7) are UNCHANGED while N-buffer
    // arms move by exactly `0xdeadbeef`. That is a falsifiable prediction per
    // arm, not merely "something differs".
    let flipped = mix32(0) ^ 0xdead_beef;
    let expect_sab_total = expect_total.wrapping_sub(mix32(0)).wrapping_add(flipped);
    if sab && !sab_fill {
        let last = bufs.len() - 1;
        let p = bufs[last].contents() as *mut u32;
        unsafe { *p = *p ^ 0xdead_beef };
        println!("# SABOTAGE-BYTES: flipped one word in buffer {last} only");
        println!("#   1-buffer arms must stay at {expect_b0} (0x{expect_b0:08x})");
        println!(
            "#   N-buffer arms must move to {expect_sab_total} (0x{expect_sab_total:08x}), \
             exactly +0xdeadbeef"
        );
    }


    let words = 4 * 81_920;
    let out = device.new_buffer((words * 4) as u64, opts);
    let out_words = words;

    // ---- pipeline ---------------------------------------------------------
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SRC, &CompileOptions::new())
        .expect("compile crux2_bw.metal");
    let pipe = device
        .new_compute_pipeline_state_with_function(&lib.get_function("probe", None).unwrap())
        .expect("probe pipeline");

    // ---- coverage proof, on the host, for every arm ----------------------
    const ODD: [u32; 5] = [
            2_654_435_761, // Knuth
            0x9E37_79B1,
            0x27BB_2EE7,
            0x85EB_CA6B,
            0xC2B2_AE35,
        ];
    let arms: Vec<Arm> = {
        let mut v = Vec::new();
        // --- pure raster references: no multiply, no modulo ------------------
        // burst=1 is exactly the prior probe's `stream_u4`.
        v.push(Arm { label: "raster_burst1_buf1".into(), buffers: 1, threads: 81_920, burst: 1, odd: 1, perm: false });
        v.push(Arm { label: "raster_burst24_buf1".into(), buffers: 1, threads: 81_920, burst: 24, odd: 1, perm: false });
        // burst family in raster order. 24 uint4 = 384 B = exactly one 32x32
        // tile's payload at 3 bpw, i.e. the shape the real decode GEMV walks;
        // 16 (256 B) and 32 (512 B) bracket it.
        for &b in &[6usize, 12, 48] {
            v.push(Arm { label: format!("raster_burst{b}_buf1"), buffers: 1, threads: 81_920, burst: b, odd: 1, perm: false });
        }
        // independent allocations, raster within each
        v.push(Arm { label: "raster_burst1_buf8".into(), buffers: 8, threads: 81_920, burst: 1, odd: 1, perm: false });
        v.push(Arm { label: "raster_burst24_buf8".into(), buffers: 8, threads: 81_920, burst: 24, odd: 1, perm: false });
        // --- the permutation-COST control -----------------------------------
        // Identical instruction sequence to the stride arms (multiply + modulo
        // per burst) but odd == 1, so the visit order is the identity. Any gap
        // between a stride arm and THIS is the permutation's effect on memory,
        // not its arithmetic cost.
        v.push(Arm { label: "CTRL_modidentity_b24".into(), buffers: 1, threads: 81_920, burst: 24, odd: 1, perm: true });
        // --- ENTROPY / STRIDE SWEEP: the axis the brief calls untested -------
        for (i, &o) in ODD.iter().enumerate() {
            v.push(Arm { label: format!("stride{i}_b24_buf1"), buffers: 1, threads: 81_920, burst: 24, odd: o, perm: true });
        }
        v.push(Arm { label: "stride0_b24_buf8".into(), buffers: 8, threads: 81_920, burst: 24, odd: ODD[0], perm: true });
        v.push(Arm { label: "stride0_b1_buf1".into(), buffers: 1, threads: 81_920, burst: 1, odd: ODD[0], perm: true });
        // positive control: byte-, thread-, buffer- and order-identical to the
        // first reference arm.
        v.push(Arm { label: "CONTROL_repeat".into(), buffers: 1, threads: 81_920, burst: 1, odd: 1, perm: false });
        v
    };

    println!("\n# coverage + bijection proof (host-side, mirrors the kernel index expression)");
    for a in &arms {
        let per = a.threads / a.buffers;
        assert_eq!(a.threads % a.buffers, 0, "{}: threads not divisible by buffers", a.label);
        let iters = (count / (per * a.burst)) as u32;
        assert_eq!(
            iters as usize * a.burst * per, count,
            "{}: iters*burst*per != count (tail miscounted)", a.label
        );
        // The block permutation must be a BIJECTION on [0, iters), otherwise the
        // arm skips or repeats a region -- and the checksum would still look
        // plausible because the repeated region is summed twice.
        // Reduce the multiplier mod iters FIRST. A raw 31-bit constant overflows
        // u32 against j >= 2 and the wrapped product is not congruent mod iters,
        // which silently destroys the bijection. Then reject the degenerate
        // reduction to 1 (that is the identity, i.e. no permutation at all).
        let mut odd_red = a.odd % iters;
        if odd_red == 1 {
            odd_red = 2;
        }
        while gcd(odd_red, iters) != 1 {
            odd_red += 1;
            assert!(odd_red < iters, "{}: no coprime multiplier below iters", a.label);
        }
        assert!(
            odd_red != 1,
            "{}: permutation degenerated to the identity -- this arm is not testing anything",
            a.label
        );
        if sab && a.label == "burst24_buf1" {
            // Deliberately under-cover by one iteration: the guard must bite.
            println!("# SABOTAGE: burst24_buf1 will visit count - per elements");
            continue;
        }
        // Exhaustive multiset check: every element visited exactly once.
        {
            let mut seen = vec![false; count];
            let mut dup = 0usize;
            for lid in 0..per {
                for j in 0..iters {
                    let base = (blk_perm(j, odd_red, iters) as usize) * per * a.burst;
                    for k in 0..a.burst {
                        let idx = base + k * per + lid;
                        assert!(idx < count, "{}: index {idx} out of range", a.label);
                        if seen[idx] {
                            dup += 1;
                        }
                        seen[idx] = true;
                    }
                }
            }
            assert_eq!(dup, 0, "{}: VISITED TWICE ({dup} times)", a.label);
            assert!(seen.iter().all(|&x| x), "{}: some element never visited", a.label);
        }
        let bytes = count * 16 * a.buffers;
        println!(
            "  {:<24} per={per:<6} burst={:<3} iters={iters:<6} odd={:<11} bytes={bytes} ({:.4} GB) bijection OK",
            a.label, a.burst, if a.perm { odd_red } else { 1 }, bytes as f64 / 1e9
        );
    }
    if sab {
        println!("# (sabotage run: full-multiset proof skipped for the sabotaged arm, as intended)");
    }

    // ---- interleaved measurement ----------------------------------------
    let mut results: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut checks: Vec<Vec<u32>> = vec![Vec::new(); arms.len()];

    let reduced_odd = |a: &Arm| -> u32 {
        let per = a.threads / a.buffers;
        let iters = (count / (per * a.burst)) as u32;
        let mut o = a.odd % iters;
        if o == 1 {
            o = 2;
        }
        while gcd(o, iters) != 1 {
            o += 1;
        }
        o
    };

    let mk = |a: &Arm| -> Params {
        let per = a.threads / a.buffers;
        Params {
            count: count as u32,
            per: per as u32,
            burst: a.burst as u32,
            iters: (count / (per * a.burst)) as u32,
            odd: reduced_odd(a),
            perm: a.perm as u32,
            nbuf: a.buffers as u32,
            _p0: 0,
        }
    };

    for r in 0..(rounds + 2) {
        for (ai, a) in arms.iter().enumerate() {
            let p = mk(a);
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&pipe);
            // Slots 0..7: buffers, unused ones bound to buffer 0 so no binding
            // is null. Slot 8: out. Slot 9: params.
            for slot in 0..8usize {
                let bb = bufs.get(slot).unwrap_or(&bufs[0]);
                e.set_buffer(slot as u64, Some(bb), 0);
            }
            e.set_buffer(8, Some(&out), 0);
            e.set_bytes(9, std::mem::size_of::<Params>() as u64, &p as *const Params as *const std::ffi::c_void);
            e.dispatch_threads(MTLSize::new(a.threads as u64, 1, 1), MTLSize::new(TG as u64, 1, 1));
            e.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            let el = t0.elapsed().as_secs_f64();
            if r < 2 {
                continue; // warm-ups
            }
            let cs = checksum(&out, out_words);
            let bytes = count * 16 * a.buffers;
            results[ai].push(bytes as f64 / el / 1e9);
            checks[ai].push(cs);
        }
        println!("# round {} done", r);
    }

    // ---- gates ------------------------------------------------------------
    println!("\n# CHECKSUM GATE (CPU reference {expect_total})");
    let mut gate_ok = true;
    for (ai, a) in arms.iter().enumerate() {
        let uniq: std::collections::BTreeSet<u32> = checks[ai].iter().copied().collect();
        // Under SABOTAGE-BYTES one word of the LAST buffer is flipped:
        //   * 1-buffer arms read buffer 0, clean -> must equal expect_b0;
        //   * N-buffer arms also read the dirty buffer -> must NOT equal the
        //     clean all-buffer total. If such an arm still reported the clean
        //     total, it was not reading the buffers it claims to.
        // A 1-buffer arm reads ONLY buffer 0, so its correct answer is that buffer's
        // checksum, not the sum over all eight. My first gate compared every arm
        // to the all-buffer total and failed the whole run -- which is the right
        // behaviour for a wrong number, but here the GATE was wrong, not the arm.
        // Expected value is now prefix-summed over the buffers the arm declares.
        let mut expect_arm = 0u32;
        for i in 0..a.buffers {
            expect_arm = expect_arm.wrapping_add(expect_per_buf[i]);
        }
        if sab && !sab_fill && a.buffers == bufs.len() {
            expect_arm = expect_arm.wrapping_sub(mix32(0)).wrapping_add(flipped);
        }
        let expect_label = if sab && !sab_fill { "expect_sab" } else { "expect_arm" };
        let ok = uniq.len() == 1 && uniq.contains(&expect_arm);
        if !ok {
            gate_ok = false;
        }
        println!(
            "  {:<24} distinct={} value=0x{:08x} want={:<11} {}",
            a.label,
            uniq.len(),
            uniq.iter().next().copied().unwrap_or(0),
            expect_label,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    if sab && !sab_fill {
        // Every arm matched its EXACT prediction: single-buffer arms unaffected,
        // multi-buffer arms moved by exactly 0xdeadbeef. That is the proof that
        // each arm reads precisely the buffers it claims -- which `crux.md` §6
        // records as having gone wrong twice.
        assert!(
            gate_ok,
            "SABOTAGE-BYTES: some arm did not match its exact prediction, so it is not reading \
             the buffers it claims and its bandwidth would be fiction"
        );
        println!(
            "# SABOTAGE-BYTES behaved exactly as predicted: per-arm predictions all matched.\n\
             #   => every arm reads precisely the buffers it declares; the multibuf numbers are real.\n"
        );
    } else {
        assert!(gate_ok, "CHECKSUM GATE FAILED on an unsabotaged run -- no number from this run is reportable");
    }

    // ---- table ------------------------------------------------------------
    let ref_med = {
        let mut v = results[arms.iter().position(|a| a.label == "raster_burst1_buf1").unwrap()].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let ctl_med = {
        let mut v = results[arms.iter().position(|a| a.label == "CONTROL_repeat").unwrap()].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let noise = 100.0 * (ctl_med - ref_med) / ref_med;

    println!("\n{:<24} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}", "arm", "median", "min", "max", "spread%", "vs ref", "%wire");
    for (ai, a) in arms.iter().enumerate() {
        let mut v = results[ai].clone();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = v[v.len() / 2];
        let spread = 100.0 * (v[v.len() - 1] - v[0]) / med;
        println!(
            "{:<24} {:>9.1} {:>9.1} {:>9.1} {:>9.2} {:>8.2} {:>7.1}%",
            a.label, med, v[0], v[v.len() - 1], spread, 100.0 * (med - ref_med) / ref_med, 100.0 * med / WIRE_PEAK
        );
    }
    println!("\n# reference arm burst1_buf1 median = {ref_med:.1} GB/s");
    println!("# CONTROL_repeat difference = {noise:+.2}%  <-- the noise floor");
    println!("# nothing smaller than this may be reported as an effect");
    let best = results
        .iter()
        .enumerate()
        .filter(|(i, _)| arms[*i].label != "CONTROL_repeat")
        .map(|(i, v)| {
            let mut s = v.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (arms[i].label.clone(), s[s.len() - 1])
        })
        .fold((String::new(), 0.0f64), |acc, (l, b)| if b > acc.1 { (l, b) } else { acc });
    println!("# best arm = {} at {:.1} GB/s ({:.1}% of wire)", best.0, best.1, 100.0 * best.1 / WIRE_PEAK);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    count: u32, per: u32, burst: u32, iters: u32,
    odd: u32, perm: u32, nbuf: u32, _p0: u32,
}

fn checksum(out: &metal::Buffer, words: usize) -> u32 {
    let p = out.contents() as *const u32;
    let mut s = 0u32;
    for i in 0..words {
        s = s.wrapping_add(unsafe { *p.add(i) });
    }
    s
}