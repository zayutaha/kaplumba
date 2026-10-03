//! PROBE (measurement only, not a regression gate): how much anonymous memory
//! this box will actually give a process before macOS starts compressing and
//! paging it out.
//!
//! ## Why
//!
//! Every capacity claim about 4 bpw on this box turns on one number that nobody
//! had measured: the point at which `mmap` + touch stops producing free pages
//! and starts producing compressor pages. `AGENTS.md` §10.3 records two
//! measurement mistakes made here already (`vm_stat`'s `Pages free` read as
//! pressure on a healthy box; `top`'s unit parsed as GiB when it was MiB), and
//! `memory_proof.rs` re-derives supply as
//! free + inactive + purgeable + speculative for exactly that reason.
//!
//! This measures the threshold directly, on the real kernel, with the real
//! desktop load, by asking for pages and watching which column pays.
//!
//! ## What it asserts
//!
//! Nothing. It prints a table and fails only if it cannot allocate at all.
//! A regression gate on a *supply* figure would be a gate on the desktop's
//! memory use, which is not this repository's to assert.
//!
//! ## What would make these numbers wrong
//!
//! * The page size and the reclaimable set are read live, not assumed, so a
//!   different macOS would show up in the printed `page_size` / `page_total`
//!   rather than as a silently wrong total.
//! * The threshold depends on what else is resident. This must be run on a box
//!   in the same state as the runs it is being compared against, and it says so
//!   in its output (`baseline_free_GiB`) so a mismatched pair is visible.
//! * It walks in 128 MiB steps, so the reported threshold is accurate to
//!   ±128 MiB, not exact. Halve `STEP` to tighten it.

use std::time::Instant;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct VmStat64 {
    free_count: u64,
    active_count: u64,
    inactive_count: u64,
    wire_count: u64,
    compressor_page_count: u64,
    purgeable_count: u64,
    speculative_count: u64,
    page_size: u64,
    pages_total: u64,
}

fn host_vm() -> VmStat64 {
    // SAFETY: zeroed, then filled by `host_statistics64`, which writes exactly
    // `HOST_VM_INFO64_COUNT` integers. Shaped as `memory_proof.rs:130-146`
    // shapes it, which the memory gate exercises.
    #[allow(deprecated)]
    unsafe {
        let mut count = libc::HOST_VM_INFO64_COUNT;
        assert_eq!(
            count as usize,
            std::mem::size_of::<VmStat64>() / std::mem::size_of::<u64>(),
            "VmStat64 no longer matches HOST_VM_INFO64_COUNT; re-check the field order"
        );
        let mut out = VmStat64::default();
        let host = libc::mach_host_self();
        let kr = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut out as *mut VmStat64 as libc::host_info64_t,
            &mut count,
        );
        assert_eq!(kr, libc::KERN_SUCCESS, "host_statistics64 failed: {kr}");
        out
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

/// 128 MiB. See the module doc for what the granularity costs.
const STEP: u64 = 128 * 1_048_576;
/// Stop once the compressor is clearly engaged, which is the answer. Well past
/// any figure this box could plausibly hand out.
const CEILING: u64 = 22 * 1_073_741_824;

/// Anonymous memory, one `mmap` region, with every page touched.
///
/// Touched one byte per 4 KiB of *virtual* space on purpose: this is what the
/// engine does when it fills its arenas, and it is what forces the kernel to
/// supply real pages rather than reservations.
fn touch(bytes: u64) -> Vec<*mut u8> {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    assert_ne!(ptr, libc::MAP_FAILED, "mmap failed at {bytes} bytes");
    let stride = 4096usize;
    let base = ptr as *mut u8;
    let mut touched = 0usize;
    let mut i = 0usize;
    while (i as u64) < bytes {
        unsafe { base.add(i).write_volatile(0) };
        touched += 1;
        i += stride;
    }
    assert_eq!(touched as u64 * stride as u64, bytes);
    // Keep the mapping alive for the caller's lifetime by leaking the pointer.
    vec![base]
}

#[test]
#[ignore = "measures this box's memory supply; run alone, with no model loaded"]
fn probe_supply_threshold() {
    let base = host_vm();
    println!("BUDGET page_size            = {}", base.page_size);
    println!("BUDGET page_total           = {}", base.pages_total);
    println!(
        "BUDGET total_GiB            = {:.3}",
        gib(base.page_size * base.pages_total)
    );
    println!("BUDGET baseline_free_GiB    = {:.3}", gib(base.free_count * base.page_size));
    println!(
        "BUDGET baseline_free+spec_GiB = {:.3}",
        gib((base.free_count + base.speculative_count) * base.page_size)
    );
    println!(
        "BUDGET baseline_compressor_GiB = {:.3}",
        gib(base.compressor_page_count * base.page_size)
    );
    println!(
        "BUDGET baseline_inactive_GiB   = {:.3}",
        gib(base.inactive_count * base.page_size)
    );
    println!(
        "BUDGET baseline_purgeable_GiB  = {:.3}",
        gib(base.purgeable_count * base.page_size)
    );

    let start_swap = swap_used_bytes();
    println!("BUDGET baseline_swap_GiB    = {:.3}", gib(start_swap));
    println!("BUDGET  step_GiB  free_GiB  spec_GiB  compr_GiB  wire_GiB  swap_GiB  ms");

    let mut held: Vec<Vec<*mut u8>> = Vec::new();
    let mut allocated: u64 = 0;
    let mut first_compressed_at: Option<u64> = None;
    let mut first_swapped_at: Option<u64> = None;
    while allocated < CEILING {
        let t = Instant::now();
        held.push(touch(STEP));
        allocated += STEP;
        let now = host_vm();
        let swap = swap_used_bytes();
        let free = gib(now.free_count * now.page_size);
        let spec = gib(now.speculative_count * now.page_size);
        let compr = gib(now.compressor_page_count * now.page_size);
        let wire = gib(now.wire_count * now.page_size);
        if compr > gib(base.compressor_page_count * base.page_size) + 0.25 && first_compressed_at.is_none()
        {
            first_compressed_at = Some(allocated);
        }
        if swap > start_swap + 64 * 1_048_576 && first_swapped_at.is_none() {
            first_swapped_at = Some(allocated);
        }
        println!(
            "BUDGET {:>8.2} {:>9.3} {:>9.3} {:>10.3} {:>9.3} {:>10.3} {:>6.0}",
            gib(allocated),
            free,
            spec,
            compr,
            wire,
            gib(swap),
            t.elapsed().as_secs_f64() * 1000.0
        );
        // Once the compressor is well engaged and swap is moving, more steps
        // measure the thrash regime rather than the supply. That regime is
        // measured by the decode runs, not here.
        if compr > 4.0 {
            println!("BUDGET stopping: compressor is already {:.2} GiB", compr);
            break;
        }
    }
    println!(
        "BUDGET ANSWER first_compression_above_baseline_at = {}",
        first_compressed_at.map_or("never".to_string(), gib_str)
    );
    println!(
        "BUDGET ANSWER first_swap_growth_at = {}",
        first_swapped_at.map_or("never".to_string(), gib_str)
    );
    println!("BUDGET ANSWER total_touched = {}", gib(allocated));
    // Drop nothing: the point is the peak, and `held` goes out of scope here.
    drop(held);
}

fn gib_str(bytes: u64) -> String {
    format!("{:.3} GiB", gib(bytes))
}

/// `vm.swapusage`, parsed as bytes.
///
/// `sysctl -n vm.swapusage` is the only swap figure that exists here and it is
/// the one AGENTS.md §10.3 points at. Parsed by hand rather than by stripping
/// non-digits, because that stripping is how `gpuq` once reported 554 GiB
/// instead of 554 MiB.
fn swap_used_bytes() -> u64 {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.swapusage"])
        .output()
        .expect("sysctl");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut used = 0u64;
    for field in text.split_whitespace() {
        if let Some(rest) = field.strip_prefix("used") {
            let digits: String = rest.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
            let value: f64 = digits.parse().unwrap_or(0.0);
            let mib = if text.contains("M") { 1.0 } else { 1024.0 };
            used = (value * mib * 1_048_576.0) as u64;
        }
    }
    used
}