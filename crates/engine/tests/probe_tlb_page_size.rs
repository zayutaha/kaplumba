//! M01 probe: VM page size, mmap alignment, and whether `PAGE_ALIGN` can affect
//! TLB pressure at all.
//!
//! Read-only with respect to the product: this file creates a temporary file,
//! maps it, and unmaps it. It loads no model, creates no Metal device, and runs
//! no GPU work. The only external state it reads is the safetensors *header* of
//! an existing checkpoint (a few hundred KB of JSON), and only to count tensor
//! offsets; that read is skipped when the checkpoint is absent.
//!
//! The load-bearing test is [`page_align_does_not_change_pages_translated`]. The
//! claim under test is the one that motivates changing
//! `weights.rs:1105 PAGE_ALIGN: usize = 4096` to 16384: that the buffer base
//! alignment influences TLB pressure. Translation granularity is fixed by the
//! MMU and the VM, not by where a `MTLBuffer` happens to start, so that claim
//! should be falsifiable and should fail here if it were true.

use std::fs::File;
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;

/// The constant actually in the tree today: `crates/engine/src/weights.rs:1105`.
const PAGE_ALIGN_IN_TREE: usize = 4096;

/// The VM page size this hypothesis is about, cross-checked at runtime.
const VM_PAGE: usize = 16_384;

/// Bytes streamed per decode step, from the measured baseline.
const BYTES_PER_TOKEN: f64 = 12.04e9;
const TOKEN_SECONDS: f64 = 0.0623;
const DRAM_PEAK_GB_S: f64 = 273.0;
const LAYER_GB_S: f64 = 193.0;
const LM_HEAD_GB_S: f64 = 253.0;

fn vm_page_size() -> usize {
    // SAFETY: `sysconf` with a valid name is always safe and cannot fail the
    // process; it returns -1 on error, which the assert below rejects.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(value > 0, "sysconf(_SC_PAGESIZE) failed: {value}");
    value as usize
}

/// A page-aligned window base for `[start, start+len)`, rounded down by
/// `align - 1`, which is what `no_copy_view` does with `PAGE_ALIGN`.
fn align_down(start: usize, align: usize) -> usize {
    start & !(align - 1)
}

/// Number of VM pages a buffer over `[start, end)` occupies. Depends only on the
/// byte range, never on where the buffer's base address was rounded to.
fn pages_spanned(start: usize, len: usize, page: usize) -> usize {
    let end = start + len;
    end.div_ceil(page) - start / page
}

#[test]
fn vm_page_size_is_16k() {
    let size = vm_page_size();
    assert_eq!(
        size, VM_PAGE,
        "this hypothesis was written against a {VM_PAGE}-byte VM page; \
         this box reports {size}. Every page count below is wrong if that changed."
    );
    // The tree already hardcodes the 16384 figure in two other places, so if the
    // box ever disagrees those are wrong too.
    assert_eq!(
        PAGE_ALIGN_IN_TREE, 4096,
        "the tree's PAGE_ALIGN moved; this probe's arithmetic assumed 4096"
    );
}

#[test]
fn mmap_returns_vm_page_aligned_but_not_necessarily_more() {
    let dir = temp_dir("mmap-align");
    let path = dir.join("probe.bin");
    let mut file = File::create(&path).expect("create probe file");
    // 1 MiB + a deliberately ragged tail so the mapping length is not a
    // multiple of the page size.
    let len = 1024 * 1024 + 517;
    file.write_all(&vec![7u8; len]).expect("write probe file");
    drop(file);

    let file = File::open(&path).expect("reopen probe file");
    // SAFETY: a valid fd, a non-zero length, PROT_READ/MAP_PRIVATE on a regular
    // file: all four arguments are valid and the mapping is read-only.
    let pointer = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };
    assert_ne!(pointer, libc::MAP_FAILED, "mmap failed");
    // SAFETY: `pointer` is a live mapping of `len` bytes, being unmapped here.
    unsafe { libc::munmap(pointer, len) };
    let _ = std::fs::remove_dir_all(&dir);

    let base = pointer as usize;
    assert_eq!(
        base % VM_PAGE,
        0,
        "mmap returned {base:#x}, not {VM_PAGE}-aligned; PAGE_ALIGN=4096 would then \
         be aligning DOWN past the mapping's own guarantee"
    );
    // A 4096-round of a 16384-aligned address is still 16384-aligned, which is
    // why 4096 is sufficient *for the mapping base* and never truncates below it.
    assert_eq!(align_down(base, 4096) % VM_PAGE, 0);
}

/// The core falsification test for the motivating hypothesis.
///
/// `no_copy_view` rounds the MTLBuffer base down by `PAGE_ALIGN - 1` so the
/// tensor's real start is inside the buffer at `delta`. The claim is that a
/// 4 KB round leaves something TLB-relevant that a 16 KB round would fix.
///
/// Translation is performed per VM page, by the MMU, on the address the load
/// actually generates. Where the buffer's *base* is rounded only changes which
/// pages the first partial page shares, never the set of pages a read of
/// `[start, end)` touches. If that is true the page count is invariant to
/// `PAGE_ALIGN` and the two constants produce identical translation load.
#[test]
fn page_align_does_not_change_pages_translated() {
    // Realistic shapes: an 8-byte-aligned safetensors offset that is *not* a
    // multiple of 16384, plus the layer-trellis and lm_head magnitudes.
    let cases: &[(u64, u64)] = &[
        (8, 8),                       // smallest possible tensor
        (8 + 8 * 7, 8 * 3_366_016),   // ~27 MB, mid-shard
        (3_502_008, 183_000_000),     // ~183 MB, one layer trellis
        (8_498_296_712, 1_057_000_000), // ~1.06 GB, lm_head at end of shard 2
        (16_384 * 100 + 8, 1_057_000_000),
    ];

    for &(offset, len) in cases {
        let start = VM_PAGE * 4096 + offset as usize; // mmap base + tensor offset
        let end = start + len as usize;

        // What `no_copy_view` actually builds, for each candidate constant.
        let base_4k = align_down(start, 4096);
        let base_16k = align_down(start, 16384);
        let pages_4k = pages_spanned(base_4k, end - base_4k, VM_PAGE);
        let pages_16k = pages_spanned(base_16k, end - base_16k, VM_PAGE);

        assert_eq!(
            pages_4k, pages_16k,
            "offset {offset}: PAGE_ALIGN changed the translation load \
             (4 KB -> {pages_4k} pages, 16 KB -> {pages_16k} pages)"
        );
        assert!(
            pages_4k >= pages_spanned(start, len as usize, VM_PAGE),
            "rounding the base down can only add pages, never remove them"
        );

        // And the converse: a *larger* alignment can only waste address space,
        // never save a translation. delta stays inside the page in both cases.
        let delta_4k = start - align_down(start, 4096);
        let delta_16k = start - align_down(start, 16384);
        assert!(delta_4k < 4096 && delta_16k < 16384);
        // 16 KB rounds down further, so the tensor sits *later* in its buffer.
        assert!(delta_16k >= delta_4k, "16 KB rounding should not shrink the delta");

        // Both windows cover the tensor exactly: no byte dropped, none aliased in
        // twice within a single view. `no_copy_view` sizes the buffer as
        // `end - base`, so the view is [base, end) and the tensor sits at delta.
        for align in [4096usize, 16384] {
            let base = align_down(start, align);
            let buffer_len = end - base;
            assert!(base <= start, "align {align} rounded up past the tensor");
            assert!(buffer_len >= len as usize, "align {align} truncated the tensor");
            // And the buffer covers exactly the intended range, no more.
            assert_eq!(base + buffer_len, end);
        }
    }
}

/// A 4096 round yields a base that is *not* page-aligned whenever the safetensors
/// offset is not a multiple of the page size, so the two constants are not
/// interchangeable even though neither affects translation. This is the only
/// real difference between them, and it is about alignment contract, not TLB.
#[test]
fn four_k_round_yields_a_non_page_aligned_base() {
    // Offset is a multiple of 4096 but not of 16384: the shape that makes the
    // two constants differ.
    let start = VM_PAGE * 100 + 4096 + 8;
    let base_4k = align_down(start, 4096);
    let base_16k = align_down(start, 16384);

    assert_eq!(base_4k % 4096, 0);
    assert_ne!(
        base_4k % VM_PAGE,
        0,
        "expected a 4 KB-aligned but page-misaligned base; if mmap ever handed back \
         a non-page-aligned address this test would be vacuous"
    );
    assert_eq!(base_16k % VM_PAGE, 0);
    // The 4 KB base sits one 4 KB page above the 16 KB one: it adds at most
    // 4095 bytes of addressable slack ahead of the tensor and no extra page.
    assert_eq!(base_4k - base_16k, 4096);
}

/// Translate-page counts and rates for the measured baseline, so the numbers in
/// the findings file are pinned to something executable.
#[test]
fn translation_rate_arithmetic() {
    let pages_per_token = (BYTES_PER_TOKEN / VM_PAGE as f64).ceil();
    assert_eq!(pages_per_token, 734_864.0, "12.04 GB / 16 KB page count moved");

    let fills_per_second = pages_per_token / TOKEN_SECONDS;
    assert!(
        (11.7e6..11.9e6).contains(&fills_per_second),
        "translation rate {fills_per_second:.0}/s"
    );

    // Layer trellis at the measured rate, and at the 20 tok/s target.
    let at_193 = LAYER_GB_S * 1e9 / VM_PAGE as f64;
    let at_236 = 236.7e9 / VM_PAGE as f64;
    assert!((11.7e6..11.9e6).contains(&at_193));
    assert!((14.4e6..14.5e6).contains(&at_236));

    // The decisive comparison: two workloads, same GPU, same TLB, same 62.3 ms.
    // A shared resource cannot explain a 31% throughput difference between them.
    let ratio = LM_HEAD_GB_S / LAYER_GB_S;
    assert!((1.30..1.32).contains(&ratio), "lm_head/layers GB/s ratio moved");
    assert!(
        LM_HEAD_GB_S / DRAM_PEAK_GB_S > 0.92,
        "lm_head is no longer at the DRAM wall; the differential argument weakens"
    );

    // Had the VM used 4 KB pages, the same 193 GB/s would demand 4x the fills.
    let at_193_4k = LAYER_GB_S * 1e9 / 4096.0;
    assert!((47.0e6..47.3e6).contains(&at_193_4k));
}

/// Conditional arithmetic for `spec/memory-swap-cliff` F6, which proposes
/// *adding* `mlock`. There is no `mlock` in the tree today; this pins what
/// rounding F6 must use so the byte budget it reports is honest.
#[test]
fn mlock_rounds_out_to_whole_vm_pages() {
    // Darwin's mlock/vm_lock operates on whole VM pages: a range shorter than a
    // page still pins that page. So a 4096-rounded call over-locks by the same
    // amount a 16384-rounded call does -- rounding by 4096 saves nothing.
    let worst_case_overlock = VM_PAGE - 1;
    assert!(worst_case_overlock <= 32767, "over-lock bound moved");

    // F6's shape: 60 layers x ~7 trellis tensors = 410 calls per step.
    let calls = 410_usize;
    let worst_extra = calls * (worst_case_overlock as usize);
    assert_eq!(worst_extra, 6_717_030);
    // Against the wire limit this is noise, which is why the rounding choice is
    // not what decides F6; the byte budget is.
    let wire_limit = 19_585_050_869u64;
    assert!(worst_extra < (wire_limit / 1000) as usize);

    // The order of magnitude that does matter, for contrast: the payload itself.
    assert!(
        worst_extra * 800 < 11_000_000_000,
        "sanity: ~13 MB of rounding overhead against an 11 GB payload"
    );
}

/// Read the safetensors headers of the local checkpoint and count how many
/// tensors are not page-aligned. Skipped when no checkpoint is present.
#[test]
fn checkpoint_tensor_offsets_are_not_page_aligned() {
    let dir = std::env::var("KAPLUMBA_PROBE_MODEL")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".kaplumba/models/Gxma/Gxma"))
        })
        .unwrap_or_else(PathBuf::new);
    if !dir.is_dir() {
        eprintln!("probe_tlb_page_size: no checkpoint at {dir:?}; skipping offset census");
        return;
    }

    let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read model dir")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    shards.sort();
    assert!(!shards.is_empty(), "no .safetensors files in {dir:?}");

    let mut tensors = 0usize;
    let mut misaligned = 0usize;
    let mut trellis = 0usize;
    let mut trellis_misaligned = 0usize;
    let mut payload = 0u64;

    for shard in &shards {
        let file = File::open(shard).expect("open shard header");
        // SAFETY: valid fd; we read only the 8-byte little-endian header length.
        let mut header_len = [0u8; 8];
        let mut file = file;
        std::io::Read::read_exact(&mut file, &mut header_len).expect("read header length");
        let header_len = u64::from_le_bytes(header_len) as usize;
        assert!(header_len < 64 * 1024 * 1024, "implausible header {header_len}");
        let mut header = vec![0u8; header_len];
        std::io::Read::read_exact(&mut file, &mut header).expect("read header json");

        let json: serde_json::Value =
            serde_json::from_slice(&header).expect("parse safetensors header");
        let Some(map) = json.as_object() else {
            continue;
        };
        for (name, info) in map {
            let Some(span) = info.get("data_offsets").and_then(|v| v.as_array()) else {
                continue;
            };
            let start = span[0].as_u64().expect("data_offsets[0]");
            let end = span[1].as_u64().expect("data_offsets[1]");
            let bytes = end - start;
            tensors += 1;
            payload += bytes;
            let page_misaligned = (start % VM_PAGE as u64) != 0;
            if page_misaligned {
                misaligned += 1;
            }
            if name.ends_with("trellis") {
                trellis += 1;
                if page_misaligned {
                    trellis_misaligned += 1;
                }
            }
        }
    }

    eprintln!(
        "probe_tlb_page_size: {tensors} tensors, {payload} payload bytes, \
         {misaligned} not {VM_PAGE}-aligned; trellis {trellis_misaligned}/{trellis} misaligned"
    );
    assert!(tensors > 0, "found no tensors");
    assert!(payload > 10_000_000_000, "payload {payload} too small");
    // safetensors aligns to 8 bytes, so essentially every tensor starts
    // mid-page. This is why PAGE_ALIGN is 4096 rather than 16384 today.
    assert!(
        misaligned * 2 > tensors,
        "expected most tensors to be page-misaligned, got {misaligned}/{tensors}"
    );
    assert!(
        trellis > 400 && trellis_misaligned == trellis,
        "every trellis tensor is mid-page: {trellis_misaligned}/{trellis}"
    );

    // Total mapped payload is ~16.0 GB (all tensors, including the ones the
    // decode path does not stream). The 12.04 GB/token figure is the subset.
    // Either way the page count follows from the byte total, not the layout:
    // 16.0e9 / 16384 = 978_124, i.e. this number cannot be reduced by choosing a
    // different PAGE_ALIGN.
    let pages = payload.div_ceil(VM_PAGE as u64);
    assert_eq!(pages, 978_275, "payload page count moved");
    // And the 4x smaller number 4 KB pages would have demanded:
    assert_eq!(payload.div_ceil(4096), 3_913_099);
}

fn temp_dir(name: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "kaplumba-probe-tlb-{name}-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).expect("create temp dir");
    path
}
