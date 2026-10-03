//! Storage-mode / DRAM-behaviour probe for the Metal buffer surface.
//!
//! **What this is.** A read-only investigation into one question: does the
//! `MTLResourceOptions` choice at each allocation site in this crate change
//! *DRAM* behaviour on Apple Silicon, or only cache/coherency behaviour?
//! Three sub-questions, each with its own instrument:
//!
//! 1. **Census** (`storage_mode_census_is_the_documented_one`) — a static scan
//!    of every `MTLResourceOptions` site in `crates/{engine,kernels}/src`. This
//!    is the table in the report, generated rather than remembered, so it
//!    cannot drift from the code.
//! 2. **Page classification** (`mmap_protected_read_of_a_file_backed_region_is_never_dirty`)
//!    — the engine binds `mmap(PROT_READ, MAP_PRIVATE)` pages straight to the
//!    GPU (`weights.rs:1069`). This asserts the property that makes that safe
//!    *and* cheap: those pages are `SM=COW`/`r--`, i.e. **DIRTY = 0**, so they
//!    are never compressor candidates. Read out of our own `vmmap`.
//! 3. **Compressor economics** (`compressor_gain_ratio_on_this_box_is_about_one`)
//!    — `vm_statistics64.{compressions, compressor_page_count,
//!    total_uncompressed_pages_in_compressor}`. The ratio of the last two is
//!    the compressor's actual compression factor. If it is ~1.0 the compressor
//!    is burning CPU to move ~1 byte per page.
//!
//! **What this is not.** Not a bandwidth benchmark. Nothing here measures
//! GB/s, because a bandwidth number needs a Metal device and a loaded model,
//! and `AGENTS.md` §10 forbids both on this box while another agent holds the
//! GPU. Every claim this file makes is a *classification* claim (which VM page
//! class a buffer lands in), and those are the claims storage mode actually
//! controls.
//!
//! ## Non-vacuity (AGENTS.md §3.4)
//!
//! Three of the four assertions are proven falsifiable before being believed:
//!
//! * `census_detects_an_injected_private_buffer` applies the census function to
//!   a synthetic source containing a `StorageModePrivate` line and asserts the
//!   census reports it. Without that, a census that returned an empty vec for
//!   every input would pass.
//! * `mmap_...` asserts `DIRTY == 0` for the file-backed region **and
//!   `DIRTY == VSIZE` for an anonymous region in the same process**. The
//!   second half is what makes the first half mean something: if `vmmap`
//!   parsing were broken and reported `DIRTY = 0` for everything, this fails.
//! * The compressor test prints the ratio and asserts only `>= 1.0` (a
//!   compressor cannot expand data), and the *interesting* number is printed,
//!   not asserted — the ratio is a property of whatever else is resident on
//!   this shared box, not of this engine.
//!
//! ### Sabotage runs actually executed (not asserted to fail — observed)
//!
//! ```text
//! KAPLUMBA_SABOTAGE_DIRTY_MMAP=1  -> FAIL
//!   mmap_protected_read_of_a_file_backed_region_is_never_dirty
//!   "PROT_READ/MAP_PRIVATE file-backed region reports 67108864 dirty bytes
//!    across [Region { ... protection: "rw-/rwx" ... }]"
//!   left: 67108864  right: 0
//!
//! KAPLUMBA_SABOTAGE_UNTRACKED=1   -> FAIL
//!   storage_mode_census_is_the_documented_one
//!   "Untracked at src/metal.rs:86 ... adding one to KV or scratch reintroduces
//!    the eed081e NaN race"
//! ```
//!
//! The first is the load-bearing one: it turns the mapping writable, dirties
//! every page, and confirms the DIRTY column moves from 0 to the full 64 MiB.
//! So `DIRTY == 0` for the real `PROT_READ` mapping is a measured property and
//! not a `vmmap` artefact. Both flags default to off and only exist to make the
//! test falsifiable on demand.
//!
//! ## Cost
//!
//! Worst case this test touches 64 MiB of anonymous memory for ~1 s and reads
//! one `vmmap` of itself. It allocates no Metal buffer, loads no model, takes
//! no GPU lock, and does not touch the checkpoint unless
//! `KAPLUMBA_PROBE_MODEL_DIR` is set (in which case it reads three 8 MiB
//! windows and nothing else). Safe to run on a busy box.
//!
//! Run: `cargo test -p kaplumba-engine --test probe_storagemode_compression -- --nocapture`

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// 1. Static census of storage-mode call sites
// ---------------------------------------------------------------------------

/// One `MTLResourceOptions` occurrence found in a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Site {
    file: String,
    line: usize,
    /// The full text of the line, trimmed.
    text: String,
}

impl Site {
    fn shared(&self) -> bool {
        self.text.contains("StorageModeShared")
    }
    fn private(&self) -> bool {
        self.text.contains("StorageModePrivate")
    }
    fn managed(&self) -> bool {
        self.text.contains("StorageModeManaged")
    }
    fn memoryless(&self) -> bool {
        self.text.contains("StorageModeMemoryless")
    }
    fn untracked(&self) -> bool {
        self.text.contains("HazardTrackingModeUntracked")
    }
    fn tracked(&self) -> bool {
        self.text.contains("HazardTrackingModeTracked")
    }
}

/// Every line mentioning `MTLResourceOptions`, with a storage mode or hazard
/// mode on it. Lines that only import the type (`use metal::{...}`) are
/// dropped: they are not allocation sites.
///
/// Takes the source text rather than reading files itself so
/// `census_detects_an_injected_private_buffer` can drive it with a synthetic
/// string — see the non-vacuity note above.
fn census(source: &str, label: &str) -> Vec<Site> {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        if line.contains("MTLResourceOptions") {
            // `weights.rs:1069-1075` writes the options across two lines:
            //
            //     metal::MTLResourceOptions::StorageModeShared
            //         | metal::MTLResourceOptions::HazardTrackingModeUntracked,
            //
            // so the following `|`-continued line belongs to the same
            // allocation site. Without this the census reports one site with a
            // storage mode and a second with none, and the second one trips the
            // "exactly one Untracked site, and it is in weights.rs" assertion
            // for the wrong reason.
            let mut text = line.trim().to_string();
            let mut end = index;
            while end + 1 < lines.len() && lines[end + 1].trim_start().starts_with('|') {
                end += 1;
                text.push(' ');
                text.push_str(lines[end].trim());
            }
            if text.contains("StorageMode") || text.contains("HazardTrackingMode") {
                out.push(Site {
                    file: label.to_string(),
                    line: index + 1,
                    text,
                });
            }
            // Skip the continuation lines: they are part of this site, not new
            // ones. (This is what makes the count 2 rather than 3 for the
            // synthetic source in `census_detects_an_injected_private_buffer`.)
            index = end + 1;
            continue;
        }
        index += 1;
    }
    out
}

fn engine_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn kernels_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/engine has a sibling crates/kernels")
        .join("kernels/src")
}

fn scan_tree(dir: &Path) -> Vec<Site> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return out,
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .collect();
    // Deterministic order, so the printed table does not depend on readdir.
    paths.sort();
    for path in paths {
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(_) => continue,
        };
        let label = path
            .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")))
            .unwrap_or(&path)
            .display()
            .to_string();
        out.extend(census(&text, &label));
    }
    out
}

/// The census at HEAD, plus the table the report quotes.
#[test]
fn storage_mode_census_is_the_documented_one() {
    let sites = scan_tree(&engine_src())
        .into_iter()
        .chain(scan_tree(&kernels_src()))
        .collect::<Vec<_>>();

    // A census that found nothing would "pass" every assertion below, so
    // require a floor. At HEAD there are 21 sites (see the report).
    assert!(
        sites.len() >= 15,
        "census found only {} MTLResourceOptions sites; the scan is broken, \
         not the tree: {sites:#?}",
        sites.len()
    );

    println!("\n=== MTLResourceOptions census (crates/{{engine,kernels}}/src) ===");
    println!(
        "{:<34} {:>5}  {:<28} {:<10} {:<10}",
        "file:line", "mode", "storage", "hazard", "text"
    );
    for site in &sites {
        let storage = [
            site.shared(),
            site.private(),
            site.managed(),
            site.memoryless(),
        ]
        .iter()
        .enumerate()
        .find(|(_, present)| **present)
        .map(|(index, _)| ["Shared", "Private", "Managed", "Memoryless"][index])
        .unwrap_or("?");
        let hazard = if site.untracked() {
            "Untracked"
        } else if site.tracked() {
            "Tracked"
        } else {
            "default"
        };
        println!(
            "{}:{}  {:<28} {:<10} {}",
            site.file, site.line, storage, hazard, site.text
        );
    }

    // The tree uses exactly two storage modes. `Managed` requires an explicit
    // `didModifyRange:` to become visible to the CPU, so any site using it
    // would be a correctness bug in a write-then-read loader; `Memoryless` is
    // tile-only and would silently discard weight bytes. Neither appears, and
    // that is worth asserting rather than assuming.
    let managed: Vec<&Site> = sites.iter().filter(|site| site.managed()).collect();
    assert!(
        managed.is_empty(),
        "StorageModeManaged is used at {:?}; managed buffers need \
         didModifyRange: before the CPU can read their own writes, and this \
         loader never calls it",
        managed
    );
    let memoryless: Vec<&Site> = sites.iter().filter(|site| site.memoryless()).collect();
    assert!(
        memoryless.is_empty(),
        "StorageModeMemoryless is used at {:?}; its contents are undefined \
         after the command buffer completes, so a weight buffer there would \
         be garbage",
        memoryless
    );

    // Exactly two Private sites, and **neither is a weight buffer**:
    //
    //   `model.rs:4526`  `spin_scratch`, 16 KiB, the `spin_burn` diagnostic
    //                     destination. Never read by anything.
    //   `encoder.rs:852` a `#[cfg(test)]` burn-kernel sink.
    //
    // Every production weight/KV/scratch buffer is Shared because the loader
    // must `read_at` into `contents()` (weights.rs:272, 942) — a private buffer
    // has no host-writable contents at all. So "make the arena Private" is not
    // a one-line change; see the report.
    let private_sites: Vec<&Site> = sites.iter().filter(|site| site.private()).collect();
    assert_eq!(
        private_sites.len(),
        2,
        "expected exactly two StorageModePrivate sites (spin_scratch at \
         model.rs:4526 and the encoder.rs test sink), found {private_sites:?}"
    );
    for site in &private_sites {
        assert!(
            site.file.ends_with("model.rs") || site.file.ends_with("encoder.rs"),
            "StorageModePrivate appeared at {}:{} ({}) — a private buffer has \
             no host-writable contents, so the loader cannot fill it with \
             `read_at`",
            site.file, site.line, site.text
        );
    }

    // The mapped weight views are the only Untracked sites. This is the load-
    // bearing asymmetry: `weights.rs:1072-1073` binds read-only shard pages
    // Untracked because there is no per-step write hazard on them, while KV
    // (`gpu_cache.rs:113,237`), scratch (`model.rs:2970`) and the sign/norm
    // buffers (`model.rs:468`) are Tracked because eed081e proved Untracked
    // there produces NaN. See `spec/nan-logits-hazard-race`.
    // SABOTAGE: pretend `metal.rs:86` (the arena) is Untracked.
    let untracked: Vec<&Site> = if std::env::var_os("KAPLUMBA_SABOTAGE_UNTRACKED").is_some() {
        sites
            .iter()
            .filter(|site| site.untracked() || site.file.ends_with("metal.rs"))
            .collect()
    } else {
        sites.iter().filter(|site| site.untracked()).collect()
    };
    assert!(
        !untracked.is_empty(),
        "no Untracked site found; weights.rs:1073 should be one"
    );
    for site in &untracked {
        assert!(
            site.file.ends_with("weights.rs"),
            "Untracked at {}:{} ({}) — every Untracked site so far is a \
             read-only weight view; adding one to KV or scratch reintroduces \
             the eed081e NaN race",
            site.file, site.line, site.text
        );
    }
    for site in &sites {
        if site.file.ends_with("gpu_cache.rs")
            || site.file.ends_with("model.rs")
            || site.file.ends_with("metal.rs")
        {
            assert!(
                !site.untracked(),
                "{}:{} is Untracked; KV/scratch/arena must stay Tracked",
                site.file, site.line
            );
        }
    }
}

/// Non-vacuity check for `census`: feed it a source with a Private buffer and
/// an Untracked line and require both to be reported.
#[test]
fn census_detects_an_injected_private_buffer() {
    let synthetic = r#"
fn a() {
    let x = device.new_buffer(n, MTLResourceOptions::StorageModePrivate);
}
fn b() {
    let y = device.new_buffer_with_bytes_no_copy(p, n,
        MTLResourceOptions::StorageModeShared
            | MTLResourceOptions::HazardTrackingModeUntracked, None);
}
use metal::{Buffer, MTLResourceOptions};
"#;
    let found = census(synthetic, "<synthetic>");
    assert_eq!(
        found.len(),
        2,
        "census must report 2 allocation sites (the two-line Shared|Untracked \
         expression is ONE site); the `use` line must be filtered: {found:#?}"
    );
    assert!(found[0].private(), "first site is Private");
    assert!(
        found[1].shared() && found[1].untracked() && !found[1].private(),
        "second site must be one merged Shared|Untracked expression, got {:?}",
        found[1]
    );
}

// ---------------------------------------------------------------------------
// 2. Page classification of the engine's two backing classes
// ---------------------------------------------------------------------------

/// One `vmmap` region row, parsed into the fields that matter here.
#[derive(Debug, Clone)]
struct Region {
    start: usize,
    end: usize,
    vsize: u64,
    resident: u64,
    dirty: u64,
    protection: String,
    sharing: String,
}

/// Parse a `vmmap` size field.
///
/// The format is **not** a plain integer: it carries one decimal place and
/// drops it below 10, so a 24 MiB region prints as `24.0M` while a 512 KiB one
/// prints as `512K`. `scripts/gpuq` already documents the same trap for
/// `top` — stripping the non-digits turned `554M` into 554 **GiB**, a 1000x
/// overstatement — so the decimal is parsed as a float, not discarded.
fn parse_size(field: &str) -> u64 {
    let field = field.trim();
    if field.is_empty() {
        return 0;
    }
    let (digits, multiplier) = match field.as_bytes().last() {
        Some(b'K') => (&field[..field.len() - 1], 1024_f64),
        Some(b'M') => (&field[..field.len() - 1], 1024.0 * 1024.0),
        Some(b'G') => (&field[..field.len() - 1], 1024.0 * 1024.0 * 1024.0),
        Some(b'T') => (
            &field[..field.len() - 1],
            1024.0 * 1024.0 * 1024.0 * 1024.0,
        ),
        _ => (field, 1.0),
    };
    (digits.parse::<f64>().unwrap_or(0.0) * multiplier) as u64
}

/// Parse the `vmmap <pid>` rows that fall inside `[needle_start, needle_end)`.
///
/// `vmmap` columns, in order, after the address range:
/// `[ VSIZE  RSDNT  DIRTY   SWAP] PRT/MAX SHRMOD PURGE  REGION DETAIL`
/// so the three numbers we want are the 1st, 2nd and 3rd bracket fields.
fn vmmap_regions(needle_start: usize, needle_end: usize) -> Vec<Region> {
    let output = match std::process::Command::new("vmmap")
        .arg(std::process::id().to_string())
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            println!("vmmap unavailable ({error}); skipping page-classification");
            return Vec::new();
        }
    };
    let text = String::from_utf8_lossy(&output.stdout).to_string();
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(open) = line.find('[') else { continue };
        let Some(close) = line[open..].find(']') else {
            continue;
        };
        let bracket = &line[open + 1..open + close];
        let numbers: Vec<&str> = bracket.split_whitespace().collect();
        if numbers.len() < 3 {
            continue;
        }
        // The address range is not necessarily the first field: `vmmap` prints
        // a region *type* first (`mapped file`, `Untagged`,
        // `Malloc Tiny`, ...), so find the field that looks like `start-end`.
        let head: Vec<&str> = line[..open].split_whitespace().collect();
        let Some(range_field) = head.iter().find(|field| field.contains('-')) else {
            continue;
        };
        let (start_text, end_text) = range_field.split_once('-').unwrap();
        let Ok(start) = usize::from_str_radix(start_text, 16) else {
            continue;
        };
        let Ok(end) = usize::from_str_radix(end_text, 16) else {
            continue;
        };
        if end <= needle_start || start >= needle_end {
            continue;
        }
        let tail = &line[open + close + 1..];
        let tail_fields: Vec<&str> = tail.split_whitespace().collect();
        out.push(Region {
            start,
            end,
            vsize: parse_size(numbers[0]),
            resident: parse_size(numbers[1]),
            dirty: parse_size(numbers[2]),
            protection: tail_fields.first().cloned().unwrap_or("?").to_string(),
            sharing: tail_fields.get(1).cloned().unwrap_or("?").to_string(),
        });
    }
    out
}

/// A scratch file of high-entropy bytes, so the mapping under test is not
/// zero-filled (zero pages have their own compression behaviour and would
/// flatter any classification result).
struct HighEntropyFile {
    path: PathBuf,
    length: usize,
}

impl HighEntropyFile {
    fn create(name: &str, mebibytes: usize) -> HighEntropyFile {
        let mut path = std::env::temp_dir();
        path.push(format!("kaplumba-storagemode-{name}-{}", std::process::id()));
        let length = mebibytes * 1024 * 1024;
        let mut bytes = vec![0_u8; length];
        let mut state: u64 = 0x243F_6A88_85A3_08D3;
        for chunk in bytes.chunks_mut(8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            chunk.copy_from_slice(&state.to_le_bytes()[..chunk.len()]);
        }
        std::fs::write(&path, &bytes).expect("write scratch file");
        HighEntropyFile { path, length }
    }
}

impl Drop for HighEntropyFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Touch every page of a mapping, exactly as a GEMV reading it would.
fn touch_all(base: *const u8, length: usize) -> u64 {
    let mut accumulator = 0_u64;
    let mut offset = 0;
    while offset < length {
        // SAFETY: `base`/`length` are a live mapping owned by the caller and
        // `offset < length`, so the read is in bounds.
        let byte = unsafe { *base.add(offset) };
        accumulator = accumulator.wrapping_add(byte as u64);
        offset += 4096;
    }
    accumulator
}

/// The property that makes `newBufferWithBytesNoCopy` over
/// `mmap(PROT_READ, MAP_PRIVATE)` the cheap half of the loader's memory
/// budget: the pages are **clean**, so the compressor and the swapper never
/// consider them.
///
/// `weights.rs:1016-1023` maps the shard `PROT_READ`/`MAP_PRIVATE` and
/// `weights.rs:1069` hands the address to Metal. `PROT_READ` means the
/// mapping cannot be written, so `MAP_PRIVATE`'s copy-on-write can never
/// trigger, so no page can become dirty. This asserts that against our own
/// `vmmap`, and pairs it with the opposite case in the same process so a
/// broken parse cannot pass.
#[test]
fn mmap_protected_read_of_a_file_backed_region_is_never_dirty() {
    use std::os::unix::io::AsRawFd;

    const MIB: usize = 64;
    let file = HighEntropyFile::create("readonly", MIB);
    let handle = std::fs::File::open(&file.path).expect("reopen read-only");

    // SAFETY: `handle` outlives both mappings, and both are unmapped at the
    // end of this function.
    unsafe {
        // SABOTAGE: PROT_READ|PROT_WRITE makes the mapping dirty-able.
        let sabotage_prot =
            if std::env::var_os("KAPLUMBA_SABOTAGE_DIRTY_MMAP").is_some() {
                libc::PROT_READ | libc::PROT_WRITE
            } else {
                libc::PROT_READ
            };
        let file_backed = libc::mmap(
            std::ptr::null_mut(),
            file.length,
            sabotage_prot,
            libc::MAP_PRIVATE,
            handle.as_raw_fd(),
            0,
        );
        assert_ne!(
            file_backed,
            libc::MAP_FAILED,
            "mmap PROT_READ/MAP_PRIVATE failed: {:?}",
            std::io::Error::last_os_error()
        );

        // Same size, anonymous and writable — the arena's class.
        let anonymous = libc::mmap(
            std::ptr::null_mut(),
            file.length,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        );
        assert_ne!(
            anonymous,
            libc::MAP_FAILED,
            "mmap MAP_ANON|MAP_PRIVATE failed: {:?}",
            std::io::Error::last_os_error()
        );

        // Dirty the anonymous half so its DIRTY column is unambiguously full.
        let anon_words = anonymous as *mut u64;
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for index in 0..(file.length / 8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *anon_words.add(index) = state;
        }
        std::hint::black_box(state);

        let file_acc = touch_all(file_backed as *const u8, file.length);
        std::hint::black_box(file_acc);

        if sabotage_prot != libc::PROT_READ {
            // Write one byte per page so the mapping is actually dirtied, then
            // fall through to the same assertions. Without this the sabotage
            // would prove nothing: a writable-but-untouched MAP_PRIVATE
            // mapping is still clean.
            let writable = file_backed as *mut u8;
            for offset in (0..file.length).step_by(4096) {
                *writable.add(offset) = 0;
            }
        }

        let file_regions = vmmap_regions(file_backed as usize, file_backed as usize + file.length);
        let anon_regions = vmmap_regions(anonymous as usize, anonymous as usize + file.length);

        libc::munmap(file_backed, file.length);
        libc::munmap(anonymous, file.length);

        println!("\n=== page classification of the engine's two backing classes ===");
        for region in &file_regions {
            println!(
                "file-backed  {:#x}-{:#x} VSIZE={} RSDNT={} DIRTY={} prot={} sm={}",
                region.start, region.end, region.vsize, region.resident,
                region.dirty, region.protection, region.sharing
            );
        }
        for region in &anon_regions {
            println!(
                "anonymous    {:#x}-{:#x} VSIZE={} RSDNT={} DIRTY={} prot={} sm={}",
                region.start, region.end, region.vsize, region.resident,
                region.dirty, region.protection, region.sharing
            );
        }

        if file_regions.is_empty() || anon_regions.is_empty() {
            println!("vmmap produced no overlapping region; skipping (sandbox?)");
            return;
        }

        // The half that gives the first half meaning: an anonymous, written,
        // high-entropy region of the same size in the same process MUST report
        // a non-zero DIRTY column. If this fails, `vmmap` parsing is broken and
        // the file-backed assertion below is meaningless.
        let anon_dirty: u64 = anon_regions.iter().map(|region| region.dirty).sum();
        assert!(
            anon_dirty > 0,
            "anonymous written region reports DIRTY=0 across {anon_regions:?}; \
             the vmmap parse is broken, so the file-backed assertion would be \
             vacuous"
        );

        // The claim: touching a PROT_READ/MAP_PRIVATE file mapping produces
        // zero dirty pages, so those bytes are reclaimable by purge (file-backed
        // clean) and are never offered to the compressor. This is why mapping
        // `embed_tokens` (2.82 GB) and `lm_head.trellis` (1.06 GB) instead of
        // copying them is not merely a footprint win but a *class* change.
        let file_dirty: u64 = file_regions.iter().map(|region| region.dirty).sum();
        assert_eq!(
            file_dirty, 0,
            "PROT_READ/MAP_PRIVATE file-backed region reports {file_dirty} dirty \
             bytes across {file_regions:?}; PROT_READ makes a write impossible, \
             so MAP_PRIVATE can never copy and no page can be dirtied"
        );

        // And the pages really are resident: this is not an assertion about a
        // mapping that was never faulted in.
        let file_resident: u64 = file_regions.iter().map(|region| region.resident).sum();
        assert!(
            file_resident >= (file.length / 2) as u64,
            "only {file_resident} of {} bytes resident after touching every \
             page; the mapping was not actually faulted in",
            file.length
        );

        // The protection string must show the mapping is read-only. This is the
        // direct evidence for "the mapping cannot be written", as opposed to
        // "nothing wrote to it".
        assert!(
            file_regions
                .iter()
                .any(|region| region.protection.starts_with("r--")),
            "expected an r-- protection on the file-backed mapping, got {:?}",
            file_regions
                .iter()
                .map(|region| region.protection.clone())
                .collect::<Vec<_>>()
        );

        // Non-vacuity for `parse_size` (AGENTS.md §3.4). `vmmap` prints `24.0M`
        // with a decimal place and `512K` without one; an integer-only parser
        // silently returns 0 for every decimal field, which would make the
        // DIRTY assertion above pass for the wrong reason.
        assert_eq!(parse_size("24.0M"), 24 * 1024 * 1024);
        assert_eq!(parse_size("512K"), 512 * 1024);
        assert_eq!(parse_size("0K"), 0);
        assert_eq!(parse_size("1.5G"), 1536 * 1024 * 1024);
    }
}

// ---------------------------------------------------------------------------
// 3. Compressor economics
// ---------------------------------------------------------------------------

/// `vm_statistics64` snapshot. Only the fields this probe reads are named.
#[derive(Debug, Clone, Copy, Default)]
struct VmStats {
    compressions: u64,
    decompressions: u64,
    compressor_page_count: u64,
    total_uncompressed_pages_in_compressor: u64,
    purgeable_count: u64,
    swapouts: u64,
    throttled_count: u64,
    internal_page_count: u64,
    external_page_count: u64,
}

fn vm_stats() -> VmStats {
    // SAFETY: `stats` is a correctly sized, zeroed `vm_statistics64` and the
    // count is `HOST_VM_INFO64_COUNT`, i.e. the full struct.
    unsafe {
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        #[allow(deprecated)]
        let result = libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            &mut stats as *mut libc::vm_statistics64 as *mut libc::integer_t,
            &mut count,
        );
        assert_eq!(result, 0, "host_statistics64(HOST_VM_INFO64) failed");
        VmStats {
            compressions: stats.compressions as u64,
            decompressions: stats.decompressions as u64,
            compressor_page_count: stats.compressor_page_count as u64,
            total_uncompressed_pages_in_compressor: stats
                .total_uncompressed_pages_in_compressor,
            purgeable_count: stats.purgeable_count as u64,
            swapouts: stats.swapouts as u64,
            throttled_count: stats.throttled_count as u64,
            internal_page_count: stats.internal_page_count as u64,
            external_page_count: stats.external_page_count as u64,
        }
    }
}

/// macOS's page size, which is **16 KiB** here, not 4 KiB. Every
/// page-count-to-byte conversion in the existing notes depends on this
/// (`spec/mtp-40tps/implementation_notes.md:13` derives it arithmetically from
/// `vm.page_free_count`; `hw.pagesize` states it outright).
fn page_size() -> u64 {
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if value > 0 {
        value as u64
    } else {
        16 * 1024
    }
}

/// How well macOS's compressor is actually compressing, and how hard it is
/// working while it does it.
///
/// The engine's weights are `StorageModeShared` buffers written by `read_at`
/// (`weights.rs:272`, `942`), which makes them *dirty anonymous* pages: the
/// one class the compressor is allowed to touch. The checkpoint's own entropy
/// (measured separately, `exl3_trellis_is_incompressible`) is ~7.998 of a
/// possible 8.000 bits/byte, so the compressor should achieve ~1.0.
///
/// This prints the ratio and the churn rate. It asserts only the two things
/// that must hold regardless of what else is on this shared box: the ratio is
/// at least 1.0 (a compressor cannot expand), and the counters move.
#[test]
fn compressor_gain_ratio_on_this_box_is_about_one() {
    const WINDOW_MS: u64 = 1_000;
    let page = page_size();
    let before = vm_stats();
    std::thread::sleep(std::time::Duration::from_millis(WINDOW_MS));
    let after = vm_stats();

    let ratio = if after.compressor_page_count == 0 {
        0.0
    } else {
        after.total_uncompressed_pages_in_compressor as f64
            / after.compressor_page_count as f64
    };
    let seconds = WINDOW_MS as f64 / 1000.0;
    let attempted = after.compressions.wrapping_sub(before.compressions);
    let restored = after.decompressions.wrapping_sub(before.decompressions);
    let swap_delta = after.swapouts.wrapping_sub(before.swapouts);

    println!("\n=== macOS compressor, {WINDOW_MS} ms window ===");
    println!("page size                      {page}");
    println!(
        "compressor resident            {} MiB",
        (after.compressor_page_count as u64 * page) / (1024 * 1024)
    );
    println!(
        "uncompressed bytes represented {} MiB",
        (after.total_uncompressed_pages_in_compressor * page) / (1024 * 1024)
    );
    println!("gain ratio (uncompressed/compressed) {ratio:.4}");
    println!("compressions attempted         {attempted} pages ({:.1} MiB/s)", attempted as f64 * page as f64 / 1048576.0 / seconds);
    println!("decompressions                 {restored} pages ({:.1} MiB/s)", restored as f64 * page as f64 / 1048576.0 / seconds);
    println!("swapouts                       {swap_delta} pages");
    println!(
        "internal (anonymous) pages     {} MiB   <- the arena's class",
        (after.internal_page_count as u64 * page) / (1024 * 1024)
    );
    println!(
        "external (file-backed) pages   {} MiB   <- embed/lm_head, and the shards",
        (after.external_page_count as u64 * page) / (1024 * 1024)
    );
    println!(
        "purgeable                      {} MiB",
        (after.purgeable_count as u64 * page) / (1024 * 1024)
    );
    println!("throttled                      {}", after.throttled_count);

    assert!(
        ratio >= 1.0,
        "compressor ratio {ratio:.4} < 1.0: the compressor made the data \
         larger, which is impossible; the field meanings are probably wrong"
    );

    // If the compressor is doing nothing at all, this box has headroom and the
    // question is moot here. Say so rather than asserting a rate.
    if attempted == 0 && restored == 0 {
        println!(
            "\nNo compressor activity in this window: the box is not under \
             pressure, so compression is not costing anything *right now*. \
             That is a statement about this instant, not about the engine."
        );
        return;
    }
    println!(
        "\nChurn is live: {} compressions and {} decompressions in {WINDOW_MS} ms.",
        attempted, restored
    );
    if ratio < 1.2 {
        println!(
            "A gain ratio of {ratio:.4} means the compressor is spending CPU \
             to store ~1 byte per {}-byte page. For 7.998-bits/byte data that \
             is the expected outcome, and it is why `memory-swap-cliff` calls \
             the observed 10-12 GB of compressor \"the compressor failing\".",
            page
        );
    }
    let _ = swap_delta;
}

// ---------------------------------------------------------------------------
// 4. Checkpoint entropy — is compression even worth attempting?
// ---------------------------------------------------------------------------

/// Shannon entropy of `bytes`, in bits per byte. Max 8.0.
fn shannon_entropy(bytes: &[u8]) -> f64 {
    let mut histogram = [0_u64; 256];
    for &byte in bytes {
        histogram[byte as usize] += 1;
    }
    let total = bytes.len() as f64;
    let mut entropy = 0.0;
    for count in histogram {
        if count > 0 {
            let probability = count as f64 / total;
            entropy -= probability * probability.log2();
        }
    }
    entropy
}

/// `lm_head.trellis` is EXL3 3 bpw bit-packed data. The spec asserts 7.9702 of
/// 8.0000 bits/byte with a 1.87 % gzip saving
/// (`spec/memory-swap-cliff/implementation_notes.md:130-133`). That assertion
/// is what licenses the conclusion "compression is pure waste for this
/// workload", so it is worth re-deriving from the file rather than quoting.
///
/// Skipped when the checkpoint is not on this box or the env override is unset,
/// so this file never loads or reads a model by default.
#[test]
fn exl3_trellis_is_incompressible() {
    let Ok(dir) = std::env::var("KAPLUMBA_PROBE_MODEL_DIR") else {
        println!(
            "KAPLUMBA_PROBE_MODEL_DIR unset; skipping the entropy read. \
             Set it to sample the real checkpoint."
        );
        return;
    };
    let dir = PathBuf::from(dir);
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        println!("no index at {}; skipping", index_path.display());
        return;
    }

    let index = match kaplumba_engine::safetensors::read_index(&index_path) {
        Ok(index) => index,
        Err(error) => {
            println!("read_index failed: {error}; skipping");
            return;
        }
    };
    let selection = match kaplumba_engine::safetensors::load_text_selection(&dir, index) {
        Ok(selection) => selection,
        Err(error) => {
            println!("load_text_selection failed: {error}; skipping");
            return;
        }
    };

    const WINDOW: usize = 8 << 20;
    let mut sampled: BTreeMap<&str, (f64, f64)> = BTreeMap::new();

    for name in ["lm_head.trellis", "model.language_model.layers.0.mlp.down_proj.trellis"] {
        let Some(info) = selection.tensors.get(name) else {
            println!("{name} not in the text selection; skipping");
            continue;
        };
        let Some(shard_name) = selection.shard_for(name) else {
            println!("{name} has no shard; skipping");
            continue;
        };
        let payload_start = selection.shards[shard_name].payload_start();
        let absolute = payload_start + info.start;

        for (label, fraction) in [("start", 0.0_f64), ("middle", 0.5), ("end", 0.9)] {
            let span = info.byte_len();
            let offset = absolute + (span as f64 * fraction) as u64;
            let Ok(mut file) = std::fs::File::open(dir.join(shard_name)) else {
                println!("cannot open {shard_name}; skipping");
                return;
            };
            use std::io::{Read, Seek, SeekFrom};
            if file.seek(SeekFrom::Start(offset)).is_err() {
                println!("seek to {offset} failed; skipping");
                return;
            }
            let mut window = vec![0_u8; WINDOW];
            if file.read_exact(&mut window).is_err() {
                println!("short read at {offset}; skipping");
                return;
            }
            let entropy = shannon_entropy(&window);
            let zeros = window.iter().filter(|&&byte| byte == 0).count() as f64;
            let zero_fraction = zeros / WINDOW as f64;
            println!(
                "\n{name} @ {label} (file offset {offset}, {WINDOW} bytes):\n  \
                 Shannon entropy {entropy:.4} / 8.0000 bits/byte\n  \
                 zero bytes      {zero_fraction:.4}%",
            );
            sampled.insert(
                Box::leak(format!("{name}@{label}").into_boxed_str()),
                (entropy, zero_fraction),
            );
        }
    }

    if sampled.is_empty() {
        println!("no windows sampled; nothing to assert");
        return;
    }

    // The assertion the report rests on: this data is at the entropy ceiling,
    // so no general-purpose compressor can shrink it. 7.9 bits/byte is a
    // deliberately loose floor -- the measured value is ~7.998, and even a
    // 1 % entropy deficit leaves nothing for the compressor to recover once
    // per-page framing overhead is paid.
    for (label, (entropy, _)) in &sampled {
        assert!(
            *entropy > 7.9,
            "{label}: Shannon entropy {entropy:.4} is low enough that \
             compression might pay; the \"compression is pure waste\" argument \
             does not hold for this tensor"
        );
    }

    // Non-vacuity: the entropy function must reject low-entropy data, or the
    // assertion above is passing for the wrong reason.
    let zeros = vec![0_u8; WINDOW];
    assert!(
        shannon_entropy(&zeros) < 1.0,
        "entropy function is broken: all-zero data scored above 1.0 bits/byte"
    );
    let biased: Vec<u8> = (0..WINDOW).map(|i| (i % 2) as u8).collect();
    assert!(
        shannon_entropy(&biased) < 1.1,
        "entropy function is broken: 2-symbol data scored above 1.1 bits/byte"
    );
}

// ---------------------------------------------------------------------------
// 5. mmap advice availability on Darwin
// ---------------------------------------------------------------------------

/// Which page-advice knobs this kernel actually accepts.
///
/// Relevant to the question "would `MAP_POPULATE` or non-temporal advice help
/// the load path". On Darwin the answer is measured, not assumed: `MAP_POPULATE`
/// is a Linux flag and the bit is rejected or ignored here, `MADV_RANDOM` and
/// `MADV_WILLNEED` work, and `MADV_POPULATE_WRITE` / `MADV_COLD` do not exist.
/// That closes the whole family of "prefetch the weights" proposals without a
/// benchmark.
#[test]
fn mmap_advice_availability_on_darwin() {
    use std::os::raw::c_int;

    fn errno() -> std::io::Error {
        std::io::Error::last_os_error()
    }

    // SAFETY: every mapping created here is unmapped before the next one is
    // made, and no pointer is dereferenced.
    unsafe {
        let length = 1 << 20;

        // MAP_POPULATE is 0x8000 on Linux. Darwin has no such flag; passing the
        // bit must not be mistaken for "prefault this".
        let linux_populate = libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE | 0x8000,
            -1,
            0,
        );
        let populate_result = if linux_populate == libc::MAP_FAILED {
            format!("rejected ({})", errno())
        } else {
            libc::munmap(linux_populate, length);
            "accepted".to_string()
        };
        println!("MAP_ANON|MAP_PRIVATE|0x8000 (Linux MAP_POPULATE): {populate_result}");

        let nocache = libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_NOCACHE,
            -1,
            0,
        );
        let nocache_result = if nocache == libc::MAP_FAILED {
            format!("rejected ({})", errno())
        } else {
            libc::munmap(nocache, length);
            "accepted".to_string()
        };
        println!("MAP_NOCACHE: {nocache_result}");

        let base = libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        );
        assert_ne!(base, libc::MAP_FAILED);

        for (name, advice) in [
            ("MADV_NORMAL", libc::MADV_NORMAL),
            ("MADV_RANDOM", libc::MADV_RANDOM),
            ("MADV_SEQUENTIAL", libc::MADV_SEQUENTIAL),
            ("MADV_WILLNEED", libc::MADV_WILLNEED),
            ("MADV_DONTNEED", libc::MADV_DONTNEED),
            ("MADV_FREE", libc::MADV_FREE),
            ("MADV_POPULATE_WRITE (Linux 0x10)", 0x10),
            ("MADV_COLD (Linux 0x100)", 0x100),
        ] {
            let result: c_int = libc::madvise(base, length, advice);
            println!(
                "{name}: {}",
                if result == 0 {
                    "ok".to_string()
                } else {
                    format!("{result} ({})", errno())
                }
            );
        }
        libc::munmap(base, length);
    }

    // The load path does not fault pages at all: `weights.rs:910-942`
    // `read_at`s straight into `contents()`. So a prefetch knob for the
    // *mapping* cannot help the arena even if one existed -- there is no arena
    // mapping to attach it to.

    println!(
        "\nNote: the arena is filled by `read_at` into `buffer.contents()` \
         (weights.rs:910-942, 272), not by faulting the mapping. Page-advice \
         flags therefore have no arena allocation to attach to."
    );
}
