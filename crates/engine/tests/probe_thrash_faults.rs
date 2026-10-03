//! PROBE (measurement only, not a regression gate): per-token page-fault rate
//! for the 3 bpw and 4 bpw checkpoints, and where `lm_head.trellis` lives.
//!
//! ## Why this file exists
//!
//! `Gxma-4-bit` decodes at 0.014-0.088 tok/s while `Gxma-repacked` (3 bpw)
//! decodes at 16.1-16.3 tok/s on the same box, from the same binary, producing
//! byte-identical output. Four hypotheses were on the table (see
//! `spec/decode-ceiling/findings/THRASH-4BIT.md`):
//!
//! * **H1** `lm_head.trellis` is file-backed and re-read from SSD every token.
//! * **H2** the 14.6 GB anonymous layer trellis thrashes; macOS keeps all of it
//!   or the process collapses to disk.
//! * **H3** it is environmental (a contended box).
//! * **H4** a per-dispatch / per-layer cost that scales with bit rate.
//!
//! Only one number separates H1/H2 from H4: **major faults per generated
//! token**. H1 and H2 both predict a large `ru_majflt` delta per token (a disk
//! round trip). H4 predicts none. `ru_maxrss` cannot see any of it, and neither
//! can `--metrics peak_resident_pages_bytes` (AGENTS.md §10.3: it read 6.23 GiB
//! against a 11.69 GiB `phys_footprint` on the same run). So this probes
//! `getrusage(RUSAGE_SELF)` around each individual decode step.
//!
//! ## What is measured, and what is not
//!
//! MEASURED here:
//!
//! * `ru_minflt` / `ru_majflt` deltas around each `forward_buffer` call, and
//!   the wall-clock time of that call. Per step, in a single process, with the
//!   checkpoint selected by `KAPLUMBA_TEST_MODEL`.
//! * `host_statistics64(HOST_VM_INFO64)` page counts before/after, so
//!   system-wide compressor growth, pageins and swapins are attributable to the
//!   run and not to ambient noise. These are the *host* `pageins`/`swapins`
//!   counters, which are independent of `RUSAGE_SELF` and therefore catch I/O
//!   the process's own fault counters attribute elsewhere (e.g. to the driver).
//! * The checkpoint's own byte ledger, from the safetensors index, so the
//!   weights figure is the checkpoint's and not a number the engine reported
//!   about itself.
//!
//! REASONED, not measured: everything about *why* macOS reclaims a given page.
//! `footprint -v` gives the per-region split but not the readahead behaviour,
//! and `ru_majflt` on macOS counts readahead completions, so one "major fault"
//! can stand for many pages. The byte totals below are therefore *lower bounds*
//! on the traffic, and the honest reading is the ratio between checkpoints,
//! measured from the same binary minutes apart.
//!
//! ## Reproducing
//!
//! ```bash
//! cargo build --release --tests -p kaplumba-engine --test probe_thrash_faults
//! M=$HOME/.kaplumba/models/Gxma-repacked
//! KAPLUMBA_TEST_MODEL=$M /usr/bin/time -l \
//!   ./target/release/deps/probe_thrash_faults-* --ignored --nocapture --test-threads=1
//! ```
//!
//! One model instance at a time, always (AGENTS.md §10.2). `/usr/bin/time -l`
//! is required for `peak memory footprint`; it is not read here, because
//! AGENTS.md §10.3 records two rejected attempts at reading `phys_footprint`
//! in-process.

use std::path::Path;
use std::time::Instant;

use kaplumba_engine::model::TextModel;
use kaplumba_engine::tokenizer::GemmaTokenizer;

/// How many decode steps to time. Each one re-reads the entire weight set, so
/// the delta is measured over a long enough window that the per-step counters
/// are not swamped by the prefill's own faults.
const STEPS: usize = 12;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RUsage {
    minflt: u64,
    majflt: u64,
}

fn rusage() -> RUsage {
    // SAFETY: `getrusage` fills a `rusage` it is handed; nothing is aliased and
    // the struct is plain integers, so there is nothing to uninitialise.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            panic!("getrusage(RUSAGE_SELF) failed");
        }
        RUsage {
            minflt: usage.ru_minflt as i64 as u64,
            majflt: usage.ru_majflt as i64 as u64,
            ..RUsage::default()
        }
    }
}

/// The counters this probe needs, read out of `libc::vm_statistics64`.
///
/// A hand-rolled `#[repr(C)]` mirror is deliberately NOT used: libc's
/// `natural_t` is `u32`, so the kernel's struct mixes 4- and 8-byte fields and
/// an all-`u64` mirror is silently the wrong size. Measured on this box:
/// `HOST_VM_INFO64_COUNT` is **104** integers, and an all-`u64` nine-field
/// struct is 9 — so the mirror would have been a 380-byte overrun past the
/// count the kernel was told to write. libc derives its count from its own
/// `vm_statistics64_data_t`, so libc's struct with libc's count cannot mismatch.
#[derive(Debug, Clone, Copy, Default)]
struct VmStat64 {
    free: u64,
    active: u64,
    inactive: u64,
    wire: u64,
    purgeable: u64,
    speculative: u64,
    compressor: u64,
    pageins: u64,
    pageouts: u64,
    swapins: u64,
    swapouts: u64,
    compressions: u64,
    decompressions: u64,
    page_size: u64,
    pages_total: u64,
}

impl VmStat64 {
    /// The reclaimable-and-fast set `memory_proof.rs` defines as "supply":
    /// free + inactive + purgeable + speculative. Deliberately excludes wired
    /// and compressor pages, which are not available to anyone.
    fn supply_bytes(&self) -> u64 {
        (self.free + self.inactive + self.purgeable + self.speculative) * self.page_size
    }
}

/// `host_statistics64(HOST_VM_INFO64)`: exactly what `vm_stat` itself reads, so
/// this is not the unstable-struct FFI that `task_vm_info` is
/// (`memory_proof.rs:66-80`). Shaped as `memory_proof.rs:130-146` shapes it,
/// which the shipped memory gate exercises.
fn host_vm() -> VmStat64 {
    #[allow(deprecated)]
    unsafe {
        let mut count = libc::HOST_VM_INFO64_COUNT;
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let host = libc::mach_host_self();
        let kr = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut stats as *mut libc::vm_statistics64 as libc::host_info64_t,
            &mut count,
        );
        assert_eq!(kr, libc::KERN_SUCCESS, "host_statistics64 failed: {kr}");
        VmStat64 {
            free: stats.free_count as u64,
            active: stats.active_count as u64,
            inactive: stats.inactive_count as u64,
            wire: stats.wire_count as u64,
            purgeable: stats.purgeable_count as u64,
            speculative: stats.speculative_count as u64,
            compressor: stats.compressor_page_count as u64,
            pageins: stats.pageins,
            pageouts: stats.pageouts,
            swapins: stats.swapins,
            swapouts: stats.swapouts,
            compressions: stats.compressions,
            decompressions: stats.decompressions,
            page_size: page_size(),
            pages_total: pages_total(),
        }
    }
}

/// `vm_page_size`, the value `vm_stat` prints in its own header. 16384 on this
/// box, not 4096 (AGENTS.md §10.3).
fn page_size() -> u64 {
    unsafe { libc::vm_page_size as u64 }
}

/// Total addressable pages: `hw.memsize / page_size`. Exact on this box
/// (25,769,803,776 / 16,384 = 1,573,344 pages), and read from `sysctl` rather
/// than from a kernel counter so it cannot drift with a macOS struct change.
fn pages_total() -> u64 {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .expect("sysctl -n hw.memsize");
    let text = String::from_utf8_lossy(&out.stdout);
    let digits: String = text.trim().chars().filter(char::is_ascii_digit).collect();
    let bytes: u64 = digits
        .parse()
        .unwrap_or_else(|_| panic!("could not read hw.memsize from {text:?}"));
    let ps = page_size();
    assert_eq!(
        bytes % ps,
        0,
        "hw.memsize {bytes} is not a whole number of {ps}-byte pages, so \
         pages_total would be a guess"
    );
    bytes / ps
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

/// Signed page delta in GiB. Signed, because every interesting movement on this
/// box is a *decrease*, and `as u64` on a negative delta would wrap to ~1.8e19
/// and print a plausible wrong number (which is exactly the class of bug
/// AGENTS.md §10.3 records twice).
fn gib_delta(pages: i64, page_size: u64) -> f64 {
    pages as f64 * page_size as f64 / 1_073_741_824.0
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / 1_048_576.0
}

fn delta(after: u64, before: u64) -> i64 {
    after as i64 - before as i64
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and sole GPU access"]
fn probe_faults_per_token() {
    let dir = std::env::var("KAPLUMBA_TEST_MODEL").expect("set KAPLUMBA_TEST_MODEL");
    let root = Path::new(&dir);
    let checkpoint = root
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| dir.clone());

    let prompt = std::env::var("KAPLUMBA_PROBE_PROMPT")
        .unwrap_or_else(|_| "List the integers from 1 to 20, one per line.".to_string());
    let steps: usize = std::env::var("KAPLUMBA_PROBE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(STEPS);

    let before_load = host_vm();
    let before_load_rusage = rusage();
    let load_started = Instant::now();
    let mut model = TextModel::load_from_dir(root).expect("load");
    let load_ms = load_started.elapsed().as_millis();
    let tokenizer = GemmaTokenizer::from_model_dir(root).expect("tokenizer");
    let prompt_ids = tokenizer.encode_prompt(&prompt).expect("encode");

    // The text selection, straight from the safetensors index, so the byte
    // ledger is the checkpoint's own and not a number the engine reported about
    // itself (`memory_proof.rs:1-38` is explicit that the engine's own resident
    // figure is not a footprint).
    let index = kaplumba_engine::safetensors::read_index(&root.join("model.safetensors.index.json"))
        .expect("index");
    let selection =
        kaplumba_engine::safetensors::load_text_selection(root, index).expect("selection");
    let mut source = 0u64;
    let mut head = 0u64;
    let mut embed = 0u64;
    let mut layers = 0u64;
    for (name, info) in &selection.tensors {
        let n = info.byte_len();
        source += n;
        if name == "lm_head.trellis" {
            head += n;
        } else if name.ends_with("embed_tokens.weight") {
            embed += n;
        } else if name.contains(".trellis") {
            layers += n;
        }
    }

    println!("PROBE checkpoint            = {checkpoint}");
    println!("PROBE prompt_tokens         = {}", prompt_ids.len());
    println!("PROBE load_ms               = {load_ms}");
    println!("PROBE selection_bytes       = {source} ({:.3} GiB)", gib(source));
    println!("PROBE   layer_trellis       = {layers} ({:.3} GiB)", gib(layers));
    println!("PROBE   lm_head_trellis     = {head} ({:.3} GiB)", gib(head));
    println!("PROBE   embed_tokens        = {embed} ({:.3} GiB)", gib(embed));
    println!(
        "PROBE scratch_max_tokens    = {}",
        model.scratch_max_tokens()
    );
    println!(
        "PROBE scratch_resident_bytes= {} ({:.3} GiB)  [ForwardScratch::resident_bytes() omits raw_b and raw_c]",
        model.scratch_resident_bytes(),
        gib(model.scratch_resident_bytes())
    );

    let vm_before_prefill = host_vm();
    let r0 = rusage();
    println!(
        "PROBE LOAD     minflt={} majflt={} maj_GiB={:.3} host_pageins={} host_swapins={} host_compressions={} ms={}",
        r0.minflt - before_load_rusage.minflt,
        r0.majflt - before_load_rusage.majflt,
        gib((r0.majflt - before_load_rusage.majflt) * before_load.page_size),
        delta(vm_before_prefill.pageins, before_load.pageins),
        delta(vm_before_prefill.swapins, before_load.swapins),
        delta(vm_before_prefill.compressions, before_load.compressions),
        load_ms
    );
    println!(
        "PROBE LOAD_HOST free={:.3}GiB supply={:.3}GiB compressor={:.3}GiB wire={:.3}GiB pageouts={} swapouts={}",
        gib_delta(delta(vm_before_prefill.free, before_load.free), before_load.page_size),
        gib_delta(
            delta(
                vm_before_prefill.supply_bytes(),
                before_load.supply_bytes()
            ),
            before_load.page_size
        ),
        gib_delta(
            delta(vm_before_prefill.compressor, before_load.compressor),
            before_load.page_size
        ),
        gib_delta(
            delta(vm_before_prefill.wire, before_load.wire),
            before_load.page_size
        ),
        delta(vm_before_prefill.pageouts, before_load.pageouts),
        delta(vm_before_prefill.swapouts, before_load.swapouts),
    );
    println!(
        "PROBE HOST page_size={} page_total={} total_GiB={:.3} baseline_supply_GiB={:.3}",
        before_load.page_size,
        before_load.pages_total,
        gib(before_load.page_size * before_load.pages_total),
        gib(before_load.supply_bytes())
    );

    // --- prefill -----------------------------------------------------------
    let positions = (0..prompt_ids.len()).map(|p| p as u32).collect::<Vec<_>>();
    let t0 = Instant::now();
    let logits = model.forward_buffer(&prompt_ids, &positions).expect("prefill");
    let prefill_ms = t0.elapsed().as_millis();
    let r1 = rusage();
    let vm_after_prefill = host_vm();

    println!(
        "PROBE PREFILL ms={} tok_s={:.4} minflt={} majflt={} maj_GiB={:.3}",
        prefill_ms,
        prompt_ids.len() as f64 / (prefill_ms as f64 / 1000.0).max(1e-9),
        r1.minflt - r0.minflt,
        r1.majflt - r0.majflt,
        gib((r1.majflt - r0.majflt) * before_load.page_size)
    );
    println!(
        "PROBE PREFILL_HOST pageins={} pageouts={} swapins={} swapouts={} compressions={} decompressions={} free_GiB={:.3} compressor_GiB={:.3}",
        delta(vm_after_prefill.pageins, vm_before_prefill.pageins),
        delta(vm_after_prefill.pageouts, vm_before_prefill.pageouts),
        delta(vm_after_prefill.swapins, vm_before_prefill.swapins),
        delta(vm_after_prefill.swapouts, vm_before_prefill.swapouts),
        delta(vm_after_prefill.compressions, vm_before_prefill.compressions),
        delta(
            vm_after_prefill.decompressions,
            vm_before_prefill.decompressions
        ),
        gib_delta(
            delta(vm_after_prefill.free, vm_before_prefill.free),
            before_load.page_size
        ),
        gib_delta(
            delta(vm_after_prefill.compressor, vm_before_prefill.compressor),
            before_load.page_size
        ),
    );

    // --- per-token ---------------------------------------------------------
    //
    // The head runs once per step, so every step re-reads the same bytes the
    // next one will. If any of them are being served from swap, the step time
    // and the major-fault count both scale with it, and the ratio between
    // checkpoints is the answer.
    let mut argmax = argmax_of(&logits);
    let mut position = prompt_ids.len() as u32;
    let mut min_tot = 0u64;
    let mut maj_tot = 0u64;
    let mut ms_tot = 0u128;
    println!("PROBE STEP  ms      minflt   majflt   maj_MiB   tok_s");
    for step in 0..steps {
        let a = rusage();
        let t = Instant::now();
        let row = model
            .forward_buffer(&[argmax], &[position])
            .expect("decode step");
        let ms = t.elapsed().as_millis();
        let b = rusage();
        let dmin = b.minflt - a.minflt;
        let dmaj = b.majflt - a.majflt;
        min_tot += dmin;
        maj_tot += dmaj;
        ms_tot += ms as u128;
        println!(
            "PROBE {step:>4}  {ms:>6}  {dmin:>7}  {dmaj:>7}  {:>8.1}  {:>7.4}",
            mib(dmaj * before_load.page_size),
            1000.0 / (ms as f64).max(1e-9)
        );
        argmax = argmax_of(&row);
        position += 1;
    }
    let vm_after = host_vm();
    println!(
        "PROBE DECODE ms={} minflt={} majflt={} tok_s={:.4}",
        ms_tot,
        min_tot,
        maj_tot,
        steps as f64 / (ms_tot as f64 / 1000.0).max(1e-9)
    );
    println!(
        "PROBE PER_TOKEN ms={:.1} minflt={:.1} majflt={:.1} maj_MiB_per_token={:.1}",
        ms_tot as f64 / steps as f64,
        min_tot as f64 / steps as f64,
        maj_tot as f64 / steps as f64,
        mib(maj_tot * before_load.page_size) / steps as f64
    );
    println!(
        "PROBE DECODE_HOST pageins={} pageouts={} swapins={} swapouts={} compressions={} decompressions={}",
        delta(vm_after.pageins, vm_after_prefill.pageins),
        delta(vm_after.pageouts, vm_after_prefill.pageouts),
        delta(vm_after.swapins, vm_after_prefill.swapins),
        delta(vm_after.swapouts, vm_after_prefill.swapouts),
        delta(vm_after.compressions, vm_after_prefill.compressions),
        delta(vm_after.decompressions, vm_after_prefill.decompressions),
    );
    println!(
        "PROBE END free_GiB={:.3} supply_GiB={:.3} compressor_GiB={:.3} wire_GiB={:.3}",
        gib(vm_after.free * vm_after.page_size),
        gib(vm_after.supply_bytes()),
        gib(vm_after.compressor * vm_after.page_size),
        gib(vm_after.wire * vm_after.page_size),
    );

    // Keep the model alive to the end so nothing is dropped before the last
    // host reading.
    drop(model);
}

fn argmax_of(row: &[u16]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}