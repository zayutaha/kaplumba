//! PROBE (measurement only, `probe_offload_`): where do the bytes of a
//! checkpoint actually go?
//!
//! Three questions, three levels of cost:
//!
//!   1. `probe_offload_header_ledger`  -- headers only. No GPU, no mmap, no
//!      allocation. Answers "what could be offloaded" from the file alone.
//!   2. `probe_offload_weight_residency` -- one `TextWeights::load_dedicated`
//!      with an unlimited arena. Answers "what IS the resident/mapped
//!      partition" plus per-file-range residency via `mincore(2)`.
//!   3. `probe_offload_decode_residency` -- the full `TextModel` plus a few
//!      decode steps. Answers "how much of `embed_tokens` does decode touch".
//!
//! HOW THE RESIDENCY NUMBER IS OBTAINED, and why it is the number we want.
//!
//! `mincore(2)` on a `PROT_READ`/`MAP_PRIVATE` *file* mapping reports whether
//! each page of that file is in memory. macOS has one unified buffer cache, so
//! residency is a property of the **file page**, not of any particular mapping:
//! a second mapping of the same shard observes exactly the pages the engine's
//! mapping has faulted in. So this probe opens its own read-only view of each
//! shard and mincores it, and reads nothing through it (mapping does not fault;
//! only touching a page does). The `mmap` here never dirties a page and never
//! copies, so it cannot change what it measures.
//!
//! HOW THIS COULD BE WRONG: if the engine had copied a page out of the mapping
//! into anonymous memory, that anonymous copy would NOT show up here. That is
//! the point of the probe, not a flaw in it -- but it does mean this measures
//! *file-backed* residency, and the anonymous side must be accounted separately
//! (via `allocation_breakdown` and `/usr/bin/time -l`).
//!
//! A negative control is included: the vision-tower byte range of shard 3 is
//! never read by text-only inference, so it must report ~0 resident. If it does
//! not, `mincore` is reporting something other than what we think it is.

use kaplumba_engine::metal::MetalContext;
use kaplumba_engine::model::TextModel;
use kaplumba_engine::safetensors::{self, ShardHeader};
use kaplumba_engine::weights::TextWeights;

const EMBED: &str = "model.language_model.embed_tokens.weight";
const HEAD: &str = "lm_head.trellis";

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000.0
}

fn model_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("KAPLUMBA_TEST_MODEL")
            .expect("set KAPLUMBA_TEST_MODEL to the checkpoint directory"),
    )
}

/// Resident bytes inside `[start, end)` of an already-mapped file.
struct Residency;

impl Residency {
    fn in_ranges(ptr: *const u8, len: usize, ranges: &[(u64, u64, String)]) -> Vec<(u64, u64)> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let pages = (len as u64).div_ceil(page);
        let mut bits = vec![0u8; pages as usize];
        let rc = unsafe {
            libc::mincore(
                ptr as *mut libc::c_void,
                len,
                bits.as_mut_ptr() as *mut libc::c_char,
            )
        };
        assert_eq!(rc, 0, "mincore failed: {}", std::io::Error::last_os_error());
        let mut out = Vec::new();
        for (start, end, _) in ranges {
            // Pages that *overlap* the range, so a tensor sharing its first or
            // last page with a neighbour is not silently truncated.
            let first = (start / page) as usize;
            let last = (end.saturating_sub(1) / page) as usize;
            let mut resident = 0_u64;
            for page_index in first..=last {
                if page_index < bits.len() && bits[page_index] & 1 == 1 {
                    resident += page;
                }
            }
            let span = end - start;
            out.push((span, resident.min(span)));
        }
        out
    }
}

/// A read-only view of one shard, used only for `mincore`.
struct Probe {
    ptr: *mut libc::c_void,
    len: usize,
    path: String,
}

impl Probe {
    fn open(path: &std::path::Path) -> Self {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path).unwrap();
        let len = file.metadata().unwrap().len() as usize;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap {}", path.display());
        Self {
            ptr,
            len,
            path: path.display().to_string(),
        }
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

/// Resident bytes of one named tensor, by mincore-ing a fresh read-only view of
/// its shard.
fn resident_of(
    selection: &safetensors::TextTensorSelection,
    root: &std::path::Path,
    name: &str,
) -> u64 {
    let shard_name = selection.shard_for(name).expect("shard");
    let probe = Probe::open(&root.join(shard_name));
    let (start, end) = selection.shards[shard_name]
        .absolute_range(name)
        .expect("range");
    Residency::in_ranges(probe.ptr as *const u8, probe.len, &[(start, end, name.to_string())])[0].1
}

/// Header-only ledger: every text tensor's shard, absolute file offset, size,
/// dtype and 4-byte alignment. No GPU, no mmap, no allocation.
#[test]
fn probe_offload_header_ledger() {
    let root = model_dir();
    let index = safetensors::read_index(&root.join("model.safetensors.index.json")).unwrap();
    let selection = safetensors::load_text_selection(&root, index).unwrap();

    println!("LEDGER dir={}", root.display());
    println!(
        "LEDGER text_bytes = {} ({:.3} GiB)",
        selection.text_bytes,
        gib(selection.text_bytes)
    );

    let mut trellis = 0_u64;
    let mut sign = 0_u64;
    let mut other = 0_u64;
    let mut misaligned: Vec<(String, u64, u64)> = Vec::new();
    for (name, info) in &selection.tensors {
        let shard_name = selection.shard_for(name).expect("shard");
        let shard: &ShardHeader = &selection.shards[shard_name];
        let (start, _end) = shard.absolute_range(name).expect("range");
        let len = info.byte_len();
        if name.ends_with(".trellis") {
            trellis += len;
            if start % 4 != 0 {
                misaligned.push((name.clone(), start, len));
            }
        } else if name.ends_with(".suh")
            || name.ends_with(".svh")
            || name.ends_with(".mcg")
            || name.ends_with(".su")
            || name.ends_with(".sv")
        {
            sign += len;
        } else {
            other += len;
        }
    }
    println!(
        "LEDGER trellis = {} ({:.3} GiB), signs = {} ({:.3} MiB), other = {} ({:.3} GiB)",
        trellis,
        gib(trellis),
        sign,
        mb(sign),
        other,
        gib(other)
    );
    println!(
        "LEDGER text tensors whose absolute file offset is NOT 4-byte aligned: {}",
        misaligned.len()
    );
    for (name, start, len) in &misaligned {
        println!(
            "LEDGER   MISALIGNED {name}: abs_off={start} off%4={} off%4096={} len={len} \
             ({:.3} GiB)",
            start % 4,
            start % 4096,
            gib(*len)
        );
    }
    for (shard_name, shard) in &selection.shards {
        println!(
            "LEDGER shard {shard_name}: header_len={} payload_start={} payload_start%4096={} \
             file_size={}",
            shard.header_len,
            shard.payload_start(),
            shard.payload_start() % 4096,
            shard.file_size
        );
    }
    for name in [EMBED, HEAD] {
        let info = selection.tensors.get(name).expect("tensor");
        let shard_name = selection.shard_for(name).expect("shard");
        let shard = &selection.shards[shard_name];
        let (start, end) = shard.absolute_range(name).expect("range");
        println!(
            "LEDGER {name}: shard={shard_name} payload_start={} abs_off={start} abs_end={end} \
             len={} ({:.3} GiB) dtype={:?} off%4={} off%4096={}",
            shard.payload_start(),
            info.byte_len(),
            gib(info.byte_len()),
            info.dtype,
            start % 4,
            start % 4096,
        );
    }

    // The partition the *default* loader produces, derived from the code:
    //   skip_copy = offload_bf16_embedding || (!converted && (zero_copy || head_trellis))
    // so exactly two text tensors are mapped and everything else is arena.
    let embed = selection.tensors[EMBED].byte_len();
    let head = selection.tensors[HEAD].byte_len();
    println!(
        "LEDGER EXPECTED default split: mapped = embed({embed}) + head({head}) = {} \
         ({:.3} GiB); arena payload = {} ({:.3} GiB)",
        embed + head,
        gib(embed + head),
        selection.text_bytes - embed - head,
        gib(selection.text_bytes - embed - head)
    );
    assert_eq!(
        embed + head + (selection.text_bytes - embed - head),
        selection.text_bytes
    );
}

/// One unlimited-arena weight load. Prints the exact resident/mapped partition
/// and the per-range file residency of every shard.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the full 4-bpw arena"]
fn probe_offload_weight_residency() {
    let root = model_dir();
    let index = safetensors::read_index(&root.join("model.safetensors.index.json")).unwrap();
    let selection = safetensors::load_text_selection(&root, index).unwrap();

    let shard_names: Vec<String> = selection.shards.keys().cloned().collect();
    let mut probes: Vec<Probe> = Vec::new();
    let mut ranges: Vec<Vec<(u64, u64, String)>> = Vec::new();
    for shard_name in &shard_names {
        probes.push(Probe::open(&root.join(shard_name)));
        ranges.push(Vec::new());
    }
    let slot = |shard_name: &str| shard_names.iter().position(|s| s == shard_name).unwrap();

    for (name, _) in &selection.tensors {
        let shard_name = selection.shard_for(name).expect("shard").to_string();
        let index = slot(&shard_name);
        let shard = &selection.shards[&shard_name];
        let (start, end) = shard.absolute_range(name).expect("range");
        let label = if name == EMBED || name == HEAD {
            format!("OFFLOADED {name}")
        } else {
            format!("text {name}")
        };
        ranges[index].push((start, end, label));
    }
    // Negative control: the vision tower is excluded from the text selection, so
    // nothing in a text-only load ever reads it.
    for (shard_name, shard) in &selection.shards {
        let index = slot(shard_name);
        let mut total = 0_u64;
        for (name, _) in &shard.tensors {
            if !(name.starts_with("model.vision_tower") || name.starts_with("model.embed_vision")) {
                continue;
            }
            let (start, end) = shard.absolute_range(name).expect("range");
            ranges[index].push((start, end, "NEG-CONTROL vision".to_string()));
            total += end - start;
        }
        if total > 0 {
            println!(
                "CONTROL {shard_name}: vision range = {total} ({:.3} GiB)",
                gib(total)
            );
        }
    }

    let census = |probes: &Vec<Probe>, ranges: &Vec<Vec<(u64, u64, String)>>| {
        ranges
            .iter()
            .enumerate()
            .flat_map(|(i, r)| {
                Residency::in_ranges(probes[i].ptr as *const u8, probes[i].len, r)
                    .into_iter()
                    .zip(r.iter())
                    .map(|((span, resident), (_, _, label))| (label.clone(), span, resident))
            })
            .collect::<Vec<_>>()
    };

    let before = census(&probes, &ranges);
    let mut embed_before = 0_u64;
    for (label, span, resident) in &before {
        if label == &format!("OFFLOADED {EMBED}") {
            embed_before = *resident;
        }
    }

    let context = MetalContext::new().unwrap();
    let weights = TextWeights::load_dedicated(&root, &selection, context, u64::MAX).unwrap();

    let mut mapped_names = Vec::new();
    let mut mapped_bytes = 0_u64;
    let mut arena_payload = 0_u64;
    for (name, info) in &selection.tensors {
        let tensor = weights.tensor(name).unwrap_or_else(|| panic!("{name}"));
        if tensor.spans.is_empty() {
            mapped_names.push(name.as_str());
            mapped_bytes += info.byte_len();
        } else {
            arena_payload += info.byte_len();
        }
    }
    mapped_names.sort_unstable();
    println!("SPLIT mapped_names = {mapped_names:?}");
    println!(
        "SPLIT text_bytes     = {} ({:.3} GiB)",
        selection.text_bytes,
        gib(selection.text_bytes)
    );
    println!(
        "SPLIT mapped_bytes   = {mapped_bytes} ({:.3} GiB)",
        gib(mapped_bytes)
    );
    println!(
        "SPLIT arena_payload  = {arena_payload} ({:.3} GiB)",
        gib(arena_payload)
    );
    println!(
        "SPLIT arena_capacity = {} ({:.3} GiB)",
        weights.resident_bytes(),
        gib(weights.resident_bytes())
    );
    println!(
        "SPLIT sums_to_text   = {}",
        mapped_bytes + arena_payload == selection.text_bytes
    );
    println!(
        "SPLIT head_len       = {} ({:.3} GiB); its u16 element count = {} ({:.1} MB)",
        selection.tensors[HEAD].byte_len(),
        gib(selection.tensors[HEAD].byte_len()),
        selection.tensors[HEAD].byte_len() / 2,
        mb(selection.tensors[HEAD].byte_len() / 2)
    );

    let after = census(&probes, &ranges);
    for ((label, span, resident), (_, _, before_resident)) in after.iter().zip(before.iter()) {
        if label.starts_with("OFFLOADED") || label.starts_with("NEG-CONTROL") {
            println!(
                "RESIDENT {label}: span={span} resident_after_load={resident} \
                 ({:.3} MiB) resident_before_load={before_resident}",
                *resident as f64 / 1_048_576.0
            );
        }
    }
    let embed_after = after
        .iter()
        .find(|(label, _, _)| label == &format!("OFFLOADED {EMBED}"))
        .map(|(_, span, resident)| (*span, *resident))
        .unwrap();
    println!(
        "RESIDENT embed_tokens: BEFORE load {embed_before} of {}, AFTER load {} of {} ({:.3} MiB, \
         {:.4}% of the table)",
        embed_after.0,
        embed_after.1,
        embed_after.0,
        embed_after.1 as f64 / 1_048_576.0,
        100.0 * embed_after.1 as f64 / embed_after.0.max(1) as f64
    );
}

/// Full `TextModel` load plus a handful of decode steps, then the same
/// `mincore` census. This is where `embed_tokens` residency is decided.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; full model load + decode"]
fn probe_offload_decode_residency() {
    let root = model_dir();
    let index = safetensors::read_index(&root.join("model.safetensors.index.json")).unwrap();
    let selection = safetensors::load_text_selection(&root, index).unwrap();
    let embed_len = selection.tensors[EMBED].byte_len();
    let head_len = selection.tensors[HEAD].byte_len();

    let embed_before = resident_of(&selection, &root, EMBED);
    let head_before = resident_of(&selection, &root, HEAD);
    println!(
        "FULL BEFORE load: embed resident {embed_before} of {embed_len} ({:.4}%); \
         lm_head.trellis resident {head_before} of {head_len} ({:.2}%)",
        100.0 * embed_before as f64 / embed_len as f64,
        100.0 * head_before as f64 / head_len as f64
    );

    let mut model = TextModel::load_from_dir(&root).unwrap();
    let (weights, kv, scratch) = model.allocation_breakdown();
    println!(
        "FULL weights_reported = {weights} ({:.3} GiB)",
        gib(weights)
    );
    println!("FULL kv_reported      = {kv} ({:.3} GiB)", gib(kv));
    println!(
        "FULL scratch_reported = {scratch} ({:.3} GiB)",
        gib(scratch)
    );
    println!(
        "FULL allocated_bytes  = {} ({:.3} GiB)",
        model.allocated_bytes(),
        gib(model.allocated_bytes())
    );

    let embed_after_load = resident_of(&selection, &root, EMBED);
    let head_after_load = resident_of(&selection, &root, HEAD);
    println!(
        "FULL after load:  embed resident {embed_after_load} of {embed_len} ({:.4}%); \
         lm_head.trellis resident {head_after_load} of {head_len} ({:.2}%)",
        100.0 * embed_after_load as f64 / embed_len as f64,
        100.0 * head_after_load as f64 / head_len as f64
    );

    // 6 prefill tokens then `steps` single-token decode steps, through the
    // *GPU* path (`forward_buffer_into_vec`), which is what `generate_gpu` --
    // the default in `crates/cli/src/main.rs` -- calls.
    //
    // NOT `TextModel::forward`: that path goes through
    // `ElementKernels::embedding_scaled_buffer`, which requires
    // `embedding.length() == hidden_size * vocab_size * 2` exactly, and a
    // zero-copy view is a page-aligned window that is LONGER than the tensor.
    // With `embed_tokens` mapped -- the default configuration -- `forward`
    // therefore always fails with
    //   Kernel(InvalidInput("embedding buffer length does not match dimensions"))
    // and so does `generate` behind `--legacy-path`. Recorded, not fixed: this
    // probe is not allowed to touch production code.
    let prompt: Vec<u32> = vec![2, 105, 1159, 1205, 255, 235_248];
    let steps: u32 = std::env::var("KAPLUMBA_PROBE_STEPS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(1);
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let mut logits = Vec::new();
    model
        .forward_buffer_into_vec(&prompt, &positions, &mut logits)
        .unwrap();
    println!(
        "PREFILL {} tokens -> {} logits",
        prompt.len(),
        logits.len()
    );

    let embed_after_prefill = resident_of(&selection, &root, EMBED);
    println!(
        "FULL embed_tokens resident after prefill = {embed_after_prefill} of {embed_len} \
         ({:.4}%, {:.3} MiB)",
        100.0 * embed_after_prefill as f64 / embed_len as f64,
        embed_after_prefill as f64 / 1_048_576.0
    );

    // POSITIVE CONTROL. Everything above is only meaningful if `mincore` can see
    // `embed_tokens` residency at all. Pull 4096 known rows through the file and
    // re-measure: the count must go up by approximately that many bytes, or the
    // numbers above are a tool failure and not a fact about the model.
    //
    // 4096 rows x 5376 bf16 = 44,040,192 bytes.
    let rows = 4096_u64;
    let row_bytes = 5376 * 2;
    let want = rows * row_bytes;
    let shard_name = selection.shard_for(EMBED).expect("shard");
    let (embed_start, _) = selection.shards[shard_name]
        .absolute_range(EMBED)
        .expect("range");
    {
        use std::os::unix::fs::FileExt;
        let file = std::fs::File::open(root.join(shard_name)).unwrap();
        let mut offset = embed_start;
        let mut buf = vec![0_u8; row_bytes as usize];
        for _ in 0..rows {
            file.read_exact_at(&mut buf, offset).unwrap();
            offset += row_bytes;
        }
    }
    let embed_control = resident_of(&selection, &root, EMBED);
    println!(
        "CONTROL positive: read {rows} rows ({want} bytes) of {EMBED} through the file; \
         resident went {embed_after_prefill} -> {embed_control} (+{} bytes, expected \
         ~{want})",
        embed_control.saturating_sub(embed_after_prefill) as i64
    );
    assert!(
        embed_control >= embed_after_prefill + want - 2 * 16384,
        "positive control failed: mincore did not see {} bytes of embed residency after \
         they were read ({} -> {})",
        want,
        embed_after_prefill,
        embed_control
    );

    let mut token = logits.first().copied().unwrap_or(0) as u32;
    for step in 0..steps {
        let positions = vec![prompt.len() as u32 + step];
        model
            .forward_buffer_into_vec(&[token], &positions, &mut logits)
            .unwrap();
        token = logits.first().copied().unwrap_or(0) as u32;
    }
    println!("DECODE {steps} single-token steps done, last argmax-ish {token}");

    let embed_after = resident_of(&selection, &root, EMBED);
    let head_after = resident_of(&selection, &root, HEAD);
    println!(
        "FULL embed_tokens resident: before_load={embed_before} after_load={embed_after_load} \
         after_prefill={embed_after_prefill} after_{steps}_decode={embed_after} of {embed_len} \
         ({:.4}% of the table, {:.3} MiB)",
        100.0 * embed_after as f64 / embed_len as f64,
        embed_after as f64 / 1_048_576.0
    );
    println!(
        "FULL lm_head.trellis resident: before_load={head_before} after_load={head_after_load} \
         after_decode={head_after} of {head_len} ({:.2}% of the tensor, {:.3} GiB)",
        100.0 * head_after as f64 / head_len as f64,
        gib(head_after)
    );

    println!(
        "FULL reported vs real: allocated_bytes={} ({:.3} GiB) but the text tensors alone are \
         {} ({:.3} GiB); the difference is the anonymous lm_head realignment copy",
        model.allocated_bytes(),
        gib(model.allocated_bytes()),
        selection.text_bytes,
        gib(selection.text_bytes)
    );
}