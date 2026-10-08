//! Which tensors are resident, and which are mapped.
//!
//! The resident/mapped split is a deliberate performance decision, not an
//! implementation detail, so it is pinned here. Getting it wrong is silent and
//! expensive in both directions:
//!
//!   * Mapping the 11 GB of layer weights lets the OS evict them. Decode streams
//!     every one of them on every token, so the model falls back to disk and swap:
//!     measured 3.9 GiB of swap and ~1.4 tok/s.
//!   * Copying `embed_tokens` or `lm_head.trellis` into the arena costs 3.7 GiB
//!     for data that is either read one row at a time or is small enough to stay
//!     hot in the file cache.
//!
//! Both mistakes produce plausible-looking output, so only a test will catch a
//! regression.

use kaplumba_engine::metal::MetalContext;
use kaplumba_engine::model::TextModel;
use kaplumba_engine::safetensors;
use kaplumba_engine::weights::TextWeights;

const EMBED: &str = "model.language_model.embed_tokens.weight";
const HEAD: &str = "lm_head.trellis";
const LAYER_Q: &str = "model.language_model.layers.0.self_attn.q_proj.trellis";
const LAYER_DOWN: &str = "model.language_model.layers.30.mlp.down_proj.trellis";
/// Set by the escape-hatch test when it re-executes this binary.
const CHILD: &str = "KAPLUMBA_RESIDENCY_CHILD";

fn load() -> TextWeights {
    let dir = std::env::var("KAPLUMBA_TEST_MODEL")
        .expect("set KAPLUMBA_TEST_MODEL to the Gemma 4 checkpoint directory");
    let root = std::path::Path::new(&dir);
    let index = safetensors::read_index(&root.join("model.safetensors.index.json")).unwrap();
    let selection = safetensors::load_text_selection(root, index).unwrap();
    let context = MetalContext::new().unwrap();
    TextWeights::load_dedicated(root, &selection, context, u64::MAX).unwrap()
}

fn is_mapped(weights: &TextWeights, name: &str) -> bool {
    weights
        .tensor(name)
        .unwrap_or_else(|| panic!("{name} is missing"))
        .spans
        .is_empty()
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn layer_weights_stay_resident() {
    let weights = load();
    for name in [LAYER_Q, LAYER_DOWN] {
        assert!(
            !is_mapped(&weights, name),
            "{name} must be resident: decode streams it every token, and mapping it \
             caused 3.9 GiB of swap at ~1.4 tok/s"
        );
    }
    // Every layer trellis, not just two samples.
    let mapped: Vec<&String> = weights
        .tensors()
        .keys()
        .filter(|name| {
            name.starts_with("model.language_model.layers.") && name.ends_with(".trellis")
        })
        .filter(|name| is_mapped(&weights, name))
        .collect();
    assert!(
        mapped.is_empty(),
        "{} layer trellis tensors are mapped but must be resident, e.g. {:?}",
        mapped.len(),
        mapped.iter().take(3).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn bulk_read_mostly_tensors_are_mapped_by_default() {
    let weights = load();
    assert!(
        is_mapped(&weights, EMBED),
        "{EMBED} (2.82 GB) should be mapped: one row is read per token"
    );
    assert!(
        !is_mapped(&weights, HEAD),
        "{HEAD} (1.06 GB) is resident in the shared arena to prevent NVMe eviction under memory pressure"
    );
}

/// `KAPLUMBA_RESIDENT_WEIGHTS=1` must put everything back in the arena.
///
/// The flag is read while weights load, so it cannot be toggled inside a running
/// process. This re-executes the test binary with the flag set and lets a child
/// test do the asserting.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn resident_weights_escape_hatch_puts_everything_in_the_arena() {
    // asserts via a child process; see `child_resident_weights_are_in_the_arena`
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "child_resident_weights_are_in_the_arena",
        ])
        .env(CHILD, "1")
        .env("KAPLUMBA_RESIDENT_WEIGHTS", "1")
        .status()
        .expect("failed to re-exec the test binary");
    assert!(status.success(), "escape-hatch child test failed");
}

/// Child half of [`resident_weights_escape_hatch_puts_everything_in_the_arena`].
///
/// Deliberately *not* `#[ignore]`d so that `--ignored` (which the parent uses to
/// select the child) filters it out of the parent's own run. It is a no-op
/// unless the parent invoked it.
#[test]
fn child_resident_weights_are_in_the_arena() {
    if std::env::var(CHILD).is_err() {
        return;
    }
    let weights = load();
    for name in [EMBED, HEAD, LAYER_Q] {
        assert!(
            !is_mapped(&weights, name),
            "{name} is still mapped with KAPLUMBA_RESIDENT_WEIGHTS=1"
        );
    }
}

/// The resident/mapped split, as byte counts, so the invariants can be checked
/// without a checkpoint.
///
/// This used to be `assert!(weights < 11_500_000_000, ...)`, a snapshot of one
/// configuration: 3 bpw passes it with 499 MB to spare and 4 bpw fails it,
/// while `memory_proof.rs` asserted the exact opposite (`weights >
/// 14_000_000_000`) and 4 bpw passed that one. The two assertions are mutually
/// exclusive at every configuration, and the two checkpoints in this repo are
/// the two sides of the swap cliff, so each assertion had been "passing"
/// exactly once -- against whichever run was done last. What the loader
/// actually guarantees is a *partition* and a *relationship*, and that holds for
/// every checkpoint and every environment flag.
#[derive(Debug, Clone, Copy)]
struct Split {
    /// Text-tensor bytes that are mapped out of the shard and not in the arena.
    mapped_bytes: u64,
    /// Text-tensor bytes that were allocated in the arena, at *source* length.
    arena_payload: u64,
    /// Every text byte in the checkpoint. The vision tower is excluded upstream
    /// (`safetensors.rs:469-470`), so this is not the file size.
    text_bytes: u64,
    /// `TextWeights::resident_bytes()` = arena `capacity_bytes`.
    arena_capacity: u64,
}

/// Names that must be mapped rather than arena-resident, and under what
/// conditions.
///
/// `KAPLUMBA_RESIDENT_WEIGHTS=1` deliberately puts everything in the arena
/// (the escape hatch above), and `KAPLUMBA_ZERO_COPY_WEIGHTS` maps everything.
/// Neither is a default configuration, so the "the offload must be doing work"
/// assertions only apply when neither flag is set.
fn offload_active() -> bool {
    std::env::var_os("KAPLUMBA_RESIDENT_WEIGHTS").is_none()
        && std::env::var_os("KAPLUMBA_ZERO_COPY_WEIGHTS").is_none()
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

/// The invariants, as a function so they can be exercised on recorded byte
/// counts without loading anything (`splits_add_up_for_both_known_checkpoints`).
fn check_split(split: &Split, mapped_names: &[&str], offload: bool) {
    // 1. The partition is exact and total: every text byte is either mapped out
    //    of the shard or allocated in the arena, and nothing is both or neither.
    //    This is the invariant that actually decides the memory budget, and it
    //    cannot drift with the checkpoint.
    assert_eq!(
        split.mapped_bytes + split.arena_payload,
        split.text_bytes,
        "mapped {} + arena payload {} != text_bytes {}: a text tensor is counted \
         twice, or not at all",
        gib(split.mapped_bytes),
        gib(split.arena_payload),
        gib(split.text_bytes),
    );

    // 2. The arena cannot be holding bytes that are accounted for as mapped
    //    somewhere else -- the thing the old `< 11_500_000_000` was trying to
    //    say. `allocate_dedicated` (weights.rs:200-235) allocates one
    //    exactly-sized buffer per tensor, so capacity can only be *smaller* than
    //    the source payload (bf16 -> fp16 halves the converted tensors), never
    //    larger.
    assert!(
        split.arena_capacity <= split.arena_payload,
        "arena capacity {:.2} GiB exceeds its {2} source payload of {:.2} GiB: \
         something mapped is also resident",
        gib(split.arena_capacity),
        split.mapped_bytes,
        gib(split.arena_payload),
    );

    // 3. ...and it cannot be absurdly small either: the only transformation at
    //    load is bf16 -> fp16, which halves, so capacity >= payload / 2. This is
    //    the floor that would catch an arena that silently allocated nothing.
    assert!(
        split.arena_capacity * 2 >= split.arena_payload,
        "arena capacity {:.2} GiB is under half of its {:.2} GiB payload",
        gib(split.arena_capacity),
        gib(split.arena_payload),
    );

    if !offload {
        return;
    }
    // 4. Under the default configuration the offload must be doing the work it
    //    claims, and doing it on exactly the two tensors it names. `mapped_names`
    //    is the exhaustive set, so this catches the offload stopping *and* a
    //    third tensor being mapped by accident (which is how the layer trellis
    //    reached the 1.4 tok/s configuration).
    let mut expected = vec![EMBED, HEAD];
    expected.sort_unstable();
    let mut got = mapped_names.to_vec();
    got.sort_unstable();
    assert_eq!(
        got, expected,
        "the mapped set must be exactly the offloaded tensors: {:?} != {:?}",
        got, expected,
    );
    assert!(
        split.mapped_bytes * 3 < split.text_bytes,
        "only {:.2} GiB of {:.2} GiB of text is mapped; the offload is not doing \
         the work it exists to do",
        gib(split.mapped_bytes),
        gib(split.text_bytes),
    );
}

/// The invariant, checked against both checkpoints in `~/.kaplumba/models`.
///
/// These byte counts are the ledger in
/// `spec/memory-swap-cliff/implementation_notes.md` §0, read from the
/// safetensors headers: 3 bpw (`Gxma-repacked`) and 4 bpw (`Gxma-4-bit`). The
/// point of this test is that the *same* assertions satisfy both -- which is
/// exactly what the two mutually exclusive thresholds could not do.
#[test]
fn splits_add_up_for_both_known_checkpoints() {
    struct Recorded {
        label: &'static str,
        split: Split,
    }
    let embed = 2_818_572_288_u64;
    for recorded in [
        Recorded {
            label: "Gxma-repacked, 3 bpw",
            split: Split {
                mapped_bytes: embed + 1_056_964_608,
                arena_payload: 14_876_563_172 - embed - 1_056_964_608,
                text_bytes: 14_876_563_172,
                arena_capacity: 11_001_026_276,
            },
        },
        Recorded {
            label: "Gxma-4-bit, 4 bpw",
            split: Split {
                mapped_bytes: embed + 1_409_286_144,
                arena_payload: 18_889_725_668 - embed - 1_409_286_144,
                text_bytes: 18_889_725_668,
                arena_capacity: 14_661_867_236,
            },
        },
    ] {
        check_split(
            &recorded.split,
            &[EMBED, HEAD],
            true,
        );
        println!(
            "SPLIT {}: text {:.2} GiB, mapped {:.2} GiB, arena capacity {:.2} GiB",
            recorded.label,
            gib(recorded.split.text_bytes),
            gib(recorded.split.mapped_bytes),
            gib(recorded.split.arena_capacity),
        );
    }
}

/// The invariants have teeth: an offload that stopped working, a layer tensor
/// that got mapped, and an arena that started holding mapped bytes are all
/// rejected. (AGENTS.md §3.4 -- these are the failure modes the assertions above
/// exist to catch, so they are themselves tested.)
#[test]
fn the_split_checks_reject_the_three_failures_they_exist_for() {
    let good = Split {
        mapped_bytes: 2_818_572_288 + 1_056_964_608,
        arena_payload: 14_876_563_172 - 2_818_572_288 - 1_056_964_608,
        text_bytes: 14_876_563_172,
        arena_capacity: 11_001_026_276,
    };
    check_split(&good, &[EMBED, HEAD], true);

    // The offload stopped: nothing is mapped, so the arena holds everything.
    let offload_stopped = Split {
        mapped_bytes: 0,
        arena_payload: good.text_bytes,
        text_bytes: good.text_bytes,
        arena_capacity: good.text_bytes,
    };
    assert!(
        std::panic::catch_unwind(|| check_split(&offload_stopped, &[], true)).is_err(),
        "an arena holding the whole checkpoint must be rejected"
    );

    // The offload stopped *silently* -- the bytes still add up, but the
    // quantities are the wrong ones. Only the mapped-set assertion sees it.
    let silently_resident = Split {
        mapped_bytes: 0,
        arena_payload: good.text_bytes,
        text_bytes: good.text_bytes,
        arena_capacity: 11_001_026_276 + 2_818_572_288 + 1_056_964_608,
    };
    assert!(
        std::panic::catch_unwind(|| check_split(&silently_resident, &[], true)).is_err(),
        "an arena holding the mapped tensors must be rejected by the capacity bound"
    );

    // A layer trellis tensor joined the mapped set: the partition still adds up
    // and the capacity bound still holds, so only the exhaustive mapped-set
    // assertion can see this -- and it is the one that predicts the 1.4 tok/s
    // collapse.
    let layer_mapped = Split {
        mapped_bytes: good.mapped_bytes + 137_625_856,
        arena_payload: good.arena_payload - 137_625_856,
        text_bytes: good.text_bytes,
        arena_capacity: good.arena_capacity - 137_625_856,
    };
    assert!(
        std::panic::catch_unwind(|| check_split(&layer_mapped, &[EMBED, HEAD, LAYER_Q], true))
            .is_err(),
        "a mapped layer tensor must be rejected"
    );

    // The escape hatch turns the offload off on purpose, and must stay legal.
    check_split(&good, &[], false);
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn resident_and_mapped_weights_add_up_to_the_checkpoint() {
    let path = std::env::var("KAPLUMBA_TEST_MODEL").unwrap();
    let root = std::path::Path::new(&path);
    let index = safetensors::read_index(&root.join("model.safetensors.index.json")).unwrap();
    let selection = safetensors::load_text_selection(root, index).unwrap();
    let context = MetalContext::new().unwrap();
    let weights =
        TextWeights::load_dedicated(root, &selection, context, u64::MAX).unwrap();

    let mut mapped_names = Vec::new();
    let mut mapped_bytes = 0_u64;
    let mut arena_payload = 0_u64;
    for (name, info) in &selection.tensors {
        let tensor = weights
            .tensor(name)
            .unwrap_or_else(|| panic!("{name} is in the selection but not loaded"));
        if tensor.spans.is_empty() {
            mapped_names.push(name.as_str());
            mapped_bytes += info.byte_len();
        } else {
            arena_payload += info.byte_len();
        }
    }
    let split = Split {
        mapped_bytes,
        arena_payload,
        text_bytes: selection.text_bytes,
        arena_capacity: weights.resident_bytes(),
    };
    let model = TextModel::load_from_dir(root).unwrap();
    let (reported, kv, scratch) = model.allocation_breakdown();
    println!(
        "RESIDENCY text={:.2} GiB mapped={:.2} GiB arena={:.2} GiB kv={:.2} GiB \
         scratch={:.3} GiB total={:.2} GiB",
        gib(split.text_bytes),
        gib(split.mapped_bytes),
        gib(reported),
        gib(kv),
        gib(scratch),
        gib(model.allocated_bytes()),
    );
    assert_eq!(
        reported, split.arena_capacity,
        "allocation_breakdown's weights must be TextWeights::resident_bytes()",
    );
    check_split(&split, &mapped_names, offload_active());
}
