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
        is_mapped(&weights, HEAD),
        "{HEAD} (1.06 GB) should be mapped: measured -1.00 GiB, output unchanged"
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

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn resident_footprint_fits_the_twelve_gib_budget() {
    let model = TextModel::load_from_dir(std::path::Path::new(
        &std::env::var("KAPLUMBA_TEST_MODEL").unwrap(),
    ))
    .unwrap();
    let (weights, kv, scratch) = model.allocation_breakdown();
    let gib = |b: u64| b as f64 / 1_073_741_824.0;
    println!(
        "RESIDENCY weights={:.2} GiB kv={:.2} GiB scratch={:.3} GiB total={:.2} GiB",
        gib(weights),
        gib(kv),
        gib(scratch),
        gib(model.allocated_bytes())
    );
    // The mapped tensors must not reappear in the engine's own accounting.
    assert!(
        weights < 11_500_000_000,
        "resident weights {:.2} GiB suggest the offload stopped working",
        gib(weights)
    );
}
