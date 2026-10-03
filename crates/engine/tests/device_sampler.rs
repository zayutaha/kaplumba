//! Device-side sampler: the correctness properties of `spec/device-sampler`.
//!
//! Everything here loads the real model, so every test is `#[ignore]`:
//! ~12 GiB resident, one process at a time (AGENTS.md §10).
//!
//! ```text
//! KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma/Gxma \
//!   cargo test --release --test device_sampler -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The tests are in task order: T1 pins which row a pick reads, T2 pins that
//! the device sampler reproduces the host sampler, T3 pins that the pipeline is
//! ungated and that temperature 0 did not move.

use kaplumba_engine::model::{HeadPick, TextModel};
use kaplumba_engine::sampler::{SamplerConfig, SamplerParams};
use kaplumba_kernels::encoder::MetalEncoder;

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL").expect("set KAPLUMBA_TEST_MODEL to a checkpoint directory")
}

fn load() -> TextModel {
    TextModel::load_from_dir(&std::path::PathBuf::from(model_path())).expect("load model")
}

/// Dispatch count of one pipelined decode step for `pick`.
///
/// `encode_softcap` is exactly one dispatch and nothing else in the head varies
/// with `pick` yet, so the *difference* in dispatch count between `Argmax` and
/// `Sample` **is** whether the softcap ran. That is the property T1 has to pin,
/// and it is observable without running a sampler.
///
/// One model for both measurements: two 12 GiB instances do not fit (AGENTS.md
/// §10), and the engine's own flock would refuse the second anyway.
fn head_dispatches(model: &mut TextModel) -> (usize, usize) {
    model.reset_gpu_caches();
    let queue = model.forward_queue();
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    model
        .forward_step_pipelined(&mut encoder, 0, Some(2), 0, 0, HeadPick::Argmax)
        .expect("argmax step");
    let argmax = encoder.dispatches_encoded();
    let pick = sample_pick(model);
    model
        .forward_step_pipelined(&mut encoder, 1, Some(2), 1, 1, pick)
        .expect("sample step");
    let sample = encoder.dispatches_encoded() - argmax;
    encoder.abandon();
    (argmax, sample)
}

// ---------------------------------------------------------------------------
// T1 — which row a pick reads
// ---------------------------------------------------------------------------

/// **T1, dispatch-level.** `HeadPick::Sample` must encode the softcap dispatch
/// that `HeadPick::Argmax` skips.
///
/// This is the sabotage-verified form of the claim. Revert
/// `HeadPick::reads_raw_row` to `staging` — i.e. derive the row from buffer
/// presence, as `skip_softcap` used to be derived — and this test fails with
/// `sample encoded N dispatches, argmax encoded N`, which is diagnosable.
///
/// The end-to-end consequence, that the *sampled text* changes when the row is
/// not the capped one, is
/// `ablation_changes_sampled_output_but_not_the_argmax`, which needs a running
/// sampler and therefore lands in T3.
#[test]
#[ignore = "loads the model (~12 GiB)"]
fn sampling_reads_the_capped_row_and_argmax_does_not() {
    let mut model = load();
    let (argmax, sample) = head_dispatches(&mut model);
    assert_eq!(
        sample,
        argmax + 1,
        "HeadPick::Sample must encode exactly one extra dispatch -- the softcap. \
         argmax={argmax} sample={sample}. If they are equal, skip_softcap is \
         being derived from buffer presence again and the sampler is reading \
         the RAW logits."
    );
}

fn sample_pick(model: &TextModel) -> HeadPick {
    let params = SamplerParams::from_config(SamplerConfig::default(), model.vocab_size())
        .expect("device sampler params for the default config");
    HeadPick::Sample(params)
}
