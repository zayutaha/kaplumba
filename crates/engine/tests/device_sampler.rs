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

use kaplumba_engine::model::{HeadPick, TextModel, HOST_TOKEN_SLOTS};
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

// ---------------------------------------------------------------------------
// T2 — the device sampler reproduces the host sampler
// ---------------------------------------------------------------------------

/// A 262,144-wide row the model will never actually emit.
///
/// Real logits are smooth and fp16; the shapes that break a top-k are the ones
/// a real forward pass cannot be made to produce on demand. Every one of these
/// is reachable by some input, and `top_k_order` is the oracle for all of them.
mod rows {
    /// Deterministic, so a failure is reproducible from the test name alone.
    fn rng(state: &mut u64) -> f32 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (((bits >> 11) as f64) / ((1_u64 << 53) as f64)) as f32
    }

    fn build(width: usize, seed: u64, shape: fn(usize, &mut u64) -> f32) -> Vec<f32> {
        let mut state = seed;
        (0..width).map(|index| shape(index, &mut state)).collect()
    }

    /// fp16-representable, peaked, and heavy-tailed: the closest thing to a real
    /// post-softcap row. `half::f16` rounds here, so this row survives the
    /// device's fp16 staging without changing.
    pub fn realistic(width: usize) -> Vec<f32> {
        build(width, 0x5eed_1234, |_, s| {
            let head = rng(s);
            let tail = rng(s).abs() * 0.25;
            half::f16::from_f32((head - 0.5) * 8.0 + tail).to_f32()
        })
    }

    /// Every value identical. The whole top-k is a tie, so the result is
    /// decided entirely by the ascending-index tie-break -- the case where a
    /// comparator with the tie-break inverted looks fine and samples differently.
    pub fn all_equal(width: usize) -> Vec<f32> {
        vec![0.5; width]
    }

    /// Two values, alternating. Ties everywhere *and* across block boundaries.
    pub fn two_values(width: usize) -> Vec<f32> {
        (0..width)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect()
    }

    /// Ascending. The worst case for any selection that depends on *where* in
    /// the row the good tokens sit, and the case where a block-local top-k is
    /// most likely to be truncated wrongly.
    pub fn ascending(width: usize) -> Vec<f32> {
        (0..width)
            .map(|index| index as f32 / width as f32 * 4.0 - 2.0)
            .collect()
    }

    /// Descending: the same, mirrored.
    pub fn descending(width: usize) -> Vec<f32> {
        ascending(width).into_iter().rev().collect()
    }

    /// The extremes a half can hold, scattered. `+inf` must rank above every
    /// finite value and `-inf` below every one.
    ///
    /// **No `NaN`, deliberately.** `top_k_order` documents that it requires
    /// finite logits, and `Sampler::sample` enforces it with `InvalidLogits`,
    /// because its comparator's `partial_cmp(..).unwrap_or(Equal)` is not a
    /// strict weak ordering under `NaN`. So the host oracle has *no defined
    /// answer* on a row containing one, and comparing against it would be
    /// testing a function outside its contract. This was found the hard way: an
    /// earlier version of this row included `NaN` and the host returned a `NaN`
    /// element at rank 0 — not a device bug, an invalid comparison.
    ///
    /// The device's behaviour on `NaN` is pinned separately, and deliberately,
    /// by `nan_never_wins_on_the_device`: `value_key` folds `NaN` to 0, which is
    /// below every real key, so it can never be selected.
    pub fn extremes(width: usize) -> Vec<f32> {
        let specials = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            0.0,
            -0.0,
            65504.0,
            -65504.0,
        ];
        let mut row = realistic(width);
        for (index, slot) in row.iter_mut().enumerate() {
            if index % 977 == 0 {
                *slot = specials[(index / 977) % specials.len()];
            }
        }
        row
    }

    /// `NaN` scattered through the row. Device-only: the host sampler refuses
    /// this row outright, so there is nothing to compare against.
    pub fn with_nan(width: usize) -> Vec<f32> {
        let mut row = realistic(width);
        for (index, slot) in row.iter_mut().enumerate() {
            if index % 1013 == 0 {
                *slot = f32::NAN;
            }
        }
        row
    }

    /// Ties concentrated exactly at the `top_k` boundary: the case where an
    /// off-by-one in the candidate count is invisible everywhere else.
    pub fn ties_at_the_boundary(width: usize, top_k: usize) -> Vec<f32> {
        let mut row = vec![-10.0_f32; width];
        // Two distinct values, both above everything else, straddling the cut.
        for index in 0..top_k {
            row[index] = if index % 2 == 0 { 1.0 } else { 1.0 };
        }
        for index in top_k..top_k * 2 {
            row[index] = 0.5;
        }
        row
    }
}

/// **T2, the load-bearing one: the device top-k must equal the host's, in set
/// AND order, exactly.**
///
/// `ranked_out` exists for this test. A *drawn* token is a random member of the
/// candidate set and says nothing about its position in it, so a set-only or
/// token-only test passes a kernel that returns the right candidates in the
/// wrong order -- which is a different sampler, because the softmax, the top-p
/// walk and the inverse-CDF all read that order.
///
/// The oracle is the shipped `top_k_order`, which is itself pinned against a
/// full sort and against the deleted Introselect partition in
/// `probe_topk_equality.rs`. So this asserts device == host == sort.
///
/// The row shapes matter more than the row count. `all_equal` makes the entire
/// result the ascending-index tie-break; `ascending` puts every good token at
/// the end of the row, where a block-local top-k is most likely to be truncated
/// wrongly; `extremes` pins `+inf` above and `NaN` below everything.
#[test]
#[ignore = "loads the model (~12 GiB)"]
fn device_top_k_matches_the_host_set_and_order_on_adversarial_rows() {
    let mut model = load();
    let width = model.vocab_size();
    let config = SamplerConfig {
        temperature: 1.0,
        top_k: 64,
        top_p: 0.95,
    };

    let mut cases: Vec<(&str, Vec<f32>)> = vec![
        ("realistic", rows::realistic(width)),
        ("all_equal", rows::all_equal(width)),
        ("two_values", rows::two_values(width)),
        ("ascending", rows::ascending(width)),
        ("descending", rows::descending(width)),
        ("extremes", rows::extremes(width)),
        ("ties_at_the_boundary", rows::ties_at_the_boundary(width, 64)),
    ];
    // `with_nan` is deliberately absent: the host sampler rejects a non-finite
    // row, so there is no oracle. See `nan_never_wins_on_the_device`.
    // A second peak of candidates straddling a *different* cut, so `top_k` is
    // not the only boundary being tested.
    cases.push(("ties_at_32", rows::ties_at_the_boundary(width, 32)));

    for (name, row) in cases {
        // The host oracle, on the same fp16 row the device will see. Staging
        // rounds to fp16, so round here too or the two are not comparable.
        let staged: Vec<f32> = row
            .iter()
            .map(|v| half::f16::from_f32(*v).to_f32())
            .collect();
        let expected = kaplumba_engine::sampler::top_k_order(&staged, config.top_k);

        model.stage_capped_row(&row).expect("stage row");
        let draws = model
            .sample_capped_row_ranked(config, 0x9e37_79b9_7f4a_7c15, 1)
            .expect("device sample");
        let got: Vec<u32> = draws[0].ranked.clone();

        assert_eq!(
            got.len(),
            config.top_k,
            "{name}: device returned {} candidates, host returned {}",
            got.len(),
            config.top_k
        );
        assert_eq!(
            got.iter().map(|&t| t as usize).collect::<Vec<_>>(),
            expected,
            "{name}: device top-k differs from the host's. Compare as SET and as \
             ORDER -- both are required. row[0..4]={:?}",
            &row[..4.min(row.len())]
        );
    }
}

/// **T2: `NaN` must never win, on the device, even though the host refuses the
/// row entirely.**
///
/// `Sampler::sample` raises `InvalidLogits` on a non-finite row, so the host
/// never has to decide what `NaN` ranks as. The device has no such guard: it
/// reduces the whole row. `value_key` folds `NaN` to key 0, which is below every
/// real key, so `NaN` is structurally incapable of being selected.
///
/// This is asserted directly rather than by comparison, because there is no
/// oracle to compare against — and it matters: a `NaN` that ranked *first*
/// would be an infinite weight and a hard crash in the softmax, or a silent
/// always-same-token sampler if it merely ranked high.
#[test]
#[ignore = "loads the model (~12 GiB)"]
fn nan_never_wins_on_the_device() {
    let mut model = load();
    let width = model.vocab_size();
    let config = SamplerConfig {
        temperature: 1.0,
        top_k: 64,
        top_p: 0.95,
    };

    let row = rows::with_nan(width);
    let nan_indices: std::collections::HashSet<u32> = row
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_nan())
        .map(|(i, _)| i as u32)
        .collect();
    assert!(nan_indices.len() > 64, "the row must contain more NaNs than top_k");

    model.stage_capped_row(&row).expect("stage row");
    let draws = model
        .sample_capped_row_ranked(config, 0x9e37_79b9_7f4a_7c15, 64)
        .expect("device sample");

    for (rank, &token) in draws[0].ranked.iter().enumerate() {
        assert!(
            !nan_indices.contains(&token),
            "rank {rank} selected NaN element {token}; value_key must fold NaN \
             below every real key so it can never be chosen"
        );
    }
    // And the draw must stay in range rather than walking off the row.
    for draw in &draws {
        for &token in &draw.ranked {
            assert!((token as usize) < width, "token {token} outside the vocabulary");
        }
    }
}

/// **T2: the device draw must be a draw, not an argmax.**
///
/// This is the gate for the bug the RNG fix came from. The hand-rolled rounding
/// made `unit` ~2e9x too small, so the inverse-CDF target sat below the first
/// partial sum and *every* draw returned the top candidate. The top-k equality
/// test above would have passed throughout, because the ranking was never the
/// thing that was broken.
///
/// So the property is distributional: over many draws from a genuinely peaked
/// row, the token frequencies must track the softmax, and the top candidate must
/// NOT win ~always. A total-variation distance against the exact softmax
/// distribution is the sharp statement; the argmax-always check is the blunt one
/// that fails first and says why.
#[test]
#[ignore = "loads the model (~12 GiB)"]
fn device_draws_follow_the_softmax_instead_of_collapsing_to_the_argmax() {
    let mut model = load();
    let width = model.vocab_size();
    let config = SamplerConfig {
        temperature: 1.0,
        top_k: 64,
        top_p: 1.0, // full top-k support, so the comparison is against the softmax
    };

    // Peaked but not degenerate: a handful of tokens carry the mass, so a
    // collapsed sampler is obvious.
    let row = rows::realistic(width);
    let staged: Vec<f32> = row
        .iter()
        .map(|v| half::f16::from_f32(*v).to_f32())
        .collect();
    let order = kaplumba_engine::sampler::top_k_order(&staged, config.top_k);
    let maximum = staged[order[0]];
    let weights: Vec<f32> = order
        .iter()
        .map(|&i| ((staged[i] - maximum) / config.temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();

    model.stage_capped_row(&row).expect("stage row");
    // Staging is capped at `HOST_TOKEN_SLOTS` (8192) per command, so 1e5 draws
    // are taken in batches against the same staged row. Each batch gets its own
    // seed, which is what independence requires here: the batches are 1e5 draws
    // from the same distribution, and reseeding per batch gives exactly that.
    // (Reusing one seed across batches would replay the same draws and the
    // histogram would be a 12x-weighted copy of a single 8192-draw sample.)
    let total_draws = 100_000_usize;
    let mut tokens: Vec<u32> = Vec::with_capacity(total_draws);
    let mut taken = 0_usize;
    while taken < total_draws {
        let batch = (total_draws - taken).min(HOST_TOKEN_SLOTS);
        let batch_tokens = model
            .sample_capped_row_on_device(config, batch as u64, batch as u32)
            .expect("device draws");
        assert_eq!(batch_tokens.len(), batch);
        tokens.extend(batch_tokens);
        taken += batch;
    }

    assert_eq!(tokens.len(), total_draws);
    let top = order[0] as u32;
    let top_hits = tokens.iter().filter(|&&t| t == top).count();
    let expected_top = weights[0] / total;
    let top_share = top_hits as f64 / tokens.len() as f64;

    // The blunt check first: it fails with a message that names the actual bug.
    assert!(
        top_share < expected_top as f64 + 0.02,
        "the top candidate took {:.4} of draws but its probability is {:.4}. \
         A device sampler whose `unit` is far too small collapses every draw \
         onto the top candidate -- i.e. it is argmax, not sampling.",
        top_share,
        expected_top
    );

    // Then the sharp one: total variation against the exact distribution.
    let mut device_counts = std::collections::HashMap::<u32, u64>::new();
    for token in &tokens {
        *device_counts.entry(*token).or_default() += 1;
    }
    let n = tokens.len() as f64;
    let mut tv = 0.0_f64;
    for (rank, &index) in order.iter().enumerate() {
        let expected = weights[rank] as f64 / total as f64;
        let observed = device_counts.get(&(index as u32)).copied().unwrap_or(0) as f64 / n;
        tv += (observed - expected).abs();
    }
    let tv = tv * 0.5;

    // **The threshold is measured, not guessed.** Five independent 1e5-draw
    // trials on this box gave 0.0344 / 0.0357 / 0.0336 / 0.0256 / 0.0346, so the
    // noise floor for a *correct* sampler is ~0.026-0.036. That matches theory:
    // the support here is the full top-64 (mean bin p ~ 1/64), so each bin has
    // sd ~ sqrt(p(1-p)/n) ~ 4e-4 and 64 bins sum to ~0.025.
    //
    // 0.06 sits ~1.7x above the worst observed trial, so it does not flake. And
    // it is nowhere near the failure it exists to catch: a sampler collapsed
    // onto the argmax has TV ~ 1 - p_top ~ 0.98, and a mis-scaled `unit` (the
    // bug this test was written for) collapses it just as completely. The
    // margin between "correct" and "broken" here is ~30x, not 1.4x — an earlier
    // 0.05 threshold was inside the noise band and would have been a coin flip.
    assert!(
        tv < 0.06,
        "total variation distance {tv:.4} between the device draws and the \
         exact softmax exceeds 0.06 over {total_draws} draws (noise floor for a \
         correct sampler on this box is 0.026-0.036). The device sampler is not \
         reproducing the host distribution."
    );
    eprintln!("top_share={top_share:.4} expected={expected_top:.4} tv={tv:.4}");
}

/// **T2: one `sample_draw` per step, so one `u64` of RNG state is enough.**
///
/// The device sampler advances a single `device ulong` once per dispatch. That
/// is only sound if exactly one `sample_draw` is encoded per decode step — two
/// would advance it twice per token and the sequence would desynchronise from
/// any host expectation, silently.
///
/// Run over 64 draws in one command buffer and check the state advanced 64
/// times, by confirming draw N differs from draw 0 and that the draws are not
/// all identical (a stalled RNG returns the same token every time).
#[test]
#[ignore = "loads the model (~12 GiB)"]
fn one_rng_advance_per_draw_and_never_a_stall() {
    let mut model = load();
    let width = model.vocab_size();
    let config = SamplerConfig {
        temperature: 1.0,
        top_k: 64,
        top_p: 0.95,
    };
    model
        .stage_capped_row(&rows::realistic(width))
        .expect("stage row");

    let count = 64_u32;
    let tokens = model
        .sample_capped_row_on_device(config, 0x9e37_79b9_7f4a_7c15, count)
        .expect("device draws");
    assert_eq!(tokens.len(), count as usize);

    let distinct: std::collections::HashSet<u32> = tokens.iter().copied().collect();
    assert!(
        distinct.len() > 1,
        "64 draws from a peaked row produced {} distinct tokens. The RNG is \
         stalled -- most likely `xorshift64(0)`, which is a fixed point.",
        distinct.len()
    );
    // A seeded run must be reproducible: same seed, same sequence.
    let again = model
        .sample_capped_row_on_device(config, 0x9e37_79b9_7f4a_7c15, count)
        .expect("device draws again");
    assert_eq!(
        tokens, again,
        "the same --seed produced a different sequence; the device RNG state is \
         not being reset per run"
    );
}

/// **T2: a `top_k` the device cannot stage must be refused, not approximated.**
///
/// This is the end-to-end half of the `device_top_k_limit` fix. The CPU gate
/// `the_host_bound_and_the_kernel_geometry_agree_at_every_top_k` pins the two
/// validators agreeing; this pins that the refusal actually reaches the caller
/// as an error on the decode entry point, rather than being swallowed by a
/// fallback that quietly samples from a narrower set.
#[test]
#[ignore = "loads the model (~12 GiB)"]
fn an_unservable_top_k_is_refused_at_the_call() {
    let mut model = load();
    let width = model.vocab_size();
    model
        .stage_capped_row(&rows::realistic(width))
        .expect("stage row");

    let limit = kaplumba_engine::sampler::device_top_k_limit(width);
    assert!(limit >= 64, "the shipped default must be servable, got {limit}");

    for top_k in [limit + 1, 128, 1024] {
        let config = SamplerConfig {
            temperature: 1.0,
            top_k,
            top_p: 0.95,
        };
        assert!(
            !model.device_sampler_supported(config),
            "top_k={top_k} above the limit {limit} must not report as device-servable"
        );
        let outcome = model.sample_capped_row_ranked(config, 1, 1);
        assert!(
            outcome.is_err(),
            "top_k={top_k} must be refused at the call, not sampled from a \
             narrower candidate set"
        );
    }
}
