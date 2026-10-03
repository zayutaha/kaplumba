//! B3 probe: the counting top-k selector must return the DELETED selector's
//! `Vec<usize>`, byte for byte, and must be faster. **CPU-only. Opens no
//! checkpoint, dispatches no Metal.**
//!
//! # Why the equality is the whole claim
//!
//! `top_k_order` returns the candidate list `Sampler::sample` walks. A selector
//! that quietly picked a *different* list would produce plausible text, no error,
//! no failing assertion anywhere else, and would change every `--seed`
//! reproduction on the box. `sample` is a pure function of
//! `(order, logits, config, rng state)`, so an identical `order` forces an
//! identical token — which is why these gates compare the *vector*, and why the
//! last one compares the sampled token as well.
//!
//! # The oracle is the deleted code, verbatim
//!
//! `shipped_top_k_order` below is the `select_nth_unstable_by` this change
//! replaced, copied exactly (`sampler.rs:69-88` at 641e114). It is not a
//! reimplementation of something approximate: if this probe and the shipped
//! function disagree, one of them is wrong, and that is the point. It is also
//! what the timing compares against, from ONE binary, interleaved A B A B.
//!
//! # Every gate here has been observed failing (AGENTS.md 3.4)
//!
//! Each sabotage was applied to `crates/engine/src/sampler.rs`, the full gate set
//! was run, the failing assertions were recorded, and the sabotage was reverted.
//! The table is in `spec/decode-ceiling/findings/B3-topk.md`. Three real bugs
//! were found this way while writing the selection, plus one in the first draft of
//! this change; a claim of coverage that has never been seen to fail is not a
//! claim.

use std::time::Instant;

use kaplumba_engine::sampler::{top_k_order, Sampler, SamplerConfig};

const VOCAB: usize = 262_144;
const TOP_K: usize = 64;

// ---------------------------------------------------------------------------
// The oracle: `select_nth_unstable_by`, verbatim, as it was at 641e114.
// ---------------------------------------------------------------------------

/// The selection this change replaced, byte for byte.
///
/// `select_nth_unstable_by` partitions in O(n) and, because the comparator is a
/// TOTAL order (logit descending, index ascending), the first `keep` entries after
/// partitioning are exactly the set a full sort would have produced.
fn shipped_top_k_order(logits: &[f32], top_k: usize) -> Vec<usize> {
    let keep = top_k.min(logits.len());
    let mut order = (0..logits.len()).collect::<Vec<_>>();
    if keep < order.len() {
        order.select_nth_unstable_by(keep, |left, right| {
            logits[*right]
                .partial_cmp(&logits[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
    }
    order.truncate(keep);
    order.sort_by(|left, right| {
        logits[*right]
            .partial_cmp(&logits[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    order
}

/// `Sampler::sample` with the shipped selector in place of the current one, so
/// the two samplers are comparable end to end rather than only at the vector.
fn shipped_sample(logits: &[f32], config: SamplerConfig, state: &mut u64) -> u32 {
    assert!(!logits.is_empty() && logits.iter().all(|value| value.is_finite()));
    let order = shipped_top_k_order(logits, config.top_k);
    let maximum = logits[order[0]];
    let mut weights = order
        .iter()
        .map(|index| ((logits[*index] - maximum) / config.temperature).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    assert!(total.is_finite() && total > 0.0, "oracle hit the InvalidLogits path");
    let mut cumulative = 0.0;
    let mut selected = 0;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += *weight / total;
        selected = index;
        if cumulative >= config.top_p {
            break;
        }
    }
    weights.truncate(selected + 1);
    let selected_total = weights.iter().sum::<f32>();
    let target = next_f64(state) as f32 * selected_total;
    let mut running = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        running += *weight;
        if target <= running {
            return order[index] as u32;
        }
    }
    order[selected] as u32
}

fn next_f64(state: &mut u64) -> f64 {
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    let bits = value.wrapping_mul(0x2545_f491_4f6c_dd1d);
    ((bits >> 11) as f64) / ((1_u64 << 53) as f64)
}

/// The RNG state a given `--seed` starts from, spelled out here rather than called
/// off `Sampler`.
///
/// The oracle has to advance the sampler by exactly the number of steps the real one
/// does, so this must agree with `Sampler::new`. It is duplicated rather than shared so
/// this file compiles against a committed `sampler.rs` on its own: `seed_state` is
/// currently only reachable through another task's in-flight work, and a gate that
/// cannot be built from the commit it belongs to is not a gate.
fn seed_state(seed: u64) -> u64 {
    if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    }
}

// ---------------------------------------------------------------------------
// Rows. `fp16`-derived, because that is what the decode loop has:
// `generate.rs:326-330` widens the model's fp16 logit row to `f32` and hands it
// to `sample`.
// ---------------------------------------------------------------------------

fn widen(raw: &[u16]) -> Vec<f32> {
    raw.iter().map(|bits| half::f16::from_bits(*bits).to_f32()).collect()
}

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }
}

/// Random-but-finite, exponent 12..17 so every value is normal and finite.
fn row_uniform(len: usize) -> Vec<u16> {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            let exponent = 12u16 + ((rng.next_u64() >> 48) % 6) as u16;
            let mantissa = ((rng.next_u64() >> 32) & 0x03FF) as u16;
            (exponent << 10) | mantissa
        })
        .collect()
}

/// Closer to a real post-softcap row: one dominant peak, a long decaying tail,
/// and negative mass everywhere. This is the distribution the optimisation is
/// for, so it is the one the headline timing uses.
fn row_tailed(len: usize) -> Vec<u16> {
    let mut rng = Lcg(0xD1B5_4A32_D192_ED03);
    (0..len)
        .map(|index| {
            let unit = rng.unit();
            let logit = if index == 0 {
                28.0
            } else {
                24.0 - 3.2 * ((index as f64).ln() / 6.0) - 9.0 * unit + 0.4 * (rng.unit() - 0.5)
            };
            half::f16::from_f64(logit.clamp(-60.0, 60.0)).to_bits()
        })
        .collect()
}

/// A Gaussian head over a uniform floor — the shape of a confident model's row.
fn row_gaussian_head(len: usize) -> Vec<u16> {
    let mut rng = Lcg(0x2545_F491_4F6C_DD1D);
    let mut spare = None;
    let mut normal = move || {
        if let Some(value) = spare.take() {
            return value;
        }
        let (a, b) = (rng.unit(), rng.unit());
        let radius = (-2.0 * a.ln()).sqrt();
        let (sine, cosine) = (2.0 * std::f64::consts::PI * b).sin_cos();
        spare = Some(radius * sine);
        12.0 + radius * cosine
    };
    (0..len)
        .map(|_| half::f16::from_f64(normal().clamp(-60.0, 60.0)).to_bits())
        .collect()
}

fn row_ascending(len: usize) -> Vec<u16> {
    (0..len)
        .map(|index| {
            let value = -20.0 + 40.0 * (index as f64) / (len as f64);
            half::f16::from_f64(value).to_bits()
        })
        .collect()
}

fn row_descending(len: usize) -> Vec<u16> {
    let mut raw = row_ascending(len);
    raw.reverse();
    raw
}

/// Every element identical. Not a model output — the degenerate input a counting
/// selector could plausibly mishandle, and the reason the two-level refinement
/// exists instead of trusting the first bucket to be narrow.
fn row_all_equal(len: usize) -> Vec<u16> {
    vec![half::f16::from_f64(3.5).to_bits(); len]
}

/// `distinct` exact tie groups, ascending, so the tie group straddling the
/// `top_k` boundary is guaranteed rather than hoped for.
fn row_tie_dense(len: usize, distinct: usize) -> Vec<f32> {
    (0..len)
        .map(|index| ((index % distinct) as f32) * 0.25 - 8.0)
        .collect()
}

// ---------------------------------------------------------------------------
// The gate helper.
// ---------------------------------------------------------------------------

fn assert_same(logits: &[f32], k: usize, label: &str) {
    let shipped = shipped_top_k_order(logits, k);
    let current = top_k_order(logits, k);
    // Length first, then the first differing position. A bare `assert_eq!` on two
    // 4096-element vectors prints both in full, which is how a diagnosable failure
    // turns into a wall of text -- observed while sabotaging the descend direction.
    assert_eq!(
        current.len(),
        shipped.len(),
        "{label}: top_k_order returned {} candidates at k={k} (len={}) where the \
         shipped selector returned {}",
        current.len(),
        logits.len(),
        shipped.len()
    );
    if let Some((at, want, got)) = first_difference(&shipped, &current) {
        panic!(
            "{label}: top_k_order diverged from the shipped select_nth_unstable_by at \
             k={k} (len={}): position {at} is {want} in the shipped vector and {got} in \
             this one",
            logits.len()
        );
    }
}

fn first_difference(left: &[usize], right: &[usize]) -> Option<(usize, usize, usize)> {
    left.iter()
        .zip(right.iter())
        .position(|(a, b)| a != b)
        .map(|at| (at, left[at], right[at]))
}

// ---------------------------------------------------------------------------
// Gates.
// ---------------------------------------------------------------------------

/// The headline gate: byte-identical vectors on full-width rows, every `top_k`
/// that matters, across the three realistic distributions plus both adversarial
/// orderings and the degenerate all-equal row.
#[test]
fn counting_select_matches_the_shipped_select_on_full_width_rows() {
    let mut checked = 0_usize;
    for (name, logits) in [
        ("uniform", widen(&row_uniform(VOCAB))),
        ("tailed", widen(&row_tailed(VOCAB))),
        ("gaussian_head", widen(&row_gaussian_head(VOCAB))),
        ("ascending", widen(&row_ascending(VOCAB))),
        ("descending", widen(&row_descending(VOCAB))),
        ("all_equal", widen(&row_all_equal(VOCAB))),
    ] {
        for k in [1_usize, 2, 63, 64, 65, 1024] {
            assert_same(&logits, k, name);
            checked += 1;
        }
    }
    assert_eq!(checked, 36, "the row x k grid shrank");
}

/// Tie-dense rows at many widths and `top_k`s. This is the case a selection with
/// an inverted tie-break, or one that picks ties by arrival rather than by index,
/// gets wrong — and it gets it wrong while still returning a *sorted, plausible*
/// vector, so nothing downstream notices.
#[test]
fn tie_dense_rows_match_the_shipped_select() {
    let mut checked = 0_usize;
    for len in [2_usize, 3, 17, 64, 65, 513, 1024, 4096, 65_536] {
        for distinct in [1_usize, 2, 3, 7, 64, 65, len, len / 2 + 1] {
            if distinct == 0 || distinct > len {
                continue;
            }
            let logits = row_tie_dense(len, distinct);
            for k in [1_usize, 2, 3, 64, 65, len, len + 1] {
                if k == 0 {
                    continue;
                }
                assert_same(&logits, k, &format!("tie_dense(len={len},distinct={distinct})"));
                checked += 1;
            }
        }
    }
    assert!(checked > 300, "expected a large grid, got {checked}");
}

/// `k = 0`, `k = 1` and `k >= len`: the boundaries where the partition's own
/// guard (`if keep < order.len()`) does or does not fire.
#[test]
fn boundary_top_k_values_match() {
    for len in [1_usize, 2, 65, 1024] {
        let logits = row_tie_dense(len, 3);
        for k in [0_usize, 1, 2, len - 1, len, len + 1, len * 4] {
            assert_same(&logits, k, &format!("boundary(len={len})"));
        }
    }
    // `keep >= len` sorts the whole row, so it must be a plain full sort.
    let logits = row_tie_dense(64, 64);
    let k = 64;
    assert_eq!(top_k_order(&logits, k), shipped_top_k_order(&logits, k));
    assert_eq!(top_k_order(&logits, k).len(), k);
}

/// `-0.0` and `+0.0` compare EQUAL, so the shipped comparator resolves them by
/// INDEX. A key that ranked `-0.0` below `+0.0` would still be monotone on every
/// other row and would still return a sorted vector; it would just pick a
/// different SET whenever the `top_k` boundary fell inside the zeros.
///
/// The boundary straddling is built explicitly at `TOP_K - 1` / `TOP_K + 1`,
/// because a signed-zero row that does not straddle the boundary cannot detect the
/// bug at all — every candidate is zero, so every k gives the same answer.
#[test]
fn signed_zeros_straddling_the_boundary_match() {
    let base: Vec<f32> = widen(&row_tailed(4096));

    // All zeros, all four sign combinations.
    for pattern in [[-0.0_f32, 0.0], [0.0, -0.0]] {
        let mut logits = vec![0.0_f32; 4096];
        for (index, slot) in logits.iter_mut().enumerate() {
            *slot = pattern[index % 2];
        }
        for k in [1_usize, 2, 64, 65, 4096] {
            assert_same(&logits, k, "signed_zero_all");
        }
    }

    // One -0.0 and one +0.0 exactly at the boundary, with real mass above them.
    for (negative_slot, positive_slot) in [(TOP_K - 1, TOP_K + 1), (TOP_K, TOP_K - 2)] {
        let mut logits = base.clone();
        logits[negative_slot] = -0.0;
        logits[positive_slot] = 0.0;
        for k in [TOP_K - 2, TOP_K - 1, TOP_K, TOP_K + 1, TOP_K + 2] {
            assert_same(&logits, k, "signed_zero_at_boundary");
        }
    }

    // And a row that is entirely one of the two zeros, which must select the
    // LOWEST indices — the tie-break, not the sign.
    for value in [-0.0_f32, 0.0] {
        let logits = vec![value; 2048];
        assert_eq!(
            top_k_order(&logits, 8),
            (0..8).collect::<Vec<usize>>(),
            "a row of {value:e} must select the lowest indices"
        );
    }
}

/// `f32::MIN`, `f32::MAX`, the infinities and subnormals must keep their places
/// in the order. A sign-magnitude-to-two's-complement key implemented with the
/// wrong mask (or with the mask applied only to positives) inverts these against
/// ordinary values, and ordinary logits are all positive at the head, so a row of
/// `f32::MAX` and a row of `1.0` would look the same after such a break.
#[test]
fn extreme_f32_values_match() {
    let extremes = [
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        -1.0,
        1.0,
        -0.0,
        0.0,
    ];
    let mut logits: Vec<f32> = widen(&row_tailed(4096));
    // Extremes scattered through a realistic row.
    for (index, value) in extremes.iter().enumerate() {
        logits[7 + index * 331] = *value;
    }
    for k in [1_usize, 2, 12, 13, 64, 65, 1024] {
        assert_same(&logits, k, "extremes_scattered");
    }
    // A row made ONLY of extremes: the head is +inf, the tail is -inf, and
    // f32::MIN must sit above f32::MAX's negative counterpart.
    let only: Vec<f32> = extremes.iter().rev().copied().collect();
    for k in [1_usize, 2, 6, 12, 13] {
        assert_same(&only, k, "extremes_only");
    }
    // Full-width rows of one extreme each: exercises the second counting pass,
    // because one bucket holds the entire row.
    for value in [f32::MAX, f32::MIN, f32::NEG_INFINITY] {
        let logits = vec![value; VOCAB];
        for k in [1_usize, 64, 4096] {
            assert_same(&logits, k, &format!("full_width_{value:e}"));
        }
        // All equal, so the answer is the lowest indices.
        assert_eq!(
            top_k_order(&logits, 5),
            vec![0, 1, 2, 3, 4],
            "a row of {value:e} must select the lowest indices"
        );
    }
}

/// Randomised differential test over `f32` bit patterns, so the values are not
/// confined to what fp16 can express. Deterministic seed: a failing run has to be
/// reproducible.
#[test]
fn random_f32_rows_match_the_shipped_select() {
    let mut rng = Lcg(0x6A09_E667_F3BC_C909);
    for trial in 0..3000 {
        let len = 1 + (rng.next_u64() % 300) as usize;
        let logits: Vec<f32> = (0..len)
            .map(|_| {
                let mut value = f32::from_bits(rng.next_u64() as u32);
                // Finite only: `sample` rejects the rest, and the old comparator's
                // `unwrap_or(Equal)` made NaN unordered rather than ordered.
                while !value.is_finite() {
                    value = f32::from_bits(rng.next_u64() as u32);
                }
                value
            })
            .collect();
        let k = 1 + (rng.next_u64() % (len as u64 + 3)) as usize;
        assert_same(&logits, k, &format!("random_f32(trial={trial},len={len})"));
    }
}

/// The user-visible claim: for a fixed `--seed`, the SAME token sequence. Every
/// row in the sequence is different, because a decode never sees the same logits
/// twice, and the sampler state carries over between steps exactly as it does in
/// the decode loop.
#[test]
fn seeded_sampling_produces_the_identical_token_sequence() {
    let config = SamplerConfig { temperature: 1.0, top_k: TOP_K, top_p: 0.95 };
    for seed in [0_u64, 1, 12345, u64::MAX] {
        let mut sampler = Sampler::new(config, seed).expect("sampler");
        let mut oracle_state = seed_state(seed);
        let rows: Vec<Vec<f32>> = (0..24)
            .map(|step| {
                let mut row = widen(&row_tailed(4096));
                // Feed each step a different row so the walk cannot pass by
                // accident.
                row[step * 7] = 30.0 - step as f32;
                row
            })
            .collect();
        for (step, logits) in rows.iter().enumerate() {
            let mine = sampler.sample(logits).expect("sample");
            let theirs = shipped_sample(logits, config, &mut oracle_state);
            assert_eq!(
                mine, theirs,
                "seed={seed} step={step}: sampled a different token than the shipped \
                 selector would have, with no error raised"
            );
        }
    }
}

/// And at three other temperatures / top_p values, because the top-k boundary is
/// not the only place a different order shows up: the top-p walk and the
/// inverse-CDF both consume the same order.
#[test]
fn seeded_sampling_matches_across_the_config_grid() {
    for temperature in [0.05_f32, 0.7, 1.0, 2.0] {
        for top_p in [0.5_f32, 0.95, 1.0] {
            for top_k in [1_usize, 8, 64, 512] {
                let config = SamplerConfig { temperature, top_k, top_p };
                let mut sampler = Sampler::new(config, 99).expect("sampler");
                let mut oracle_state = seed_state(99);
                for step in 0..6 {
                    let mut logits = widen(&row_gaussian_head(2048));
                    logits[step * 31 % 2048] = 25.0 - step as f32;
                    let mine = sampler.sample(&logits).expect("sample");
                    let theirs = shipped_sample(&logits, config, &mut oracle_state);
                    assert_eq!(
                        mine, theirs,
                        "T={temperature} top_p={top_p} top_k={top_k} step={step}"
                    );
                }
            }
        }
    }
}

/// The greedy path must be untouched. It never calls `sample`'s selection, so this
/// is a regression guard on the refactor rather than a property of the new code:
/// `--temperature 0` must stay an argmax at any `top_k`.
#[test]
fn greedy_is_still_argmax_and_never_touches_the_selection() {
    let logits = row_tie_dense(512, 8);
    let mut unique = logits.clone();
    unique[300] = 100.0;
    // The tie rule the greedy path uses: strict `>`, so the FIRST maximum wins.
    // `row_tie_dense(512, 8)` peaks at index 7, and a naive "ties resolve to 0"
    // expectation would pass on a row whose first element is the peak and fail
    // here, so it is computed rather than hardcoded.
    let expected = |row: &[f32]| {
        let mut best = 0;
        for index in 1..row.len() {
            if row[index] > row[best] {
                best = index;
            }
        }
        best as u32
    };
    assert_eq!(expected(&logits), 7, "the tie-dense row's first argmax moved");
    for top_k in [1_usize, 64, 4096] {
        let mut sampler = Sampler::new(
            SamplerConfig { temperature: 0.0, top_k, top_p: 0.95 },
            7,
        )
        .expect("sampler");
        assert_eq!(
            sampler.sample(&unique).expect("sample"),
            expected(&unique),
            "top_k={top_k}"
        );
        // On a tie-dense row the greedy answer is the lowest argmax index, which is
        // the same tie rule the selection has to honour.
        assert_eq!(
            sampler.sample(&logits).expect("sample"),
            expected(&logits),
            "top_k={top_k}"
        );
    }
}

// ---------------------------------------------------------------------------
// Timing: shipped vs current, from ONE binary, interleaved.
// ---------------------------------------------------------------------------

/// Median, min and max of `iters` samples, per AGENTS.md 10.2.5: decode on this
/// box has been measured anywhere between 3.7 and 6.4 tok/s from the same binary
/// minutes apart, so a single reading is not a result and a difference inside the
/// spread is noise.
struct Timing {
    median: f64,
    min: f64,
    max: f64,
}

fn summarize(mut samples: Vec<f64>) -> Timing {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    Timing {
        median: samples[samples.len() / 2],
        min: samples[0],
        max: samples[samples.len() - 1],
    }
}

/// A B A B ... with the order flipped every iteration, so a drift in machine
/// speed cannot systematically favour whichever side ran first.
fn interleaved(iters: usize, mut old: impl FnMut(), mut new: impl FnMut()) -> (Timing, Timing) {
    let mut old_samples = Vec::with_capacity(iters);
    let mut new_samples = Vec::with_capacity(iters);
    for _ in 0..3 {
        old();
        new();
    }
    for iteration in 0..iters {
        let record = |body: &mut dyn FnMut(), into: &mut Vec<f64>| {
            let started = Instant::now();
            body();
            into.push(started.elapsed().as_secs_f64() * 1e3);
        };
        if iteration % 2 == 0 {
            record(&mut old, &mut old_samples);
            record(&mut new, &mut new_samples);
        } else {
            record(&mut new, &mut new_samples);
            record(&mut old, &mut old_samples);
        }
    }
    (summarize(old_samples), summarize(new_samples))
}

/// The measurement. Timings print; they are not asserted — the assertions in this
/// file are the correctness gates above, and a hard-coded millisecond threshold in
/// a test suite is a flake generator. What is asserted is the ORDERING under the
/// adversarial rows, because that is the property the choice of algorithm rests
/// on: the counting select must not be the slower one on ANY row.
#[test]
fn probe_top_k_selection_timing() {
    let config = SamplerConfig { temperature: 1.0, top_k: TOP_K, top_p: 0.95 };
    let iters = 41;

    println!("\n=== B3: top-k selection, shipped select_nth vs counting select ===");
    println!("{VOCAB}-wide fp16-derived row widened to f32, top_k={TOP_K};");
    println!("median ms/call (min..max over {iters} interleaved samples)\n");
    println!(
        "{:<15} {:>9} {:>9} {:>7}   {:>9} {:>9} {:>7}   {:>7}",
        "row", "select_ms", "lo", "hi", "count_ms", "lo", "hi", "speedup"
    );

    let mut selection: Vec<(&str, Timing, Timing)> = Vec::new();
    let mut rows: Vec<(&str, Vec<u16>)> = vec![
        ("uniform", row_uniform(VOCAB)),
        ("tailed", row_tailed(VOCAB)),
        ("gaussian_head", row_gaussian_head(VOCAB)),
        ("ascending", row_ascending(VOCAB)),
        ("descending", row_descending(VOCAB)),
        ("all_equal", row_all_equal(VOCAB)),
    ];
    // The bounded heap's worst case is an ascending row, and the shipped
    // partition's worst case is a random one, so both are measured.
    rows.push(("two_values", {
        let mut raw = row_tailed(VOCAB);
        let filler = raw[0];
        for slot in raw.iter_mut().skip(1) {
            *slot = filler;
        }
        raw[0] = half::f16::from_f64(31.0).to_bits();
        raw
    }));

    for (name, raw) in &rows {
        let logits = widen(raw);
        let (old, new) = interleaved(iters, || {
            std::hint::black_box(shipped_top_k_order(std::hint::black_box(&logits), TOP_K));
        }, || {
            std::hint::black_box(top_k_order(std::hint::black_box(&logits), TOP_K));
        });
        println!(
            "{:<15} {:>9.3} {:>9.3} {:>9.3}   {:>9.3} {:>9.3} {:>9.3}   {:>6.1}x",
            name,
            old.median,
            old.min,
            old.max,
            new.median,
            new.min,
            new.max,
            old.median / new.median
        );
        selection.push((name, old, new));
    }

    // The claim that decided the algorithm is about the WORST case, not the
    // per-row median: a selection whose worst input is better than the
    // incumbent's worst input cannot produce the kind of production regression the
    // bounded heap would have (4.95 ms on the ascending row, 2.6x the partition's
    // own worst). A per-row "must be faster" assertion would be false and would
    // have to be deleted to go green, which is how a real limitation gets hidden.
    let worst_old = selection.iter().map(|(_, old, _)| old.max).fold(0.0_f64, f64::max);
    let worst_new = selection.iter().map(|(_, _, new)| new.max).fold(0.0_f64, f64::max);
    println!(
        "\nworst single sample over every row: shipped {:.3} ms, counting {:.3} ms ({:.1}x)",
        worst_old,
        worst_new,
        worst_old / worst_new
    );
    for (name, old, new) in &selection {
        if new.median >= old.median {
            println!(
                "  note: the partition is faster on `{name}` ({:.3} vs {:.3} ms) -- \
                 Introselect is a single pass on presorted input",
                old.median,
                new.median
            );
        }
    }
    assert!(
        worst_new < worst_old,
        "the counting select's worst sample ({worst_new:.3} ms) must be below the \
         partition's worst ({worst_old:.3} ms), or it has no tail advantage at all"
    );

    // End to end through `sample`, which is what the decode step actually pays.
    println!("\nwhole `sample()` (T=1.0, top_k={TOP_K}, top_p=0.95), ms/token");
    println!("{:<15} {:>9} {:>9} {:>7}   {:>9} {:>9} {:>7}   {:>7}", "row", "shipped_ms", "lo", "hi", "now_ms", "lo", "hi", "speedup");
    for (name, raw) in &rows {
        let logits = widen(raw);
        let mut sampler = Sampler::new(config, 1).expect("sampler");
        let mut state = seed_state(1);
        let (old, new) = interleaved(iters, || {
            std::hint::black_box(shipped_sample(std::hint::black_box(&logits), config, &mut state));
        }, || {
            std::hint::black_box(sampler.sample(std::hint::black_box(&logits)).expect("sample"));
        });
        println!(
            "{:<15} {:>9.3} {:>9.3} {:>9.3}   {:>9.3} {:>9.3} {:>9.3}   {:>6.1}x",
            name,
            old.median,
            old.min,
            old.max,
            new.median,
            new.min,
            new.max,
            old.median / new.median
        );
    }

    // `top_k` sweep on the realistic row: the count is chosen from the row, so a
    // large `top_k` forces a much wider bucket and a second counting pass.
    println!("\ntop_k sweep on the tailed row (counting select), ms/call");
    let logits = widen(&row_tailed(VOCAB));
    for k in [1_usize, 8, 64, 256, 1024, 4096, 16_384] {
        let (_, new) = interleaved(11, || {}, || {
            std::hint::black_box(top_k_order(std::hint::black_box(&logits), k));
        });
        println!("  k={k:<6} {:>8.3} ms", new.median);
    }
}