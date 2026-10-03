//! B1 probe: phase decomposition of the host sampling cost on a 262,144-wide
//! fp16 logit row. **CPU-only. Opens no checkpoint, dispatches no Metal.**
//!
//! Why this exists. The `temperature 1.0` vs `temperature 0` decode gap is
//! attributed in part to "+2.87 ms host `Sampler::sample`" (`sampler.rs`) plus
//! "+0.45 ms fp16->f32 widen" (`generate.rs:326-330`). Neither number has been
//! split by phase. This probe measures each phase against the REAL
//! `sampler::top_k_order` and the REAL `Sampler::sample`, and times three
//! candidate replacements alongside, each **proved to return the shipped
//! vector** before its timing is believed
//! (`candidate_selection_is_bit_identical_to_the_shipped_one`).
//!
//! Three results are worth stating up front, because two of them contradict
//! the framing this probe was commissioned under:
//!
//! 1. The selection is a **full sort** already fixed. `top_k_order`
//!    (`sampler.rs:69`) is `select_nth_unstable_by` (Introselect, O(n)) and its
//!    own doc comment records full sort 14.71 ms -> partition 1.89 ms. So "is
//!    the top-k a full sort?" is settled; the live question is why the O(n)
//!    partition still costs 1.7 ms. Answer: it allocates and partitions a 2 MiB
//!    `Vec<usize>` (8 B x 262144) and its comparator does two INDIRECT gathers
//!    into a 1 MiB `f32` array per comparison. ~95% of the host sampler is this
//!    one call.
//! 2. `value_key` (`elements.metal:531`) is **only safe through `argmax_key`**
//!    (`:545`). See `value_key_upper_bits_are_garbage_for_negatives`.
//! 3. A u32 total order over vocab 262144 **does not exist** (16 value bits + 18
//!    index bits = 34). See `a_u32_key_cannot_total_order_the_full_row`.
//!
//! This is a timing probe, so the timings print rather than assert; the
//! assertions are the correctness gates, which are the part that must not be
//! allowed to drift.

use std::time::Instant;

use kaplumba_engine::sampler::{top_k_order, Sampler, SamplerConfig};

const VOCAB: usize = 262_144;
const TOP_K: usize = 64;

// ---------------------------------------------------------------------------
// Logit rows.
//
// `uniform` is the row `examples/bench_sampler.rs` uses, so the numbers here are
// comparable with the ones already recorded against it. `tailed` is closer to a
// real post-softcap Gemma row. `ascending` and `descending` are the adversarial
// orderings, included because a bounded heap has a data-dependent worst case
// and a claim that ignores it is not a claim.
// ---------------------------------------------------------------------------

fn row_uniform() -> Vec<u16> {
    let mut state = 0x9E37_79B9u64;
    (0..VOCAB)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let exponent = 12u16 + ((state >> 48) % 6) as u16;
            let mantissa = ((state >> 32) & 0x03FF) as u16;
            (exponent << 10) | mantissa
        })
        .collect()
}

fn row_tailed() -> Vec<u16> {
    let mut state = 0xD1B5_4A32_D192_ED03u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..VOCAB)
        .map(|index| {
            let u = next();
            let logit = if index == 0 {
                28.0
            } else {
                24.0 - 3.2 * ((index as f64).ln() / 6.0) - 9.0 * u + 0.4 * (next() - 0.5)
            };
            half::f16::from_f64(logit.clamp(-60.0, 60.0)).to_bits()
        })
        .collect()
}

fn row_ascending() -> Vec<u16> {
    (0..VOCAB)
        .map(|i| {
            let value = -20.0 + 40.0 * (i as f64) / (VOCAB as f64);
            half::f16::from_f64(value).to_bits()
        })
        .collect()
}

fn row_descending() -> Vec<u16> {
    let mut raw = row_ascending();
    raw.reverse();
    raw
}

fn widen(raw: &[u16]) -> Vec<f32> {
    raw.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect()
}

// ---------------------------------------------------------------------------
// Candidate A: bounded max-heap top-k over the widened f32 row.
//
// The smallest possible diff: same signature, same input, same output vector.
// Safe because the shipped comparator is already a TOTAL order (value
// descending, then index ascending), so the top-k set and its ordering are
// uniquely determined and any correct selection must produce that same vector.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct HeapEntry {
    logit: f32,
    index: u32,
}

/// `true` when `a` comes strictly earlier under (value desc, index asc).
#[inline]
fn heap_better(a: HeapEntry, b: HeapEntry) -> bool {
    if a.logit != b.logit {
        a.logit > b.logit
    } else {
        a.index < b.index
    }
}

/// Weak heap on `heap_better`: the ROOT is the WORST kept element, i.e. the one
/// an incoming candidate must beat. The comparisons are phrased "is this child
/// worse than the current worst", because the opposite sense builds a max-heap,
/// keeps the best element and silently fails to evict it. That bug is invisible
/// on a tie-free row, which is why the gate below uses tie-dense rows.
fn sift_down_f32(heap: &mut [HeapEntry], mut parent: usize) {
    loop {
        let left = 2 * parent + 1;
        let right = left + 1;
        let mut worst = parent;
        if left < heap.len() && heap_better(heap[worst], heap[left]) {
            worst = left;
        }
        if right < heap.len() && heap_better(heap[worst], heap[right]) {
            worst = right;
        }
        if worst == parent {
            return;
        }
        heap.swap(parent, worst);
        parent = worst;
    }
}

fn top_k_heap_f32(logits: &[f32], top_k: usize) -> Vec<usize> {
    let keep = top_k.min(logits.len());
    let mut heap: Vec<HeapEntry> = Vec::with_capacity(keep);
    for index in 0..logits.len() {
        let candidate = HeapEntry { logit: logits[index], index: index as u32 };
        if heap.len() < keep {
            heap.push(candidate);
            if heap.len() == keep {
                // Bottom-up heapify, O(k).
                for parent in (0..keep / 2).rev() {
                    sift_down_f32(&mut heap, parent);
                }
            }
        } else if heap_better(candidate, heap[0]) {
            heap[0] = candidate;
            sift_down_f32(&mut heap, 0);
        }
    }
    heap.sort_by(|a, b| {
        if a.logit != b.logit {
            b.logit.partial_cmp(&a.logit).unwrap()
        } else {
            a.index.cmp(&b.index)
        }
    });
    heap.into_iter().map(|entry| entry.index as usize).collect()
}

// ---------------------------------------------------------------------------
// The fp16 key map, transcribed from `elements.metal`.
// ---------------------------------------------------------------------------

/// `sortable16` (`elements.metal:527`). NOTE `~bits` is a 32-bit NOT, so for a
/// negative input the high 16 bits of the result are set; see
/// `value_key_upper_bits_are_garbage_for_negatives`.
#[inline]
fn sortable16(bits: u16) -> u32 {
    if (bits & 0x8000) != 0 { !(bits as u32) } else { (bits ^ 0x8000) as u32 }
}

/// `value_key` (`elements.metal:531`), verbatim.
#[inline]
fn value_key(bits: u16) -> u32 {
    if (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0 {
        return 0u32; // NaN: below -inf, can never win
    }
    if (bits & 0x7FFF) == 0 {
        return 0x8000u32; // +0.0, with -0.0 folded onto it
    }
    sortable16(bits)
}

/// `value_key` truncated to the 16 bits that actually carry the order. This is
/// what a standalone consumer needs; `argmax_key` gets it for free by doing
/// `value_key(bits) << 16`.
#[inline]
fn sortable(bits: u16) -> u16 {
    (value_key(bits) & 0xFFFF) as u16
}

/// vocab 262144 needs 18 index bits, so a total order over the whole row is
/// 16 + 18 = 34 bits and does NOT fit the u32 `argmax_key` builds.
const INDEX_BITS: u32 = 18;
const INDEX_MASK: u64 = (1 << INDEX_BITS) - 1;

/// (logit desc, index ASC) packed so a plain unsigned compare gives the host's
/// order: bigger key = better, and among equal logits the SMALLER index must
/// win, so the index bits are stored inverted. Inverting this is invisible on a
/// tie-free row and changes the selected SET on a row with duplicate logits.
#[inline]
fn wide_key(bits: u16, index: usize) -> u64 {
    ((sortable(bits) as u64) << INDEX_BITS) | ((!index as u64) & INDEX_MASK)
}

// ---------------------------------------------------------------------------
// Candidate B: bounded heap over raw fp16 bits. No `Vec<f32>`, no index vector,
// and the NaN check the host performs at `sampler.rs:178` folds in for free.
// ---------------------------------------------------------------------------

fn sift_down_key(heap: &mut [(u64, u32)], mut parent: usize) {
    loop {
        let left = 2 * parent + 1;
        let right = left + 1;
        let mut worst = parent;
        if left < heap.len() && heap[left] < heap[worst] {
            worst = left;
        }
        if right < heap.len() && heap[right] < heap[worst] {
            worst = right;
        }
        if worst == parent {
            return;
        }
        heap.swap(parent, worst);
        parent = worst;
    }
}

/// Returns the order and whether any element was NaN, which is exactly the
/// condition `Sampler::sample` rejects with `InvalidLogits`.
fn top_k_u16(raw: &[u16], top_k: usize) -> (Vec<usize>, bool) {
    let keep = top_k.min(raw.len());
    let mut heap: Vec<(u64, u32)> = Vec::with_capacity(keep);
    let mut saw_nan = false;
    for (index, bits) in raw.iter().enumerate() {
        let key = wide_key(*bits, index);
        saw_nan |= (key >> INDEX_BITS) == 0;
        let candidate = (key, index as u32);
        if heap.len() < keep {
            heap.push(candidate);
            if heap.len() == keep {
                for parent in (0..keep / 2).rev() {
                    sift_down_key(&mut heap, parent);
                }
            }
        } else if heap[0] < candidate {
            heap[0] = candidate;
            sift_down_key(&mut heap, 0);
        }
    }
    heap.sort_unstable_by(|a, b| b.cmp(a));
    (heap.into_iter().map(|(_, index)| index as usize).collect(), saw_nan)
}

// ---------------------------------------------------------------------------
// Candidate D: two-pass counting select. Strictly O(n) on EVERY input, which
// the bounded heap is not. The 16-bit sortable key is already an
// order-preserving integer, so coarse buckets locate the rank-`k` threshold in
// one pass and a second pass keeps only what is at or above it.
// ---------------------------------------------------------------------------

const BUCKETS: usize = 4096;

fn top_k_bucket(raw: &[u16], top_k: usize) -> (Vec<usize>, bool) {
    let keep = top_k.min(raw.len());
    let mut counts = vec![0u32; BUCKETS];
    let mut saw_nan = false;
    for bits in raw {
        let key = sortable(*bits);
        saw_nan |= key == 0;
        counts[(key >> 4) as usize] += 1;
    }
    let mut covered = 0u32;
    let mut boundary = 0usize;
    while boundary < BUCKETS {
        covered += counts[BUCKETS - 1 - boundary];
        if covered >= keep as u32 {
            break;
        }
        boundary += 1;
    }
    let cut_key = ((BUCKETS - 1 - boundary) as u16) << 4;
    let mut candidates: Vec<(u64, u32)> = Vec::with_capacity(keep * 8);
    for (index, bits) in raw.iter().enumerate() {
        if sortable(*bits) >= cut_key {
            candidates.push((wide_key(*bits, index), index as u32));
        }
    }
    candidates.sort_unstable_by(|a, b| b.cmp(a));
    candidates.truncate(keep);
    (candidates.into_iter().map(|(_, index)| index as usize).collect(), saw_nan)
}

// ---------------------------------------------------------------------------

/// Median (and min) of `iters` samples. AGENTS.md 10.2.5: a single reading is
/// not a result.
fn time_ms<T>(iters: u32, mut body: impl FnMut() -> T) -> (f64, f64) {
    for _ in 0..5 {
        std::hint::black_box(body());
    }
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let started = Instant::now();
        std::hint::black_box(body());
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (samples[iters as usize / 2], samples[0])
}

/// Prints the decomposition. Timings are informational; the correctness gates
/// are the other tests in this file.
#[test]
fn probe_sampling_phase_decomposition() {
    let config = SamplerConfig { temperature: 1.0, top_k: TOP_K, top_p: 0.95 };

    println!("\n=== B1: host sampling phase decomposition, ms/token ===");
    println!("262144-wide row, top_k=64, T=1.0, top_p=0.95; median of 41 (min in parens)\n");
    println!(
        "{:<11} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "row", "widen", "is_fin", "top_k*", "sample()", "heap f32", "u16 heap", "bucket"
    );

    for (name, raw) in [
        ("uniform", row_uniform()),
        ("tailed", row_tailed()),
        ("ascending", row_ascending()),
        ("descending", row_descending()),
    ] {
        let logits = widen(&raw);
        let mut sampler = Sampler::new(config, 1).expect("sampler");

        let (widen_ms, _) = time_ms(41, || widen(&raw));
        let (finite_ms, _) = time_ms(41, || {
            // The validity guard at `sampler.rs:178`, verbatim: one pass, one
            // `is_finite` per element.
            let bad = logits.is_empty() || logits.iter().any(|value| !value.is_finite());
            std::hint::black_box(bad)
        });
        let (top_k_ms, _) = time_ms(41, || top_k_order(&logits, TOP_K));
        let (sample_ms, _) = time_ms(41, || sampler.sample(&logits));
        let (heap_ms, _) = time_ms(41, || top_k_heap_f32(&logits, TOP_K));
        let (u16_ms, _) = time_ms(41, || top_k_u16(&raw, TOP_K));
        let (bucket_ms, _) = time_ms(41, || top_k_bucket(&raw, TOP_K));

        println!(
            "{:<11} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            name, widen_ms, finite_ms, top_k_ms, sample_ms, heap_ms, u16_ms, bucket_ms
        );
    }
    println!("\ntop_k* = the shipped `select_nth_unstable_by`, which is ~95% of `sample()`.");
    println!("sample() = the shipped `Sampler::sample` end to end (its own phases are");
    println!("           cache-coupled, so the columns do not sum to it).");

    // The 64-element tail, timed directly: a single call is below timer
    // resolution, so time 1000 and divide.
    let raw = row_tailed();
    let logits = widen(&raw);
    let order = top_k_order(&logits, TOP_K);
    let mut samples = Vec::new();
    for _ in 0..41 {
        let started = Instant::now();
        for _ in 0..1000 {
            // `sampler.rs:187-215`, verbatim.
            let maximum = logits[order[0]];
            let mut weights = order
                .iter()
                .map(|index| ((logits[*index] - maximum) / config.temperature).exp())
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
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
            let running = weights.iter().sum::<f32>();
            std::hint::black_box((selected, selected_total, running));
        }
        samples.push(started.elapsed().as_secs_f64() * 1e6 / 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\n64-element tail (exp + softmax + top-p walk + inverse-CDF): {:.4} us/call", samples[20]);
    println!("  -> 0.007% of `sample()`. Not a candidate for any of the 2.87 ms.");
}

// ---------------------------------------------------------------------------
// Correctness gates.
// ---------------------------------------------------------------------------

/// A candidate is only admissible if it returns the SHIPPED vector. Equality of
/// the vector, not of the sampled token, is the claim: `Sampler::sample` is a
/// pure function of (order, logits, config, rng state), so an identical order
/// forces an identical token and identical `--seed` reproducibility.
///
/// The tie-dense rows are the point of this test. A wrong index tie-break or a
/// max-heap where a min-heap was meant produces a *plausible* order on a
/// tie-free row and a different candidate SET on a tie-dense one. Both bugs were
/// actually written and caught while producing this probe.
#[test]
fn candidate_selection_is_bit_identical_to_the_shipped_one() {
    let mut cases: Vec<(String, Vec<u16>)> = vec![
        ("uniform".into(), row_uniform()),
        ("tailed".into(), row_tailed()),
        ("ascending".into(), row_ascending()),
        ("descending".into(), row_descending()),
        ("all_equal".into(), vec![half::f16::from_f64(3.5).to_bits(); 4096]),
    ];
    // Exact ties straddling the top_k boundary: the only place a tie-break
    // difference changes the SET.
    let mut boundary_ties: Vec<f32> = widen(&row_tailed())[..4096].to_vec();
    for slot in boundary_ties.iter_mut() {
        *slot = 7.25f32;
    }
    boundary_ties[TOP_K - 1] = 8.0;
    boundary_ties[TOP_K + 1] = 8.0;
    cases.push((
        "boundary_ties".into(),
        boundary_ties.iter().map(|b| half::f16::from_f32(*b).to_bits()).collect(),
    ));
    // +/-0.0 exactly on the boundary. `value_key` folds them onto one key; a
    // transcription that omits the fold ranks -0.0 below +0.0 and diverges here.
    let mut signed_zero: Vec<f32> = widen(&row_tailed())[..4096].to_vec();
    signed_zero[TOP_K - 1] = -0.0;
    signed_zero[TOP_K + 1] = 0.0;
    cases.push((
        "signed_zero_at_boundary".into(),
        signed_zero.iter().map(|b| half::f16::from_f32(*b).to_bits()).collect(),
    ));

    let mut checked = 0_usize;
    for (name, raw) in &cases {
        let logits = widen(raw);
        for k in [1_usize, 2, 63, 64, 65, 1024] {
            if k > logits.len() {
                continue;
            }
            let shipped = top_k_order(&logits, k);
            assert_eq!(top_k_heap_f32(&logits, k), shipped, "{name}: heap_f32 at k={k}");
            let (u16_order, saw_nan) = top_k_u16(raw, k);
            assert!(!saw_nan, "{name}: benign row reported NaN at k={k}");
            assert_eq!(u16_order, shipped, "{name}: u16 heap at k={k}");
            let (bucket_order, saw_nan_bucket) = top_k_bucket(raw, k);
            assert!(!saw_nan_bucket, "{name}: benign row reported NaN (bucket) at k={k}");
            assert_eq!(bucket_order, shipped, "{name}: bucket select at k={k}");
            checked += 1;
        }
    }
    assert!(checked >= 40, "expected the row x k grid to be exercised, got {checked}");
    println!("candidate selections == shipped selection: {checked} (row x k) combinations");
}

/// `sortable(x) == 0` must mean exactly "x is NaN", because that identity is
/// what lets any key-based reduction reproduce the host's `InvalidLogits`
/// rejection. Checked over all 65536 fp16 bit patterns.
#[test]
fn key_zero_is_exactly_the_nan_set() {
    for bits in 0..=u16::MAX {
        let is_nan = (bits >> 10) & 0x1F == 0x1F && (bits & 0x03FF) != 0;
        assert_eq!(
            sortable(bits) == 0,
            is_nan,
            "sortable(0x{bits:04x}) == 0 disagrees with the NaN test"
        );
    }
}

/// The key map must be order-preserving against the host's own comparison,
/// including the ties the host resolves by index. Exhaustive, not sampled: a
/// map that inverts on one exotic pattern would silently change the selected
/// set at runtime with no visible symptom.
#[test]
fn sortable_preserves_the_host_order_exhaustively() {
    let mut mapped: Vec<(u16, u16)> = (0..=u16::MAX).map(|bits| (sortable(bits), bits)).collect();
    mapped.sort_unstable();
    for pair in mapped.windows(2) {
        let (key_a, bits_a) = pair[0];
        let (key_b, bits_b) = pair[1];
        // Key 0 is the NaN bucket and collapses all 2046 NaN patterns; comparing
        // them numerically is meaningless, so the claim covers finite keys.
        if key_a == 0 || key_b == 0 {
            continue;
        }
        let value_a = half::f16::from_bits(bits_a).to_f32();
        let value_b = half::f16::from_bits(bits_b).to_f32();
        if key_a < key_b {
            assert!(
                value_a < value_b,
                "sortable inverted a strict pair: 0x{bits_a:04x} ({value_a}) mapped \
                 below 0x{bits_b:04x} ({value_b})"
            );
        } else {
            assert_eq!(
                value_a, value_b,
                "sortable collided two DIFFERENT f16 values: 0x{bits_a:04x} \
                 ({value_a}) and 0x{bits_b:04x} ({value_b}) share a key, so one of \
                 them would be unreachable by any key-based top-k"
            );
        }
    }
}

/// `value_key` computes `sortable16` on a 32-bit `uint`, so `~bits` sets bits
/// 16..31 for a negative input and the returned key lands at >= 0xFFFF0000 --
/// ABOVE every positive key. The existing greedy path never notices, because
/// `argmax_key` immediately does `value_key(bits) << 16`, which discards exactly
/// those bits.
///
/// A kernel that consumes `value_key` WITHOUT packing through `argmax_key` -- a
/// device top-64, for instance -- would rank every negative logit above every
/// positive one and return confident garbage. Pinned here so the hazard is
/// stated in a test rather than only in prose.
#[test]
fn value_key_upper_bits_are_garbage_for_negatives() {
    let negative: u16 = 0xF800; // -2.0 in fp16
    let positive: u16 = 0x4000; // +2.0 in fp16

    assert!(
        value_key(negative) > value_key(positive),
        "value_key is expected to return a 32-bit value whose top half is \
         meaningless for negatives; if this ever starts holding, the << 16 in \
         argmax_key is no longer load-bearing and the note above is stale"
    );
    // The order only becomes correct once the garbage half is discarded, which is
    // what `argmax_key` does with its `<< 16`.
    assert!(sortable(negative) < sortable(positive), "sortable must be order-preserving");
    assert!(
        (value_key(negative) << 16) < (value_key(positive) << 16),
        "argmax_key's << 16 must repair the ordering"
    );
}

/// A u32 total order over the full row does not exist: 16 bits of value plus the
/// 18 bits vocab 262144 needs for the index is 34. This is why a device top-64
/// must extend the existing two-level 65536-slot structure (where
/// `index & 0xFFFF` IS unique inside a slot, so the u32 key is exact there)
/// rather than doing a flat u32 reduction over the whole row.
#[test]
fn a_u32_key_cannot_total_order_the_full_row() {
    assert_eq!(VOCAB, 1 << 18, "this argument is about an 18-bit index");
    assert!(
        16 + 18 > 32,
        "16 value bits + 18 index bits must exceed u32, else a flat u32 key is \
         possible and the slot structure is unnecessary"
    );
    // Within one 65536-wide slot the u32 key IS exact, which is the property the
    // device path actually relies on.
    let mut slot: Vec<u32> = (0..65536u32)
        .map(|index| argmax_key_for(0xFC00 - (index % 0x0400) as u16, index))
        .collect();
    slot.sort_unstable();
    slot.dedup();
    assert_eq!(
        slot.len(),
        65536,
        "inside a 65536-wide slot the (value, inverted-index) key must be unique"
    );
}

fn argmax_key_for(bits: u16, index: u32) -> u32 {
    (value_key(bits) << 16) | ((0xFFFFu32.wrapping_sub(index & 0xFFFF)) & 0xFFFF)
}

/// Question 4, settled by construction and then measured.
///
/// `sample` computes `w_i = exp((l_i - m) / T)` with `T > 0` and `m` the
/// maximum, both independent of `i`, so `i -> j => w_i < w_j` IN REAL
/// ARITHMETIC. The order is therefore identical, ties included.
///
/// It is NOT identical in f32. `exp` underflows to exactly `0.0f32` once its
/// argument is below about -103, so at small temperature thousands of distinct
/// logits collapse onto one weight, the weight ordering degenerates into an
/// index-order tie among thousands of zeros, and a top-k taken on the computed
/// probability selects a completely different candidate set.
///
/// The rule for a device reduction is therefore not "top-k by probability, it is
/// close enough": it must rank by the raw sortable logit key, which is what
/// `top_k_order` does and what is therefore distribution-identical at every
/// temperature.
#[test]
fn top_k_by_logit_and_by_softmax_probability_diverge_only_via_f32_underflow() {
    for (temperature, expect_divergence) in
        [(0.05_f32, true), (0.5, false), (1.0, false), (2.0, false), (10.0, false)]
    {
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut divergent = 0_usize;
        let mut set_diff_total = 0_usize;
        let mut underflow_total = 0_usize;
        for _trial in 0..100 {
            // Tie-dense on purpose: fp16 has 10 mantissa bits, so a 2-unit spread
            // over 4096 entries guarantees duplicates.
            let raw: Vec<u16> = (0..4096)
                .map(|index| {
                    let base = if index % 97 == 0 { 9.5 } else { -3.0 + (next() * 2.0) as f32 };
                    half::f16::from_f64((base as f64) + 1e-7 * (next() - 0.5)).to_bits()
                })
                .collect();
            let logits = widen(&raw);
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weight = |index: &usize| ((logits[*index] - maximum) / temperature).exp();

            underflow_total += logits
                .iter()
                .filter(|value| (((**value - maximum) / temperature).exp()) == 0.0)
                .count();

            let by_logit = top_k_order(&logits, TOP_K);
            let cmp = |left: &usize, right: &usize| {
                weight(right)
                    .partial_cmp(&weight(left))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.cmp(right))
            };
            let mut by_weight: Vec<usize> = (0..logits.len()).collect();
            by_weight.select_nth_unstable_by(TOP_K, cmp);
            by_weight.truncate(TOP_K);
            by_weight.sort_by(cmp);

            if by_logit != by_weight {
                divergent += 1;
                let left: std::collections::BTreeSet<usize> = by_logit.iter().copied().collect();
                let right: std::collections::BTreeSet<usize> = by_weight.iter().copied().collect();
                set_diff_total += left.symmetric_difference(&right).count();
            }
        }
        let mean_set_diff = set_diff_total as f64 / divergent.max(1) as f64;
        if expect_divergence {
            assert_eq!(
                divergent, 100,
                "T={temperature}: expected every row to diverge (f32 underflow), got {divergent}"
            );
            assert!(
                mean_set_diff > (TOP_K / 2) as f64,
                "T={temperature}: divergence is supposed to be severe, mean |set diff| was \
                 {mean_set_diff:.1} of {TOP_K}"
            );
        } else {
            assert_eq!(
                divergent, 0,
                "T={temperature}: ordering by logit and by probability must agree exactly \
                 away from the underflow regime, but {divergent}/100 rows differed"
            );
        }
        println!(
            "T={temperature:<5}: {divergent:>3}/100 rows differ, mean |set diff| {mean_set_diff:>5.1} \
             of {TOP_K}, mean underflows/row {:>7.1} of 4096",
            underflow_total as f64 / 100.0
        );
    }
}