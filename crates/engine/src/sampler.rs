use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SamplerError {
    #[error("logits must be non-empty and finite")]
    InvalidLogits,
    #[error("temperature must be finite and non-negative")]
    InvalidTemperature,
    #[error("top-k must be positive")]
    InvalidTopK,
    #[error("top-p must be in (0, 1]")]
    InvalidTopP,
    #[error("top-k {top_k} exceeds the device sampler's limit of {limit}")]
    DeviceTopKTooLarge { top_k: usize, limit: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
        }
    }
}

pub struct Sampler {
    config: SamplerConfig,
    state: u64,
}

/// Ascending-in-value `u32` key for a logit. Two properties, both load-bearing.
///
/// * **Order preserving.** For any non-NaN pair, `key(a) < key(b)` exactly when
///   `a < b`: flipping *every* bit of a negative float and only the sign bit of a
///   positive one turns IEEE-754's sign-magnitude layout into two's-complement
///   order. That is what lets the selection compare integers instead of doing two
///   random gathers into the `f32` row per comparison, which is what the
///   comparator this replaced spent its time on.
/// * **Equal exactly where `partial_cmp` says equal.** `-0.0` sorts below `+0.0`
///   as a key but *compares equal*, and the shipped comparator then breaks that
///   tie by INDEX. So `-0.0` is folded onto `+0.0` (via `value == 0.0`, which is
///   true for both zeros). Without the fold, a row whose `top_k` boundary falls
///   inside the zeros selects a different SET than the sort, `sample()` returns a
///   different token for the same `--seed`, and nothing raises an error.
///
/// NaN has no place in this order — the shipped comparator calls every comparison
/// involving it `Equal` via `unwrap_or`, which is not even a strict weak ordering,
/// so `select_nth_unstable_by` had no defined answer for it either. `sample`
/// rejects non-finite logits (`sampler.rs:198`) before reaching here; `top_k_order`
/// inherits the same precondition.
#[inline]
fn order_key(value: f32) -> u32 {
    let bits = if value == 0.0 { 0.0_f32.to_bits() } else { value.to_bits() };
    bits ^ ((((bits as i32) >> 31) as u32) | 0x8000_0000)
}

/// The `top_k` candidates, best first: **logit descending, index ascending on
/// ties**, which is a TOTAL order, so the returned vector is uniquely determined
/// and any correct selection is obliged to return this exact sequence.
///
/// # What this replaced, and why it was still 1.9 ms
///
/// Three selectors, all timed from ONE binary (`tests/probe_topk_equality.rs`) on the same
/// 262,144-wide fp16-derived row at `top_k = 64`; median of 41 **interleaved** samples,
/// min..max in brackets (`--release`; AGENTS.md 10.2.5 — decode on this box has been measured
/// from 3.7 to 6.4 tok/s from the same binary minutes apart, so a single reading is not a
/// result):
///
/// ```text
///                                  tailed row                uniform row
/// full `sort_by` of every index    14.71 ms (older run, see examples/bench_sampler.rs)
/// `select_nth_unstable_by`         1.790 [1.672..1.967]       2.024 [1.931..2.284]
/// this (two-pass counting)         0.251 [0.225..0.328]       0.254 [0.224..0.307]
/// whole `sample()`                 1.848 -> 0.322             2.069 -> 0.321
/// ```
///
/// The partition was already O(n), so this was never a sort-bound problem. It was bound by
/// the two things an Introselect over indices cannot avoid: it builds a **2 MiB `Vec<usize>`**
/// (8 B x 262,144) on every call before it can partition anything, and its comparator does
/// **two indirect gathers into the 1 MiB `f32` row per comparison**. Counting the rank-`k`
/// threshold directly removes both — no index vector, no per-comparison gather — and leaves
/// at most a few hundred indices to rank at the end. Measured peak allocation per call:
/// **2,097,152 B -> 263,160 B**, and unchanged when the row is doubled
/// (`tests/probe_topk_alloc.rs`).
///
/// # The alternative, and why not
///
/// A bounded binary heap of `f32`/`index` pairs was measured on the same binary: 0.13 ms on
/// realistic rows (11-14x faster than the partition) but **4.95 ms on an ascending row, 2.6x
/// SLOWER than the partition's own worst**, because a data-dependent sift-down runs `log k`
/// times per late arrival. That is a production regression waiting for a logit row whose best
/// tokens sit at high token ids, to win 0.12 ms on the typical row. The counting select's own
/// worst sample over every row measured is 0.808 ms against the partition's 2.284 ms — 2.8x
/// better tail for 0.12 ms of typical-case cost. `spec/decode-ceiling/findings/B3-topk.md`.
///
/// Its one regression, stated rather than buried: on **presorted** rows Introselect is a single
/// pass, so the partition wins there (0.214 vs 0.644 ms ascending). A logit row is indexed by
/// token id and is not presorted.
///
/// # Why it is O(n) on every input
///
/// Two counting passes over `order_key`, 16 bits at a time, each a branch-free
/// histogram with a fixed 65,536-entry table. The first pass locates the bucket
/// holding the `top_k`-th best key; if that bucket is wider than `RANK_DIRECT`
/// (512) the second pass pins the exact key inside it (16 + 16 = 32, so two
/// passes always suffice). Then one gather pass keeps the winners and ranks them
/// with the shipped comparator. There is no data-dependent *structure* anywhere —
/// no heap to sift, no recursion — which is what bounds the tail: worst sample
/// over every row measured is 0.808 ms against the partition's 2.284 ms.
///
/// The passes themselves are not free of the input: consecutive elements landing
/// in the same histogram bucket serialise on store-to-load forwarding, which is
/// why a smooth (ascending) row costs 0.644 ms where a random one costs 0.251 ms.
/// That is a 2.5x spread on an input a model does not produce; the heap's spread
/// was 36x (0.13 -> 4.95 ms).
///
/// # The claim that matters
///
/// Equality, not speed. A sampler that quietly selected a different candidate set
/// would be a silent quality regression: `--seed` reproducibility depends on this
/// vector, no error is raised, and throughput would look fine. `sample` is a pure
/// function of `(order, logits, config, rng state)`, so an identical `order` forces
/// an identical token.
///
/// `tests/probe_topk_equality.rs` checks the returned vector is byte-identical to
/// the deleted partition across tie-dense rows, adversarial orderings, signed
/// zeros, `f32::MIN`/`MAX`/`NEG_INFINITY`, all-equal rows, `k = 1`, `k >= len`, and
/// fp16-derived logit distributions, and every one of those gates has been observed
/// failing against a deliberately broken selector (AGENTS.md 3.4).
/// `tests/probe_topk_alloc.rs` pins the allocation budget, because reverting this
/// function to the partition keeps every equality gate green and puts the 2 MiB
/// per-token allocation back.
///
/// Requires FINITE logits. `sample` enforces that (`sampler.rs:198`); under a NaN
/// the shipped comparator's `unwrap_or(Equal)` is not even a strict weak ordering,
/// so the old partition had no defined answer to reproduce either.
///
/// Public so `examples/bench_sampler.rs` and `tests/probe_topk_equality.rs` can
/// time this exact function against the selections it replaced, from one binary.
/// A benchmark that reimplemented either side would be timing the benchmark.
pub fn top_k_order(logits: &[f32], top_k: usize) -> Vec<usize> {
    let keep = top_k.min(logits.len());
    // The shipped order, verbatim: logit DESCENDING, then index ASCENDING.
    let better = |left: &usize, right: &usize| {
        logits[*right]
            .partial_cmp(&logits[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    };
    if keep == 0 {
        return Vec::new();
    }
    if keep >= logits.len() {
        // `keep` covers the whole row, so there is nothing to partition. This is
        // the one branch that still sorts everything, and it is the branch the
        // partition also took when `top_k >= logits.len()`.
        let mut order = (0..logits.len()).collect::<Vec<_>>();
        order.sort_by(better);
        return order;
    }

    // A bucket no wider than this is ranked by sorting it instead of counting a
    // second histogram. Realistic logits put the `top_k`-th best value in a
    // bucket a few dozen elements wide, so the usual path is ONE counting pass.
    const RANK_DIRECT: usize = 512;
    const RADIX_BITS: u32 = 16;
    const RADIX: usize = 1 << RADIX_BITS;
    const DIGIT_MASK: u32 = RADIX as u32 - 1;

    debug_assert!(
        logits.len() < u32::MAX as usize,
        "bucket counts are u32, so a row of 2^32 logits could overflow one"
    );

    let mut counts = vec![0_u32; RADIX];
    // Winners settled so far in strictly higher buckets, and winners still owed
    // from the bucket currently being examined. `above + need == keep` always.
    let mut above = 0_usize;
    let mut need = keep;
    // Bit offset of the digit being counted, and the bits above it already fixed.
    let mut shift = RADIX_BITS;
    let mut prefix = 0_u32;
    let mut prefix_bits = 0_u32;

    loop {
        if prefix_bits == 0 {
            for value in logits {
                counts[((order_key(*value) >> shift) & DIGIT_MASK) as usize] += 1;
            }
        } else {
            counts.fill(0);
            for value in logits {
                let key = order_key(*value);
                if key >> prefix_bits == prefix {
                    counts[((key >> shift) & DIGIT_MASK) as usize] += 1;
                }
            }
        }
        // Descend from the top digit: everything above it wins outright, and the
        // first digit whose running count reaches `need` holds the `need`-th best
        // key. The bucket is never empty, so this always stops at or above digit 0.
        let mut digit = RADIX - 1;
        let mut skipped = 0_usize;
        let mut within = counts[digit] as usize;
        while skipped + within < need {
            skipped += within;
            digit -= 1;
            within = counts[digit] as usize;
        }
        above += skipped;
        need -= skipped;
        // The chosen digit joins the fixed prefix, so `prefix` now names the
        // winning bucket in both the break and the refine case below.
        prefix = (prefix << RADIX_BITS) | digit as u32;

        if shift > 0 && within > RANK_DIRECT {
            // Too crowded to rank directly: count the next 16 bits of this bucket
            // alone. 16 + 16 = 32, so this second pass pins the EXACT key.
            prefix_bits += RADIX_BITS;
            shift -= RADIX_BITS;
            continue;
        }
        break;
    }

    // `[low, end)` is the winning bucket. `shift == 0` means it is the single
    // exact key. `u64`, because `end` can be 2^32.
    let low = (prefix << shift) as u64;
    let end = low + (1_u64 << shift);
    let mut order: Vec<usize> = Vec::with_capacity(above);
    let mut bucket: Vec<usize> = Vec::with_capacity(need);
    let mut taken = 0_usize;
    for (index, value) in logits.iter().enumerate() {
        let key = order_key(*value) as u64;
        if key >= end {
            order.push(index);
        } else if key >= low {
            // With `shift > 0` the bucket is at most `RANK_DIRECT` wide and every
            // member is a candidate. With `shift == 0` every member has the SAME
            // key, so the first `need` in ascending index order are exactly the
            // ones the index tie-break keeps -- which is also what keeps this Vec
            // at `need` elements instead of rebuilding the 2 MiB one this function
            // exists to avoid.
            if shift > 0 || taken < need {
                taken += 1;
                bucket.push(index);
            }
        }
    }
    debug_assert_eq!(order.len(), above, "gathered the wrong number of winners");
    order.sort_by(better);
    if shift > 0 {
        // A wide bucket collects every member and ranks them; only the exact case
        // above is forced to exactly `need` candidates.
        debug_assert!(bucket.len() >= need, "bucket lost candidates");
        bucket.sort_by(better);
        bucket.truncate(need);
    } else {
        debug_assert_eq!(bucket.len(), need, "exact bucket must yield exactly `need`");
    }
    // Every key in `order` is >= `end` and every key in `bucket` is < `end`, so
    // concatenating two correctly-ordered runs keeps the whole thing ordered.
    order.extend(bucket);
    order
}

/// Largest `top_k` the device sampler accepts.
///
/// The device path reduces the vocabulary to a bounded candidate array (a
/// per-block top-`k`, then a global top-`k` over those), so `top_k` bounds the
/// shared-memory staging array, not the vocabulary. A `top_k` above this is a
/// **fall back to the host sampler with the non-pipelined loop**, not a wider
/// device sort - building a 262,144-wide sort for a config nobody ships is how
/// the fast path stops being reviewable.
///
/// `spec/device-sampler` R7. The device path is only distribution-identical to
/// `Sampler::sample` when it applies top-k before top-p, over the same
/// `top_k` candidates (`sampler.rs:110-128`).
pub const MAX_DEVICE_TOP_K: usize = 1024;

/// The `SamplerConfig` fields the device sampler needs, in the layout its
/// parameter buffer wants.
///
/// Separate from [`SamplerConfig`] because `top_k` is a `u32` on the wire and
/// because the device path has a bound the host path does not. Constructed only
/// through [`SamplerParams::from_config`], so the validation cannot be
/// skipped by a caller that had a good reason.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
}

impl SamplerParams {
    /// Device parameters for `config`, or an error if the device path cannot
    /// serve this configuration for a `vocab`-wide row.
    ///
    /// `top_k` is **not** clamped: a `top_k` the device path cannot stage is an
    /// error the caller turns into "run the host sampler on the non-pipelined
    /// loop", and silently sampling from a different candidate set than the
    /// configuration asked for is the one failure this whole change must not
    /// have.
    pub fn from_config(config: SamplerConfig, vocab: usize) -> Result<Self, SamplerError> {
        validate_config(config)?;
        let params = Self {
            temperature: config.temperature,
            top_k: u32::try_from(config.top_k).map_err(|_| SamplerError::InvalidTopK)?,
            top_p: config.top_p,
        };
        params.validate(vocab)?;
        Ok(params)
    }

    /// Reject a configuration the device path cannot honour exactly.
    ///
    /// `vocab` bounds `top_k` the way the host path does: a `top_k` at or above
    /// the vocabulary would need the device path to sort the whole row, which
    /// is exactly the thing it is built not to do.
    pub fn validate(&self, vocab: usize) -> Result<(), SamplerError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(SamplerError::InvalidTemperature);
        }
        if self.top_k == 0 {
            return Err(SamplerError::InvalidTopK);
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(SamplerError::InvalidTopP);
        }
        let limit = MAX_DEVICE_TOP_K.min(vocab.saturating_sub(1));
        if self.top_k as usize > limit {
            return Err(SamplerError::DeviceTopKTooLarge {
                top_k: self.top_k as usize,
                limit,
            });
        }
        Ok(())
    }
}

impl Sampler {
    pub fn new(config: SamplerConfig, seed: u64) -> Result<Self, SamplerError> {
        validate_config(config)?;
        Ok(Self {
            config,
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        })
    }

    pub fn config(&self) -> SamplerConfig {
        self.config
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<u32, SamplerError> {
        if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
            return Err(SamplerError::InvalidLogits);
        }
        if self.config.temperature == 0.0 {
            return Ok(argmax(logits));
        }
        // `top_k_order` partitions in O(n) instead of sorting all 262,144
        // indices. See that function for the measured cost and the reasoning.
        let order = top_k_order(logits, self.config.top_k);
        let maximum = logits[order[0]];
        let mut weights = order
            .iter()
            .map(|index| ((logits[*index] - maximum) / self.config.temperature).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f32>();
        if !total.is_finite() || total <= 0.0 {
            return Err(SamplerError::InvalidLogits);
        }
        let mut cumulative = 0.0;
        let mut selected = 0;
        for (index, weight) in weights.iter().enumerate() {
            cumulative += *weight / total;
            selected = index;
            if cumulative >= self.config.top_p {
                break;
            }
        }
        weights.truncate(selected + 1);
        let selected_total = weights.iter().sum::<f32>();
        let target = self.next_f64() as f32 * selected_total;
        let mut running = 0.0;
        for (index, weight) in weights.iter().enumerate() {
            running += *weight;
            if target <= running {
                return Ok(order[index] as u32);
            }
        }
        Ok(order[selected] as u32)
    }

    fn next_f64(&mut self) -> f64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        let bits = value.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((bits >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

fn validate_config(config: SamplerConfig) -> Result<(), SamplerError> {
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(SamplerError::InvalidTemperature);
    }
    if config.top_k == 0 {
        return Err(SamplerError::InvalidTopK);
    }
    if !config.top_p.is_finite() || config.top_p <= 0.0 || config.top_p > 1.0 {
        return Err(SamplerError::InvalidTopP);
    }
    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    best as u32
}

/// The order `sample` would have produced with the original full sort, i.e. the
/// reference the partition is checked against.
///
/// Test-only, and *not* a reimplementation of anything on the hot path: it is
/// the deleted code, kept verbatim as the oracle. `cfg(test)` rather than a
/// `dead_code` allow, because a reference implementation that ships in the
/// binary is one more thing to keep in sync for no benefit.
#[cfg(test)]
fn reference_top_k(logits: &[f32], top_k: usize) -> Vec<usize> {
    let mut order = (0..logits.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        logits[*right]
            .partial_cmp(&logits[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    order.truncate(top_k.min(order.len()));
    order
}

#[cfg(test)]
mod tests {
    use super::{reference_top_k, Sampler, SamplerConfig};

    fn row(len: usize, distinct: usize) -> Vec<f32> {
        // `distinct` controls how many exact TIE groups there are, which is the
        // case a partition is most likely to get wrong: with a comparator that
        // broke ties arbitrarily, the selected SET could differ from the sort's
        // while the order still looked plausible.
        (0..len)
            .map(|index| ((index % distinct) as f32) * 0.25 - 8.0)
            .collect()
    }

    /// The partition-based top-k must select the same SET, in the same ORDER, as
    /// the full sort it replaced. This is a quality claim, not a speed claim: a
    /// sampler that quietly picked a different candidate set would be a silent
    /// output regression that no throughput number would reveal.
    #[test]
    fn top_k_selection_matches_a_full_sort() {
        for len in [1_usize, 2, 63, 64, 65, 1024, 262_144] {
            for top_k in [1_usize, 2, 63, 64, 65, 4096] {
                for distinct in [1_usize, 3, len] {
                    if top_k > len {
                        continue;
                    }
                    let logits = row(len, distinct);
                    let selected = super::top_k_order(&logits, top_k);
                    assert_eq!(
                        selected,
                        reference_top_k(&logits, top_k),
                        "top_k={top_k} len={len} distinct={distinct}: selection \
                         differs from the full sort"
                    );
                }
            }
        }
    }

    /// Every sample from a given seed must be identical before and after the
    /// change. The RNG is a xorshift whose sequence depends only on how many
    /// times it has been advanced, so a differing token means the selection fed
    /// different weights to the cumulative walk.
    #[test]
    fn sampling_is_reproducible_for_a_fixed_seed() {
        let logits = row(4096, 512);
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
        };
        let mut a = Sampler::new(config, 12345).expect("sampler");
        let mut b = Sampler::new(config, 12345).expect("sampler");
        for step in 0..200 {
            let left = a.sample(&logits).expect("sample");
            let right = b.sample(&logits).expect("sample");
            assert_eq!(left, right, "two identically-seeded samplers diverged at {step}");
        }
    }

    /// `top_k >= len` must not panic and must keep everything. `select_nth` is
    /// only called when `keep < len`, so this is the other half of that guard.
    #[test]
    fn top_k_larger_than_the_row_keeps_everything() {
        let logits = row(32, 32);
        let order = super::top_k_order(&logits, 1000);
        assert_eq!(order.len(), 32);
        assert_eq!(order, reference_top_k(&logits, 1000));
    }

    /// The greedy path is untouched by the change and must still be argmax.
    #[test]
    fn greedy_is_argmax_and_ignores_top_k() {
        let mut logits = vec![0.0_f32; 512];
        logits[300] = 9.0;
        for top_k in [1_usize, 64, 4096] {
            let mut sampler = Sampler::new(
                SamplerConfig {
                    temperature: 0.0,
                    top_k,
                    top_p: 0.95,
                },
                7,
            )
            .expect("sampler");
            assert_eq!(sampler.sample(&logits).expect("sample"), 300, "top_k={top_k}");
        }
    }
}

/// `order_key` is what replaces the per-comparison `f32` gathers, so it has to
/// agree with `partial_cmp` on both strict order and ties. Ties are the
/// dangerous half: a key that inverts `-0.0` against `+0.0` still sorts every
/// other row correctly, still returns a plausible top-k, and still samples
/// plausible text -- it just changes the candidate SET whenever the `top_k`
/// boundary falls inside the zeros.
#[cfg(test)]
mod order_key_tests {
    use super::order_key;

    /// The whole `f16` domain, which is the domain the logit row actually has:
    /// `generate.rs:326-330` widens fp16 to `f32` and the widening is exact.
    /// Exhaustive, because the failure mode is a single inverted pattern that
    /// would change one `--seed` output and nothing else.
    #[test]
    fn order_key_is_order_preserving_over_every_fp16_pattern() {
        let mut mapped: Vec<(u32, f32)> = (0..=u16::MAX)
            .map(|bits| {
                let value = half::f16::from_bits(bits).to_f32();
                (order_key(value), value)
            })
            .filter(|(_, value)| value.is_finite())
            .collect();
        // 65,536 patterns - 2,046 NaN - 2 infinity = 63,488 finite.
        assert_eq!(mapped.len(), 63_488, "the finite f16 domain changed size");
        mapped.sort_unstable_by_key(|(key, _)| *key);
        for pair in mapped.windows(2) {
            let ((key_a, value_a), (key_b, value_b)) = (pair[0], pair[1]);
            if key_a == key_b {
                assert_eq!(
                    value_a, value_b,
                    "order_key collided two DIFFERENT f32 values (0x{value_a:e}) and \
                     (0x{value_b:e}) at key 0x{key_a:08x}, so one of them would be \
                     unreachable by any key-based selection"
                );
                continue;
            }
            assert!(
                value_a <= value_b,
                "order_key inverted a strict pair: key 0x{key_a:08x} = {value_a:e} sorted \
                 BELOW key 0x{key_b:08x} = {value_b:e}"
            );
        }
    }

    /// The `f32`-only specials, including the two signed zeros. `partial_cmp`
    /// calls `-0.0` and `+0.0` equal, so `Equal` here demands `Equal` keys: that
    /// is the `-0.0` fold, asserted rather than described.
    #[test]
    fn order_key_agrees_with_partial_cmp_on_the_f32_specials() {
        let smallest = f32::from_bits(1);
        let values = [
            f32::NEG_INFINITY,
            f32::MIN,
            -f32::MAX,
            -1.0e30_f32,
            -65536.0,
            -1024.0,
            -1.0,
            -f32::MIN_POSITIVE,
            -smallest,
            -0.0,
            0.0,
            smallest,
            f32::MIN_POSITIVE,
            1.0,
            1024.0,
            65536.0,
            1.0e30_f32,
            f32::MAX,
            f32::INFINITY,
            // Values fp16 cannot hold, so the `f32` mantissa is exercised past the
            // 10 bits the row would ever carry.
            -1.000_000_1,
            1.000_000_1,
            3.141_592_7,
        ];
        for left in values {
            for right in values {
                let (key_left, key_right) = (order_key(left), order_key(right));
                match left.partial_cmp(&right).expect("all finite or infinities") {
                    std::cmp::Ordering::Less => assert!(
                        key_left < key_right,
                        "order_key inverted: {left:e} keyed 0x{key_left:08x} but \
                         {right:e} keyed 0x{key_right:08x}"
                    ),
                    std::cmp::Ordering::Equal => assert_eq!(
                        key_left, key_right,
                        "partial_cmp says {left:e} == {right:e}, so their keys must be \
                         equal or the index tie-break applies to the wrong side"
                    ),
                    std::cmp::Ordering::Greater => assert!(
                        key_left > key_right,
                        "order_key inverted: {left:e} keyed 0x{key_left:08x} but \
                         {right:e} keyed 0x{key_right:08x}"
                    ),
                }
            }
        }
    }

    /// The `f32` domain more broadly: a deterministic sweep of bit patterns,
    /// filtered to finite, sorted by key. Cheap insurance against a mantissa-range
    /// mistake that the fp16 sweep above cannot see (fp16 leaves 13 mantissa bits
    /// at zero, so it never tests the low ones).
    #[test]
    fn order_key_is_monotone_across_a_broad_f32_sweep() {
        let mut state = 0x243F_6A88_85A3_08D3_u64;
        let mut mapped: Vec<(u32, f32)> = (0..1 << 20)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value = f32::from_bits(state as u32);
                (order_key(value), value)
            })
            .filter(|(_, value)| value.is_finite())
            .collect();
        assert!(mapped.len() > 1 << 19, "the sweep lost too many patterns");
        mapped.sort_unstable_by_key(|(key, _)| *key);
        for pair in mapped.windows(2) {
            let ((key_a, value_a), (key_b, value_b)) = (pair[0], pair[1]);
            assert!(
                value_a <= value_b,
                "order_key inverted {value_a:e} (0x{key_a:08x}) below {value_b:e} \
                 (0x{key_b:08x})"
            );
            if key_a == key_b {
                assert_eq!(value_a, value_b, "two distinct f32 values share a key");
            }
        }
    }
}

#[cfg(test)]
mod device_params_tests {
    use super::{SamplerConfig, SamplerError, SamplerParams, MAX_DEVICE_TOP_K};

    const VOCAB: usize = 262_144;

    /// The shipped default must be servable on device, or the whole change is
    /// aimed at nothing.
    #[test]
    fn the_shipped_default_is_device_servable() {
        let params =
            SamplerParams::from_config(SamplerConfig::default(), VOCAB).expect("default config");
        assert_eq!(
            params,
            SamplerParams {
                temperature: 1.0,
                top_k: 64,
                top_p: 0.95
            }
        );
    }

    /// `top_k` above the device bound is an **error**, not a clamp. A clamp
    /// would sample from a different candidate set than the configuration asked
    /// for and report success, which is precisely the silent quality regression
    /// this change must not have (AGENTS.md §8).
    #[test]
    fn top_k_above_the_device_bound_is_an_error_not_a_clamp() {
        for top_k in [MAX_DEVICE_TOP_K + 1, 4096, 262_144] {
            let config = SamplerConfig {
                temperature: 1.0,
                top_k,
                top_p: 0.95,
            };
            assert_eq!(
                SamplerParams::from_config(config, VOCAB),
                Err(SamplerError::DeviceTopKTooLarge {
                    top_k,
                    limit: MAX_DEVICE_TOP_K,
                }),
                "top_k={top_k} must be rejected, not clamped"
            );
        }
        // ... and the boundary itself is accepted.
        assert!(SamplerParams::from_config(
            SamplerConfig {
                temperature: 1.0,
                top_k: MAX_DEVICE_TOP_K,
                top_p: 0.95,
            },
            VOCAB,
        )
        .is_ok());
    }

    /// A `top_k` at or above the vocabulary is the other half of the bound: the
    /// device path must not be asked to stage a candidate array as wide as the
    /// row it is reducing.
    #[test]
    fn top_k_at_or_above_the_vocabulary_is_rejected() {
        for (vocab, top_k) in [(64_usize, 64_usize), (64, 63), (64, 65), (64, 4096)] {
            let config = SamplerConfig {
                temperature: 1.0,
                top_k,
                top_p: 0.95,
            };
            let outcome = SamplerParams::from_config(config, vocab);
            if top_k < vocab && top_k <= MAX_DEVICE_TOP_K {
                assert!(outcome.is_ok(), "vocab={vocab} top_k={top_k} should be servable");
            } else {
                assert_eq!(
                    outcome,
                    Err(SamplerError::DeviceTopKTooLarge {
                        top_k,
                        limit: MAX_DEVICE_TOP_K.min(vocab - 1),
                    }),
                    "vocab={vocab} top_k={top_k}"
                );
            }
        }
    }

    /// Every rejection the host sampler already makes must survive the trip
    /// through `SamplerParams`, or a bad `--temperature` would reach the GPU.
    #[test]
    fn host_config_validation_is_not_bypassed() {
        for (config, expected) in [
            (
                SamplerConfig {
                    temperature: -1.0,
                    top_k: 64,
                    top_p: 0.95,
                },
                SamplerError::InvalidTemperature,
            ),
            (
                SamplerConfig {
                    temperature: f32::NAN,
                    top_k: 64,
                    top_p: 0.95,
                },
                SamplerError::InvalidTemperature,
            ),
            (
                SamplerConfig {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 0.95,
                },
                SamplerError::InvalidTopK,
            ),
            (
                SamplerConfig {
                    temperature: 1.0,
                    top_k: 64,
                    top_p: 0.0,
                },
                SamplerError::InvalidTopP,
            ),
            (
                SamplerConfig {
                    temperature: 1.0,
                    top_k: 64,
                    top_p: 1.5,
                },
                SamplerError::InvalidTopP,
            ),
        ] {
            assert_eq!(
                SamplerParams::from_config(config, VOCAB),
                Err(expected),
                "config={config:?} must be rejected the same way the host sampler rejects it"
            );
        }
    }
}
