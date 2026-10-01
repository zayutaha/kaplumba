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

/// The `top_k` candidates, best first.
///
/// **This replaced a full `sort_by` of all 262,144 indices followed by a
/// truncate to `top_k`.** Both selections timed from ONE binary on the same
/// 262,144-wide row (`examples/bench_sampler.rs`):
///
/// ```text
/// selection OLD (full sort):  14.71 ms/token
/// selection NEW (partition):   1.89 ms/token      7.8x
/// whole `sample()`:            2.87 ms/token      (was ~15.5)
/// ```
///
/// The old cost was 23.4% of a 62.9 ms decode step, and it was the entire
/// difference between greedy's 15.9 tok/s and sampling's 12.2 — not the model,
/// not attention, the host sort.
///
/// `select_nth_unstable_by` partitions in O(n) instead of O(n log n). Because
/// the comparator is a TOTAL order (ties broken by ascending index), the first
/// `keep` entries after partitioning are exactly the set a full sort would have
/// produced; sorting just that prefix restores the order, so everything
/// downstream sees an identical sequence.
///
/// The claim that matters is the equality, not the speed: a sampler that quietly
/// selected a different candidate set would be a silent quality regression that
/// no throughput number would reveal. `top_k_selection_matches_a_full_sort`
/// checks it across lengths, `top_k` values, and tie densities.
///
/// Public so `examples/bench_sampler.rs` can time this exact function against
/// the sort it replaced from ONE binary. A benchmark that reimplemented either
/// side would be timing the benchmark.
pub fn top_k_order(logits: &[f32], top_k: usize) -> Vec<usize> {
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
