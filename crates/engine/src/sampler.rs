//! Naive host sampler (T1.4 plumbing; the fast path is T4.3).
//!
//! Deterministic PCG32 + top-k/top-p over softcapped f32 logits.
//! `temp == 0` (or `top_k == 1`) is greedy argmax, which is what R8.1's
//! token-identity tests use. Defaults are the OQ4/DE9 primary sampling mode
//! (temp 1.0 / top-k 64 / top-p 0.95).
//!
//! Cost note: filters/sorts the full vocab on the host (~262k entries) per
//! token. Fine for the naive CPU reference path, replaced in T4.3.

/// Minimal PCG32 (J. O'Neill) — 8 bytes of state, no dependencies.
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64) -> Self {
        let mut s = Self {
            state: 0,
            inc: (seed << 1) | 1,
        };
        s.next_u32();
        s.state = s.state.wrapping_add(seed);
        s.next_u32();
        s
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, 1)` with 24-bit resolution (exactly representable f32).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleParams {
    /// 0 → greedy (argmax).
    pub temp: f32,
    /// 0 → no top-k truncation.
    pub top_k: usize,
    /// 0 or 1 → no top-p truncation.
    pub top_p: f32,
}

impl Default for SampleParams {
    fn default() -> Self {
        // OQ4 / DE9: primary sampling mode.
        Self {
            temp: 1.0,
            top_k: 64,
            top_p: 0.95,
        }
    }
}

impl SampleParams {
    /// Greedy (temp 0) — used by spec-decode identity tests (R8.1).
    pub fn greedy() -> Self {
        Self {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temp <= 0.0 || self.top_k == 1
    }
}

/// Pick one token id from `logits`.
///
/// Non-finite logits (NaN or ±inf) are treated as `-inf` — they can never be
/// selected unless *every* candidate is non-finite, in which case id 0 is
/// returned instead of panicking.
pub fn sample(logits: &[f32], p: &SampleParams, rng: &mut Pcg32) -> u32 {
    assert!(!logits.is_empty(), "sample: empty logits");
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        let v = if v.is_finite() { v } else { f32::NEG_INFINITY };
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    if p.is_greedy() {
        return best as u32;
    }

    let temp = p.temp.max(f32::MIN_POSITIVE);
    let mut cand: Vec<(u32, f32)> = Vec::with_capacity(logits.len());
    for (i, &v) in logits.iter().enumerate() {
        let v = if v.is_finite() { v } else { f32::NEG_INFINITY };
        cand.push((i as u32, ((v - best_v) / temp).exp()));
    }
    // Deterministic order: by weight desc, ties by token id asc.
    cand.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    if p.top_k > 0 && p.top_k < cand.len() {
        cand.truncate(p.top_k);
    }
    let total: f32 = cand.iter().map(|c| c.1).sum();
    if !(total > 0.0) || !total.is_finite() {
        return best as u32;
    }
    if p.top_p > 0.0 && p.top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut keep = cand.len();
        for (i, c) in cand.iter().enumerate() {
            cum += c.1;
            if cum / total >= p.top_p {
                keep = i + 1;
                break;
            }
        }
        cand.truncate(keep.max(1));
    }
    let total: f32 = cand.iter().map(|c| c.1).sum();
    if !(total > 0.0) || !total.is_finite() {
        return cand[0].0;
    }
    let mut u = rng.next_f32() * total;
    for c in cand.iter() {
        if u < c.1 {
            return c.0;
        }
        u -= c.1;
    }
    // f32 accumulation rounding: fall back to the last (smallest) candidate.
    cand[cand.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg32_is_deterministic_and_seed_sensitive() {
        let mut a = Pcg32::new(0);
        let mut b = Pcg32::new(0);
        let mut c = Pcg32::new(1);
        let sa: Vec<u32> = (0..8).map(|_| a.next_u32()).collect();
        let sb: Vec<u32> = (0..8).map(|_| b.next_u32()).collect();
        let sc: Vec<u32> = (0..8).map(|_| c.next_u32()).collect();
        assert_eq!(sa, sb, "same seed → same stream");
        assert_ne!(sa, sc, "different seed → different stream");
        assert!(sa.iter().any(|&x| x != 0), "not all-zero stream");
    }

    #[test]
    fn next_f32_is_in_unit_interval() {
        let mut r = Pcg32::new(42);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for _ in 0..10_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo < 0.1 && hi > 0.9, "poor coverage: {lo}..{hi}");
    }

    #[test]
    fn greedy_modes_ignore_rng() {
        let logits = [0.5f32, 3.0, -1.0, 3.0];
        let mut r1 = Pcg32::new(7);
        let mut r2 = Pcg32::new(99);
        assert_eq!(sample(&logits, &SampleParams::greedy(), &mut r1), 1);
        assert_eq!(sample(&logits, &SampleParams::greedy(), &mut r2), 1);
        let k1 = SampleParams {
            temp: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        assert_eq!(sample(&logits, &k1, &mut r1), 1, "top_k=1 → argmax");
        assert_eq!(sample(&logits, &k1, &mut r2), 1);
        // Lowest index wins exact ties.
        assert_eq!(
            sample(&logits, &SampleParams::greedy(), &mut r1),
            1,
            "tie → first index"
        );
    }

    #[test]
    fn top_p_one_top_p_full_is_exact_and_deterministic() {
        let logits = [1.0f32, 2.0, 3.0];
        let p = SampleParams {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let draw = |seed: u64| {
            let mut r = Pcg32::new(seed);
            (0..50)
                .map(|_| sample(&logits, &p, &mut r))
                .collect::<Vec<_>>()
        };
        let a = draw(1234);
        let b = draw(1234);
        let c = draw(4321);
        assert_eq!(a, b, "same seed → identical stream");
        assert_ne!(a, c, "different seed → different stream");
        assert!(a.iter().all(|&t| t < 3));
        assert!(
            a.contains(&2) && a.iter().any(|&t| t < 2),
            "all outcomes reachable"
        );
    }

    #[test]
    fn distribution_matches_analytic_probs() {
        // probs ∝ [1, 0.5, 0.25] → [4/7, 2/7, 1/7]; no truncation.
        let logits = [0.0f32, -(2.0f32).ln(), -(4.0f32).ln()];
        let p = SampleParams {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let mut r = Pcg32::new(2024);
        let n = 60_000;
        let mut counts = [0usize; 3];
        for _ in 0..n {
            counts[sample(&logits, &p, &mut r) as usize] += 1;
        }
        let want = [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0];
        for i in 0..3 {
            let got = counts[i] as f64 / n as f64;
            assert!(
                (got - want[i]).abs() < 0.02,
                "token {i}: got {got:.4} want {:.4}",
                want[i]
            );
        }
    }

    #[test]
    fn top_p_truncation_keeps_smallest_prefix() {
        // probs [0.5, 0.25, 0.25]: cumulative 0.5 / 0.75 / 1.0.
        let logits = [0.0f32, -(2.0f32).ln(), -(2.0f32).ln()];
        let mut r = Pcg32::new(11);

        // top_p = 0.5 → the first candidate already reaches the mass → 1 token.
        let p1 = SampleParams {
            temp: 1.0,
            top_k: 0,
            top_p: 0.5,
        };
        for _ in 0..2000 {
            assert_eq!(sample(&logits, &p1, &mut r), 0, "top_p must cut to 1 token");
        }

        // top_p = 0.6 → 0.5 < 0.6 → keep two; cumulative 0.75 ≥ 0.6 → stop.
        let p2 = SampleParams {
            temp: 1.0,
            top_k: 0,
            top_p: 0.6,
        };
        let mut seen = [0usize; 3];
        for _ in 0..2000 {
            seen[sample(&logits, &p2, &mut r) as usize] += 1;
        }
        assert_eq!(seen[2], 0, "third token excluded when cum(2) ≥ top_p");
        assert!(seen[0] > 0 && seen[1] > 0);

        // top_p = 0.76 → 0.75 < 0.76 → the third candidate is included.
        let p3 = SampleParams {
            temp: 1.0,
            top_k: 0,
            top_p: 0.76,
        };
        let mut seen3 = [0usize; 3];
        for _ in 0..2000 {
            seen3[sample(&logits, &p3, &mut r) as usize] += 1;
        }
        assert!(seen3[2] > 0, "third token included when cum(2) < top_p");
        assert!(seen3[0] > 900 && seen3[1] > 400 && seen3[2] > 400);
    }

    #[test]
    fn top_k_truncates_to_highest_weight_tokens() {
        let logits = [5.0f32, 4.0, 3.0, 2.0];
        let p = SampleParams {
            temp: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        let mut r = Pcg32::new(5);
        for _ in 0..2000 {
            let t = sample(&logits, &p, &mut r);
            assert!(t <= 1, "top_k=2 leaked token {t}");
        }
    }

    #[test]
    fn temperature_sharpens_distribution() {
        let logits = [0.0f32, 1.0];
        let mut r = Pcg32::new(3);
        let count_second = |temp: f32, r: &mut Pcg32| {
            let p = SampleParams {
                temp,
                top_k: 0,
                top_p: 1.0,
            };
            (0..4000).filter(|_| sample(&logits, &p, r) == 1).count()
        }; // temp 0.1 → P(token 1) ≈ 1; temp 5 → P ≈ 0.55; temp 1 → P ≈ 0.731.
        let cold = count_second(0.1, &mut r);
        let hot = count_second(5.0, &mut r);
        let warm = count_second(1.0, &mut r);
        assert!(cold > 3950, "low temp must concentrate: {cold}/4000");
        assert!((1900..2500).contains(&hot), "temp 5 flattens: {hot}/4000");
        assert!(
            (2700..3100).contains(&warm),
            "temp 1 off analytic: {warm}/4000"
        );
    }

    #[test]
    fn non_finite_logits_are_handled() {
        // NaN and both infinities are unusable → the only finite logit wins.
        let logits = [f32::NAN, 1.0, f32::NEG_INFINITY, f32::INFINITY];
        let mut r = Pcg32::new(1);
        assert_eq!(sample(&logits, &SampleParams::greedy(), &mut r), 1);
        let p = SampleParams {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        for _ in 0..500 {
            assert_eq!(sample(&logits, &p, &mut r), 1);
        }
        // A single finite candidate among infinities is always chosen.
        let mostly_inf = [f32::INFINITY, -1.0, f32::NAN];
        assert_eq!(sample(&mostly_inf, &SampleParams::greedy(), &mut r), 1);
        assert_eq!(sample(&mostly_inf, &p, &mut r), 1);
        // All -inf → falls back to index 0 instead of panicking.
        let dead = [f32::NEG_INFINITY; 4];
        assert_eq!(sample(&dead, &SampleParams::greedy(), &mut r), 0);
        assert_eq!(sample(&dead, &p, &mut r), 0);
        let all_nan = [f32::NAN; 3];
        assert_eq!(sample(&all_nan, &p, &mut r), 0);
    }

    #[test]
    #[should_panic(expected = "empty logits")]
    fn empty_logits_panics() {
        let mut r = Pcg32::new(1);
        sample(&[], &SampleParams::default(), &mut r);
    }
}
