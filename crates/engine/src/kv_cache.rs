//! Hybrid KV cache (T5.1): sliding layers keep a rotated window, full layers keep
//! everything.
//!
//! This is what makes decode *GPU-eligible* at all. The naive engine recomputed
//! the whole prefix every step, so every linear call had `rows > 1`, and the GPU
//! kernels only handle a single row (see the DE29 note in
//! `spec/rust-metal-engine/implementation_notes.md`) — that silently sent the
//! entire forward pass down the CPU path. With a cache, prefill pushes the
//! prompt once and each decode step is `rows == 1`, which is exactly the shape
//! the kernels and the attention pass are built for.
//!
//! Memory: rows are stored per layer at `n_kv * head_dim` f16 each. For Gemma 4
//! 31B that is 8 KB/row on sliding layers (16 KV heads × 256) and 4 KB/row on
//! full layers (4 × 512). A sliding layer only ever keeps `window` rows
//! (1024 → 8 MB per layer), so the whole cache is ~450 MB at 1024 tokens and
//! grows only on the ten full layers — the same shape as the spec's
//! "~1 GB SWA/KV" budget.
//!
//! The sliding window is implemented as a **contiguous rotated window**: when the
//! buffer is full the oldest row is dropped and the rest shifted by one row
//! (8 KB memmove per layer per token). That keeps key row 0 at a well-defined
//! absolute position (`kv_pos0`) for the attention kernels, which is all they
//! need — a true ring with modular indexing would be a pure kernel-complexity
//! trade for the same bandwidth, so it is deliberately not done yet.

use half::f16;

use crate::gemma4::{GemmaCfg, GemmaError, LayerKind};

/// Keys (and values) for one decoder layer.
pub struct LayerKv {
    k: Vec<f16>,
    v: Vec<f16>,
    row: usize,
    cap: usize,
    /// Rows currently stored (≤ `cap`).
    len: usize,
    /// Absolute position of stored row 0.
    start: usize,
}

impl LayerKv {
    fn new(row: usize, cap: usize) -> Self {
        Self {
            k: vec![f16::ZERO; row * cap],
            v: vec![f16::ZERO; row * cap],
            row,
            cap,
            len: 0,
            start: 0,
        }
    }

    /// Absolute position of stored row 0 (keys may not start at position 0 once
    /// a sliding window has rotated).
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of rows that must stay resident (window for sliding layers).
    fn limit(window: usize, cap: usize) -> usize {
        if window == 0 { cap } else { window.min(cap) }
    }

    fn keys(&self) -> &[f16] {
        &self.k[..self.len * self.row]
    }

    fn values(&self) -> &[f16] {
        &self.v[..self.len * self.row]
    }

    /// Append one row, dropping the oldest when a window is full.
    ///
    /// The caller has already checked the row width.
    fn push(&mut self, k_row: &[f16], v_row: &[f16], window: usize) {
        let limit = Self::limit(window, self.cap);
        if self.len == limit {
            let keep = (limit - 1) * self.row;
            self.k.copy_within(self.row..self.row + keep, 0);
            self.v.copy_within(self.row..self.row + keep, 0);
            self.start += 1;
            self.len = limit - 1;
        }
        let at = self.len * self.row;
        self.k[at..at + self.row].copy_from_slice(k_row);
        self.v[at..at + self.row].copy_from_slice(v_row);
        self.len += 1;
    }
}

/// One `LayerKv` per decoder layer.
pub struct KvCache {
    layers: Vec<LayerKv>,
    /// Tokens currently in the cache (identical for every layer).
    len: usize,
}

impl KvCache {
    /// Allocate for a context of `cap` tokens (capacity is per token, not per
    /// layer: sliding layers self-limit to the sliding window).
    pub fn new(cfg: &GemmaCfg, cap: usize) -> Self {
        let cap = cap.max(1);
        let layers = cfg
            .layers
            .iter()
            .map(|lc| {
                let window = match lc.kind {
                    LayerKind::Sliding => cfg.sliding_window,
                    LayerKind::Full => 0,
                };
                LayerKv::new(lc.n_kv * lc.head_dim, LayerKv::limit(window, cap).max(1))
            })
            .collect();
        Self { layers, len: 0 }
    }

    /// Tokens cached (the position of the next token to append).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total resident bytes (the number R5.3/the budget cares about).
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| (l.k.len() + l.v.len()) * std::mem::size_of::<f16>())
            .sum()
    }

    pub fn layer(&self, i: usize) -> &LayerKv {
        &self.layers[i]
    }

    /// Drop everything (positions restart at 0).
    pub fn reset(&mut self) {
        for l in self.layers.iter_mut() {
            l.len = 0;
            l.start = 0;
        }
        self.len = 0;
    }

    fn push(&mut self, i: usize, k_row: &[f16], v_row: &[f16], window: usize) {
        self.layers[i].push(k_row, v_row, window);
    }

    fn finish_token(&mut self) {
        self.len += 1;
    }

    fn check(&self, cfg: &GemmaCfg) -> Result<(), GemmaError> {
        if self.layers.len() != cfg.layers.len() {
            return Err(GemmaError::Config(format!(
                "kv cache has {} layers, model has {}",
                self.layers.len(),
                cfg.layers.len()
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers used by the model's cached path
// ---------------------------------------------------------------------------

impl KvCache {
    /// Append one layer's K/V row and return `(kv_pos0, kv_rows)` for attention.
    pub(crate) fn append_layer(
        &mut self,
        cfg: &GemmaCfg,
        i: usize,
        k_row: &[f16],
        v_row: &[f16],
    ) -> Result<(usize, usize), GemmaError> {
        self.check(cfg)?;
        let window = match cfg.layers[i].kind {
            LayerKind::Sliding => cfg.sliding_window,
            LayerKind::Full => 0,
        };
        let want = self.layers[i].row;
        if k_row.len() != want || v_row.len() != want {
            return Err(GemmaError::Config(format!(
                "layer {i}: kv row width {}/{want}",
                k_row.len()
            )));
        }
        if self.layers[i].cap == 0 {
            return Err(GemmaError::Config(format!("layer {i}: zero kv capacity")));
        }
        if window == 0 && self.layers[i].len == self.layers[i].cap {
            // Full layers must never silently drop context: a window of 0 means
            // "keep everything", so running out of room is a caller error.
            return Err(GemmaError::Config(format!(
                "layer {i}: kv cache full at {} rows (allocate a larger context)",
                self.layers[i].cap
            )));
        }
        self.push(i, k_row, v_row, window);
        let l = &self.layers[i];
        Ok((l.start, l.len))
    }

    pub(crate) fn layer_keys(&self, i: usize) -> &[f16] {
        self.layers[i].keys()
    }

    pub(crate) fn layer_values(&self, i: usize) -> &[f16] {
        self.layers[i].values()
    }

    pub(crate) fn end_token(&mut self) {
        self.finish_token();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{SynthSource, synth_model};

    fn cfg() -> (SynthSource, GemmaCfg) {
        synth_model(Some(30.0))
    }

    /// Row values encode their absolute position so a mis-rotation is visible.
    fn row(n: usize, tag: f32) -> Vec<f16> {
        (0..n)
            .map(|j| f16::from_f32(tag * 1000.0 + j as f32))
            .collect()
    }

    #[test]
    fn full_layers_grow_and_sliding_layers_rotate() {
        let (_src, cfg) = cfg();
        // window in the synthetic model is 4.
        assert_eq!(cfg.sliding_window, 4);
        let mut kv = KvCache::new(&cfg, 64);

        for pos in 0..6 {
            for i in 0..cfg.layers.len() {
                let lc = cfg.layers[i];
                let n = lc.n_kv * lc.head_dim;
                let (start, rows) = kv
                    .append_layer(&cfg, i, &row(n, pos as f32), &row(n, 1.0))
                    .expect("append");
                match lc.kind {
                    LayerKind::Sliding => {
                        assert_eq!(
                            rows,
                            (pos + 1).min(4),
                            "sliding layer {i} rows at pos {pos}"
                        );
                        assert_eq!(start, (pos + 1).saturating_sub(4));
                    }
                    LayerKind::Full => {
                        assert_eq!(rows, pos + 1, "full layer {i} keeps everything");
                        assert_eq!(start, 0);
                    }
                }
            }
            kv.end_token();
        }
        assert_eq!(kv.len(), 6);
        // The newest sliding row is position 5; the oldest retained is 2.
        let l0 = kv.layer(0);
        assert_eq!(l0.start(), 2);
        assert_eq!(l0.len(), 4);
        let first = l0.keys()[0].to_f32();
        assert!((first - 2000.0).abs() < 1.0, "oldest retained row is pos 2");
        // The newest retained row is position 5 (values are unique per row).
        assert!((l0.keys()[3 * l0.keys().len() / 4].to_f32() - 5000.0).abs() < 4.0);
    }

    #[test]
    fn reset_clears_positions() {
        let (_src, cfg) = cfg();
        let mut kv = KvCache::new(&cfg, 8);
        let n = cfg.layers[0].n_kv * cfg.layers[0].head_dim;
        kv.append_layer(&cfg, 0, &row(n, 1.0), &row(n, 1.0))
            .unwrap();
        kv.end_token();
        assert_eq!(kv.len(), 1);
        kv.reset();
        assert_eq!(kv.len(), 0);
        assert!(kv.is_empty());
        assert_eq!(kv.layer(0).len(), 0);
        assert_eq!(kv.layer(0).start(), 0);
    }

    #[test]
    fn bytes_are_bounded_by_window_plus_context() {
        let (_src, cfg) = cfg();
        let kv = KvCache::new(&cfg, 32);
        // Sliding layers (window 4) reserve 4 rows, full layers reserve 32.
        let mut expect = 0usize;
        for lc in &cfg.layers {
            let row_bytes = lc.n_kv * lc.head_dim * 2;
            let rows = match lc.kind {
                LayerKind::Sliding => 4,
                LayerKind::Full => 32,
            };
            // K and V.
            expect += 2 * row_bytes * rows;
        }
        assert_eq!(kv.bytes(), expect);
    }

    #[test]
    fn push_rejects_wrong_row_width() {
        let (_src, cfg) = cfg();
        let mut kv = KvCache::new(&cfg, 4);
        let n = cfg.layers[0].n_kv * cfg.layers[0].head_dim;
        let bad = row(n + 1, 0.0);
        let err = kv
            .append_layer(&cfg, 0, &bad, &row(n, 0.0))
            .expect_err("row width mismatch must be rejected, not corrupt");
        assert!(matches!(err, GemmaError::Config(_)), "got {err}");
    }
}
