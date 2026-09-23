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
use kaplumba_kernels::DeviceSpan;
use metal::{Buffer, Device, MTLResourceOptions};

use crate::gemma4::{GemmaCfg, GemmaError, LayerKind};

/// Backing store for one key/value matrix.
///
/// `Host` is the CPU path (and every unit test). `Device` is a `StorageModeShared`
/// Metal buffer: on Apple silicon that is the *same* memory the GPU reads, so the
/// host can append or rotate a row with a plain `copy_within` and the attention
/// kernel then reads the cache in place — no staging copy, and therefore no
/// command-buffer round trip per layer (which is what T4.2 needs).
enum Store {
    Host(Vec<f16>),
    Device(Buffer),
}

impl Store {
    fn new_host(n: usize) -> Self {
        Store::Host(vec![f16::ZERO; n])
    }

    fn new_device(device: &Device, n: usize) -> Self {
        Store::Device(
            device.new_buffer((n * 2).max(2) as u64, MTLResourceOptions::StorageModeShared),
        )
    }

    fn slice(&self, n: usize) -> &[f16] {
        match self {
            Store::Host(v) => &v[..n],
            Store::Device(b) => unsafe {
                std::slice::from_raw_parts(b.contents() as *const f16, n)
            },
        }
    }

    fn slice_mut(&mut self, n: usize) -> &mut [f16] {
        match self {
            Store::Host(v) => &mut v[..n],
            Store::Device(b) => unsafe {
                std::slice::from_raw_parts_mut(b.contents() as *mut f16, n)
            },
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Store::Host(v) => v.len(),
            Store::Device(b) => b.length() as usize / 2,
        }
    }

    fn buffer(&self) -> Option<&Buffer> {
        match self {
            Store::Host(_) => None,
            Store::Device(b) => Some(b),
        }
    }
}

/// Keys (and values) for one decoder layer.
pub struct LayerKv {
    k: Store,
    v: Store,
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
            k: Store::new_host(row * cap),
            v: Store::new_host(row * cap),
            row,
            cap,
            len: 0,
            start: 0,
        }
    }

    fn new_device(device: &Device, row: usize, cap: usize) -> Self {
        Self {
            k: Store::new_device(device, row * cap),
            v: Store::new_device(device, row * cap),
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
        self.k.slice(self.len * self.row)
    }

    fn values(&self) -> &[f16] {
        self.v.slice(self.len * self.row)
    }

    /// Device span covering the whole key matrix (starts at stored row 0).
    pub fn keys_span(&self) -> Option<DeviceSpan<'_>> {
        self.k.buffer().map(|buffer| DeviceSpan {
            buffer,
            byte_offset: 0,
        })
    }

    pub fn values_span(&self) -> Option<DeviceSpan<'_>> {
        self.v.buffer().map(|buffer| DeviceSpan {
            buffer,
            byte_offset: 0,
        })
    }

    /// Append one row, dropping the oldest when a window is full.
    ///
    /// The caller has already checked the row width.
    fn push(&mut self, k_row: &[f16], v_row: &[f16], window: usize) {
        let limit = Self::limit(window, self.cap);
        if self.len == limit {
            let keep = (limit - 1) * self.row;
            let (ks, vs) = (self.k.slice_mut(limit * self.row), limit * self.row);
            ks.copy_within(self.row..self.row + keep, 0);
            self.v
                .slice_mut(vs)
                .copy_within(self.row..self.row + keep, 0);
            self.start += 1;
            self.len = limit - 1;
        }
        let row = self.row;
        let at = self.len * row;
        self.k.slice_mut(at + row)[at..at + row].copy_from_slice(k_row);
        self.v.slice_mut(at + row)[at..at + row].copy_from_slice(v_row);
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
        Self::build(cfg, cap, None)
    }

    /// Same, but backed by `StorageModeShared` Metal buffers so the GPU path can
    /// read K/V in place (and kernels can write rows straight into the cache).
    pub fn new_device(cfg: &GemmaCfg, cap: usize, device: &Device) -> Self {
        Self::build(cfg, cap, Some(device))
    }

    /// True when the cache is GPU-readable (device-backed).
    pub fn is_device_backed(&self) -> bool {
        self.layers.first().is_some_and(|l| l.k.buffer().is_some())
    }

    fn build(cfg: &GemmaCfg, cap: usize, device: Option<&Device>) -> Self {
        let cap = cap.max(1);
        let layers = cfg
            .layers
            .iter()
            .map(|lc| {
                let window = match lc.kind {
                    LayerKind::Sliding => cfg.sliding_window,
                    LayerKind::Full => 0,
                };
                let row = lc.n_kv * lc.head_dim;
                let rows = LayerKv::limit(window, cap).max(1);
                match device {
                    Some(d) => LayerKv::new_device(d, row, rows),
                    None => LayerKv::new(row, rows),
                }
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
            .map(|l| (l.k.capacity() + l.v.capacity()) * std::mem::size_of::<f16>())
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

    /// Truncate / rewind the cache to `target_len` tokens (for speculative decoding rollback).
    /// `target_len` must be <= `self.len`.
    pub fn truncate(&mut self, target_len: usize) -> Result<(), GemmaError> {
        if target_len > self.len {
            return Err(GemmaError::Config(format!(
                "cannot truncate kv cache from {} to larger length {target_len}",
                self.len
            )));
        }
        if target_len == self.len {
            return Ok(());
        }
        for l in self.layers.iter_mut() {
            if target_len <= l.start {
                l.len = 0;
                l.start = target_len;
            } else {
                let valid_rows = target_len - l.start;
                if valid_rows < l.len {
                    l.len = valid_rows;
                }
            }
        }
        self.len = target_len;
        Ok(())
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

    /// Advance the shared token counter by `n` (batched prefill, T3.2).
    pub(crate) fn end_tokens(&mut self, n: usize) {
        self.len += n;
    }

    /// Reserve `n` rows at once (T3.2 batched prefill), rotating a full
    /// sliding window down by the overflow in one memmove, and return
    /// `(kv_pos0, kv_rows)` — the layer's key state **after** the whole batch
    /// is accounted for. The new rows are the last `n` rows
    /// (`kv_rows - n .. kv_rows`), which the K/V write passes then fill.
    ///
    /// `n` must be ≤ the sliding window limit: a batch bigger than the window
    /// would need multiple rotations with undefined intermediate states.
    pub fn reserve_rows(
        &mut self,
        cfg: &GemmaCfg,
        i: usize,
        pos0: usize,
        n: usize,
    ) -> Result<(usize, usize), GemmaError> {
        self.check(cfg)?;
        if n == 0 {
            return Err(GemmaError::Config("reserve_rows(n=0)".into()));
        }
        let window = match cfg.layers[i].kind {
            LayerKind::Sliding => cfg.sliding_window,
            LayerKind::Full => 0,
        };
        let l = &mut self.layers[i];
        if window == 0 {
            if l.len + n > l.cap {
                return Err(GemmaError::Config(format!(
                    "layer {i}: kv cache full at {} rows (allocate a larger context)",
                    l.cap
                )));
            }
            l.len += n;
        } else {
            let limit = LayerKv::limit(window, l.cap);
            if n > limit {
                return Err(GemmaError::Config(format!(
                    "layer {i}: batch of {n} rows exceeds the sliding window {limit}"
                )));
            }
            if l.len + n > limit {
                let d = l.len + n - limit;
                let row = l.row;
                let keep = (l.len - d) * row;
                l.k.slice_mut(l.len * row)
                    .copy_within(d * row..d * row + keep, 0);
                l.v.slice_mut(l.len * row)
                    .copy_within(d * row..d * row + keep, 0);
                l.start += d;
                l.len -= d;
            }
            l.len += n;
        }
        let (start, rows) = (l.start, l.len);
        if start + rows != pos0 + n {
            return Err(GemmaError::Config(format!(
                "layer {i}: reserve_rows(pos0={pos0}, n={n}) but window covers {}..{}",
                start,
                start + rows - 1
            )));
        }
        Ok((start, rows))
    }

    /// Reserve the row for `pos` **without writing it**, for the GPU path where
    /// the row contents are produced by kernels (RoPE straight into the cache for
    /// k, a device copy for v).
    ///
    /// Rotation is host-side on shared memory, which is safe because the previous
    /// token's command buffer has already completed (one sync per token).
    pub fn reserve_row(
        &mut self,
        cfg: &GemmaCfg,
        i: usize,
        pos: usize,
    ) -> Result<(usize, usize), GemmaError> {
        self.check(cfg)?;
        let window = match cfg.layers[i].kind {
            LayerKind::Sliding => cfg.sliding_window,
            LayerKind::Full => 0,
        };
        let l = &mut self.layers[i];
        if window == 0 && l.len == l.cap {
            return Err(GemmaError::Config(format!(
                "layer {i}: kv cache full at {} rows (allocate a larger context)",
                l.cap
            )));
        }
        if l.len == LayerKv::limit(window, l.cap) {
            let row = l.row;
            let limit = LayerKv::limit(window, l.cap);
            let keep = (limit - 1) * row;
            l.k.slice_mut(limit * row).copy_within(row..row + keep, 0);
            l.v.slice_mut(limit * row).copy_within(row..row + keep, 0);
            l.start += 1;
            l.len = limit - 1;
        }
        l.len += 1;
        let (start, rows) = (l.start, l.len);
        if start + rows - 1 != pos {
            return Err(GemmaError::Config(format!(
                "layer {i}: reserve_row for pos {pos} but window covers {}..{}",
                start,
                start + rows - 1
            )));
        }
        Ok((start, rows))
    }

    /// Device span for one stored row (K when `is_value == false`).
    pub fn row_span(&self, i: usize, is_value: bool, row: usize) -> Option<DeviceSpan<'_>> {
        let l = &self.layers[i];
        if row >= l.len {
            return None;
        }
        let store = if is_value { &l.v } else { &l.k };
        let buffer = store.buffer()?;
        Some(DeviceSpan {
            buffer,
            byte_offset: row * l.row * 2,
        })
    }

    /// Device span over the whole key matrix of layer `i`.
    pub fn layer_keys_span(&self, i: usize) -> Option<DeviceSpan<'_>> {
        self.layers[i].keys_span()
    }

    pub fn layer_values_span(&self, i: usize) -> Option<DeviceSpan<'_>> {
        self.layers[i].values_span()
    }

    /// Largest per-layer row capacity in this cache.
    pub fn capacity(&self) -> usize {
        self.layers.iter().map(|l| l.cap).max().unwrap_or(0)
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

    /// Batched reserve: sliding layers rotate by the whole overflow at once and
    /// report the same final `(start, rows)` a row-at-a-time append would; full
    /// layers grow by `n`. An oversized batch (n > window) is rejected.
    #[test]
    fn reserve_rows_batches_match_row_at_a_time() {
        let (_src, cfg) = cfg();
        assert_eq!(cfg.sliding_window, 4);

        // `one`: 7 rows one at a time, then +3 (overflow rotates by 3).
        let mut one = KvCache::new(&cfg, 64);
        for pos in 0..7 {
            for i in 0..cfg.layers.len() {
                one.reserve_rows(&cfg, i, pos, 1).expect("row reserve");
            }
            one.end_tokens(1);
        }
        for i in 0..cfg.layers.len() {
            one.reserve_rows(&cfg, i, 7, 3).expect("one +3");
        }
        one.end_tokens(3);
        assert_eq!(one.len(), 10);

        // `batch`: the same 10 rows in chunks of 4, 3, 3 (4 fits the window,
        // 4+3 overflows it and forces a multi-row rotation).
        let mut batch = KvCache::new(&cfg, 64);
        let mut bpos = 0usize;
        for c in [4usize, 3, 3] {
            for i in 0..cfg.layers.len() {
                batch.reserve_rows(&cfg, i, bpos, c).expect("batch reserve");
            }
            batch.end_tokens(c);
            bpos += c;
        }
        assert_eq!(bpos, 10);
        assert_eq!(batch.len(), 10);

        for i in 0..cfg.layers.len() {
            let lc = &cfg.layers[i];
            let ol = one.layer(i);
            let bl = batch.layer(i);
            assert_eq!(
                (bl.start(), bl.len()),
                (ol.start(), ol.len()),
                "layer {i} batch vs row-at-a-time diverged"
            );
            match lc.kind {
                LayerKind::Sliding => {
                    assert_eq!(bl.len(), 4, "sliding window stays full");
                    assert_eq!(bl.start(), 6, "positions 6..=9 retained");
                }
                LayerKind::Full => {
                    assert_eq!(bl.len(), 10, "full layer keeps all rows");
                    assert_eq!(bl.start(), 0);
                }
            }
        }

        // n > window must be rejected, not silently wrapped.
        let mut kv = KvCache::new(&cfg, 64);
        let err = kv
            .reserve_rows(&cfg, 0, 0, 5)
            .expect_err("batch larger than window");
        assert!(matches!(err, GemmaError::Config(_)), "got {err}");
    }

    #[test]
    fn truncate_rolls_back_speculative_tokens() {
        let (_src, cfg) = cfg();
        let mut kv = KvCache::new(&cfg, 64);

        // Append 10 tokens
        for pos in 0..10 {
            for i in 0..cfg.layers.len() {
                kv.reserve_rows(&cfg, i, pos, 1).expect("reserve");
            }
            kv.end_tokens(1);
        }
        assert_eq!(kv.len(), 10);

        // Truncate back to 7
        kv.truncate(7).expect("truncate to 7");
        assert_eq!(kv.len(), 7);

        // Verify layer states
        for i in 0..cfg.layers.len() {
            let l = kv.layer(i);
            match cfg.layers[i].kind {
                LayerKind::Full => {
                    assert_eq!(l.len(), 7);
                    assert_eq!(l.start(), 0);
                }
                LayerKind::Sliding => {
                    // Window is 4. Before truncate (pos 10), window was 6..=9 (start=6, len=4).
                    // After truncate to 7 (pos 0..=6), the only retained token in 6..=9 is pos 6.
                    // So start=6, len=1 (pos 6).
                    assert_eq!(l.len(), 1);
                    assert_eq!(l.start(), 6);
                }
            }
        }

        // Now append token at pos 7 and verify reserve_rows succeeds
        for i in 0..cfg.layers.len() {
            let (start, rows) = kv.reserve_rows(&cfg, i, 7, 1).expect("reserve at pos 7");
            match cfg.layers[i].kind {
                LayerKind::Full => {
                    assert_eq!(start, 0);
                    assert_eq!(rows, 8);
                }
                LayerKind::Sliding => {
                    // At pos 7, window covers pos 6 and pos 7 (start=6, rows=2)
                    assert_eq!(start, 6);
                    assert_eq!(rows, 2);
                }
            }
        }
        kv.end_tokens(1);
        assert_eq!(kv.len(), 8);

        // Truncate to larger than current len must fail
        assert!(kv.truncate(20).is_err());

        // Truncate to current len is no-op
        assert!(kv.truncate(8).is_ok());
        assert_eq!(kv.len(), 8);

        // Truncate to 0
        kv.truncate(0).expect("truncate to 0");
        assert_eq!(kv.len(), 0);
        for i in 0..cfg.layers.len() {
            assert_eq!(kv.layer(i).len(), 0);
        }
    }
}
