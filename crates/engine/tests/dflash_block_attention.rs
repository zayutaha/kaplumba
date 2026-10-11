//! spec/dflash-correctness R8: the DFlash drafter's block attention, checked
//! against a scalar host reference in all three mask configurations the
//! drafter uses, and cross-checked against the engine's *causal* attention
//! kernel where the two must agree.
//!
//! The point of the cross-check: `online_attention` is an independent
//! implementation of the causal-plus-window mask, so a kernel bug that happens
//! to reproduce a CPU bug is much less likely to survive both.

use half::f16;
use kaplumba_kernels::attention::{AttentionKernels, AttentionRequest, DflashBlockRequest};

const Q_HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const ROWS: usize = 16;

fn pattern(count: usize, seed: f32) -> Vec<u16> {
    (0..count)
        .map(|index| {
            let value = ((index as f32 * 0.37 + seed).sin() * 0.5)
                + ((index as f32 * 0.011 + seed * 2.0).cos() * 0.25);
            f16::from_f32(value).to_bits()
        })
        .collect()
}

/// Scalar reference. `bidirectional` mirrors the kernel flag, `window` is `0`
/// for "no window". Accumulates in fp32 and narrows once, like the kernel.
fn reference(
    queries: &[u16],
    ctx_keys: &[u16],
    ctx_values: &[u16],
    block_keys: &[u16],
    block_values: &[u16],
    positions: &[u32],
    n_ctx: usize,
    window: usize,
    bidirectional: bool,
) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let groups = Q_HEADS / KV_HEADS;
    let total = n_ctx + ROWS;
    let mut output = vec![0.0_f32; ROWS * Q_HEADS * HEAD_DIM];
    for row in 0..ROWS {
        let q_pos = positions[n_ctx + row];
        for q_head in 0..Q_HEADS {
            let key_head = q_head / groups;
            let mut scores = vec![f32::NEG_INFINITY; total];
            for key in 0..total {
                let k_pos = positions[key];
                let past_ok = bidirectional || k_pos <= q_pos;
                let gap = q_pos.wrapping_sub(k_pos);
                let window_ok = window == 0 || gap < window as u32;
                if !(past_ok && window_ok) {
                    continue;
                }
                let base = if key < n_ctx {
                    (key * KV_HEADS + key_head) * HEAD_DIM
                } else {
                    ((key - n_ctx) * KV_HEADS + key_head) * HEAD_DIM
                };
                let keys = if key < n_ctx { ctx_keys } else { block_keys };
                let q_base = (row * Q_HEADS + q_head) * HEAD_DIM;
                let dot: f32 = (0..HEAD_DIM)
                    .map(|d| {
                        f16::from_bits(queries[q_base + d]).to_f32()
                            * f16::from_bits(keys[base + d]).to_f32()
                    })
                    .sum();
                scores[key] = dot * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for score in scores.iter_mut() {
                if *score == f32::NEG_INFINITY {
                    *score = 0.0;
                } else {
                    *score = (*score - max).exp();
                    sum += *score;
                }
            }
            let o_base = (row * Q_HEADS + q_head) * HEAD_DIM;
            for d in 0..HEAD_DIM {
                let mut acc = 0.0_f32;
                for key in 0..total {
                    if scores[key] == 0.0 && !f32::is_finite(scores[key]) {
                        continue;
                    }
                    let base = if key < n_ctx {
                        (key * KV_HEADS + key_head) * HEAD_DIM
                    } else {
                        ((key - n_ctx) * KV_HEADS + key_head) * HEAD_DIM
                    };
                    let values = if key < n_ctx { ctx_values } else { block_values };
                    acc += scores[key] * f16::from_bits(values[base + d]).to_f32();
                }
                output[o_base + d] = acc / sum;
            }
        }
    }
    output
}

fn assert_close(actual: &[u16], expected: &[f32], tolerance: f32, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    let mut worst = (0.0_f32, 0_usize);
    for (index, value) in actual.iter().enumerate() {
        let value = f16::from_bits(*value).to_f32();
        let diff = (value - expected[index]).abs();
        if diff > worst.0 {
            worst = (diff, index);
        }
        assert!(
            diff <= tolerance,
            "{name}[{index}] expected {} actual {value} (diff {diff})",
            expected[index]
        );
    }
    eprintln!("{name}: worst diff {:.6} at {}", worst.0, worst.1);
}

#[test]
fn dflash_block_attention_matches_host_in_every_mask_mode() {
    let kernels = AttentionKernels::new().expect("Metal attention kernels should compile");
    let n_ctx = 40usize;
    let positions = (0..n_ctx + ROWS).map(|index| index as u32).collect::<Vec<_>>();

    let queries = pattern(ROWS * Q_HEADS * HEAD_DIM, 0.1);
    let ctx_keys = pattern(n_ctx * KV_HEADS * HEAD_DIM, 0.7);
    let ctx_values = pattern(n_ctx * KV_HEADS * HEAD_DIM, 1.3);
    let block_keys = pattern(ROWS * KV_HEADS * HEAD_DIM, 1.9);
    let block_values = pattern(ROWS * KV_HEADS * HEAD_DIM, 2.4);

    // The three mask modes the drafter's `layer_types` produce:
    //   sliding_attention -> causal + window
    //   full_attention     -> bidirectional
    let cases: [(&str, usize, bool); 3] = [
        ("full_attention (bidirectional)", 0, true),
        ("sliding_attention window 2048", 2048, false),
        ("sliding_attention window 4", 4, false),
    ];
    for (name, window, bidirectional) in cases {
        let expected = reference(
            &queries,
            &ctx_keys,
            &ctx_values,
            &block_keys,
            &block_values,
            &positions,
            n_ctx,
            window,
            bidirectional,
        );
        let actual = kernels
            .dflash_block_attention(DflashBlockRequest {
                queries: &queries,
                ctx_keys: &ctx_keys,
                ctx_values: &ctx_values,
                block_keys: &block_keys,
                block_values: &block_values,
                key_positions: &positions,
                rows: ROWS,
                n_ctx,
                q_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                window,
                bidirectional,
            })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_close(&actual, &expected, 3e-3, name);
    }
}

#[test]
fn dflash_block_attention_agrees_with_online_attention_when_both_are_causal() {
    // Independent cross-check. For a bidirectional=false window large enough to
    // cover the whole context, the drafter kernel and `online_attention` are
    // computing the same causal attention over the same keys, one as a fused
    // ctx+block array and one as the engine's own path. They must agree.
    let kernels = AttentionKernels::new().expect("Metal attention kernels should compile");
    let n_ctx = 24usize;
    let positions = (0..n_ctx + ROWS).map(|index| index as u32).collect::<Vec<_>>();

    let queries = pattern(ROWS * Q_HEADS * HEAD_DIM, 3.1);
    let ctx_keys = pattern(n_ctx * KV_HEADS * HEAD_DIM, 4.2);
    let ctx_values = pattern(n_ctx * KV_HEADS * HEAD_DIM, 5.5);
    let block_keys = pattern(ROWS * KV_HEADS * HEAD_DIM, 6.6);
    let block_values = pattern(ROWS * KV_HEADS * HEAD_DIM, 7.7);

    let mut keys = ctx_keys.clone();
    keys.extend_from_slice(&block_keys);
    let mut values = ctx_values.clone();
    values.extend_from_slice(&block_values);

    let from_block = kernels
        .dflash_block_attention(DflashBlockRequest {
            queries: &queries,
            ctx_keys: &ctx_keys,
            ctx_values: &ctx_values,
            block_keys: &block_keys,
            block_values: &block_values,
            key_positions: &positions,
            rows: ROWS,
            n_ctx,
            q_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            window: 0,
            bidirectional: false,
        })
        .expect("block path");

    let query_positions = positions[n_ctx..].to_vec();
    let from_engine = kernels
        .online_attention(AttentionRequest {
            queries: &queries,
            keys: &keys,
            values: &values,
            query_positions: &query_positions,
            key_positions: &positions,
            q_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            window: 0,
        })
        .expect("online path");

    // The two kernels differ in reduction order (one is simdgroup-parallel
    // per head, the other a scalar per-thread loop), so this is a numeric
    // agreement check, not a bit-exact one. fp16 with a 128-dim dot product
    // and a long softmax accumulator: 5e-3 is well inside that.
    let mut worst = 0.0_f32;
    for (index, (a, b)) in from_block.iter().zip(from_engine.iter()).enumerate() {
        let a = f16::from_bits(*a).to_f32();
        let b = f16::from_bits(*b).to_f32();
        worst = worst.max((a - b).abs());
    }
    eprintln!("dflash vs online_attention worst diff {worst:.6}");
    assert!(
        worst < 5e-3,
        "causal block attention disagrees with online_attention by {worst}"
    );
}

#[test]
fn dflash_block_attention_window_is_narrower_than_causal() {
    // Sabotage evidence for the window flag: with window == 1 only the query's
    // own position survives the mask, so every row of the block must return
    // that row's own value. If the window flag were ignored (window 0), row 0
    // would return its own value too and the test would pass vacuously -- so
    // assert a *different* row changes, not that they all change.
    let kernels = AttentionKernels::new().expect("Metal attention kernels should compile");
    let n_ctx = 8usize;
    let positions = (0..n_ctx + ROWS).map(|index| index as u32).collect::<Vec<_>>();

    let queries = pattern(ROWS * Q_HEADS * HEAD_DIM, 8.1);
    let ctx_keys = pattern(n_ctx * KV_HEADS * HEAD_DIM, 9.2);
    let ctx_values = pattern(n_ctx * KV_HEADS * HEAD_DIM, 10.3);
    let block_keys = pattern(ROWS * KV_HEADS * HEAD_DIM, 11.4);
    let block_values = pattern(ROWS * KV_HEADS * HEAD_DIM, 12.5);

    let run = |window: usize, bidirectional: bool| {
        kernels
            .dflash_block_attention(DflashBlockRequest {
                queries: &queries,
                ctx_keys: &ctx_keys,
                ctx_values: &ctx_values,
                block_keys: &block_keys,
                block_values: &block_values,
                key_positions: &positions,
                rows: ROWS,
                n_ctx,
                q_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                window,
                bidirectional,
            })
            .expect("block path")
    };

    let wide = run(0, false);
    let narrow = run(1, false);
    let changed = wide
        .iter()
        .zip(narrow.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        changed > wide.len() / 4,
        "window 1 barely differs from no window ({changed} of {} elements)",
        wide.len()
    );

    // window 1 == only the key at the query's own position, i.e. the block
    // row's own value vector, passed straight through.
    let mut expected = vec![0.0_f32; ROWS * Q_HEADS * HEAD_DIM];
    for row in 0..ROWS {
        for q_head in 0..Q_HEADS {
            let key_head = q_head / (Q_HEADS / KV_HEADS);
            for d in 0..HEAD_DIM {
                let base = (row * KV_HEADS + key_head) * HEAD_DIM;
                expected[(row * Q_HEADS + q_head) * HEAD_DIM + d] =
                    f16::from_bits(block_values[base + d]).to_f32();
            }
        }
    }
    assert_close(&narrow, &expected, 0.0, "window 1 returns the row's own value");
}
