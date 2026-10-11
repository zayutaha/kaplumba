//! spec/dflash-correctness T9: the batched-verify head must be equivalent to
//! `n` sequential single-row forwards.
//!
//! plan.md risk #7, and it is not a formality. `forward_verify` runs
//! `encode_rms_norm(rows = n)` + `encode_gemm(M = n)` where the decode path
//! runs `encode_rms_norm(rows = 1)` + `encode_gemv(M = 1)`. Those are different
//! kernels with different accumulation orders, so a divergence would corrupt
//! every token the round "verifies" while still reporting a plausible acceptance
//! rate. This is the test that makes T9 admissible.
//!
//! Ignored by default: it loads the 16 GB target. `KAPLUMBA_TEST_MODEL` selects
//! which checkpoint.

use std::path::Path;

use kaplumba_engine::model::TextModel;

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required")
}

fn argmax_row(matrix: &[u16], row: usize, vocab: usize) -> u32 {
    let base = row * vocab;
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for column in 0..vocab {
        let value = half::f16::from_bits(matrix[base + column]).to_f32();
        if value > best_value {
            best_value = value;
            best = column;
        }
    }
    best as u32
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target"]
fn batched_verify_head_matches_sequential_single_row() {
    let path = model_path();
    let dir = Path::new(&path);
    let mut model = TextModel::load_from_dir(dir).expect("target model");
    let vocab = model.vocab_size();

    // A short real token run, long enough that the row count is not degenerate
    // and short enough to keep the ignored test quick.
    let tokens: Vec<u32> = (0..48u32).map(|index| 1 + (index * 7919) % 4096).collect();

    // rows = 1 exercises the single-row head (`encode_gemv`) through the same
    // entry point, so a divergence there is the plumbing and a divergence only
    // at rows > 1 is `encode_gemm`'s MMA tile at small M.
    for rows in [1usize, 2, 3, 8, 16, 32] {
        // The window is prefilled from a reset cache, so its positions are
        // `0..rows` -- not a slice of a longer conversation's positions.
        let window = &tokens[..rows];
        let positions: Vec<u32> = (0..rows as u32).collect();

        // --- one batched pass ------------------------------------------------
        // The whole window as one chunk, like a prefill. Causal attention means
        // row `i` sees tokens `0..=i`, which is exactly what the sequential path
        // below sees, so the head is the only variable.
        model.reset_gpu_caches();
        let batched = model
            .forward_verify(window, &positions)
            .expect("batched verify");
        assert_eq!(
            batched.len(),
            rows * vocab,
            "the batched head returned {} elements for {rows} rows of {vocab}",
            batched.len()
        );

        // --- `rows` sequential single-row passes ----------------------------
        // One forward per token, with the cache growing the way a decode loop
        // grows it. `positions` must start at the (reset) cache cursor, i.e. 0.
        model.reset_gpu_caches();
        let sequential_positions: Vec<u32> = (0..rows as u32).collect();
        let mut worst = 0.0_f32;
        let mut worst_at = (0usize, 0usize);
        for (index, (token, position)) in
            window.iter().zip(sequential_positions.iter()).enumerate()
        {
            let mut row_worst = 0.0_f32;
            let row = model
                .forward_buffer(&[*token], &[*position])
                .expect("single row");
            let base = index * vocab;
            for column in 0..vocab {
                let a = half::f16::from_bits(batched[base + column]).to_f32();
                let b = half::f16::from_bits(row[column]).to_f32();
                let diff = (a - b).abs();
                row_worst = row_worst.max(diff);
                if diff > worst {
                    worst = diff;
                    worst_at = (index, column);
                }
            }
            eprintln!("rows {rows} row {index}: worst {row_worst:.6}");
            if rows == 1 {
                let column = worst_at.1;
                let a = half::f16::from_bits(batched[column]).to_f32();
                let b = half::f16::from_bits(row[column]).to_f32();
                eprintln!(
                    "  col {column}: batched {a:.4} sequential {b:.4} | argmax batched {} sequential {}",
                    argmax_row(&batched, 0, vocab),
                    argmax_row(&row, 0, vocab)
                );
            }
        }
        let scale = half::f16::from_bits(batched[worst_at.1])
            .to_f32()
            .abs()
            .max(1.0);
        eprintln!(
            "rows {rows}: worst |batched - sequential| = {worst:.6} at row {} col {} \
             (relative {:.4}%)",
            worst_at.0,
            worst_at.1,
            100.0 * worst / scale
        );
        // fp16 softcapped logits in the -30..30 band, computed by two different
        // kernels (`encode_gemv` vs `encode_gemm`), so two different accumulation
        // orders. Near |30| an fp16 quantum is 0.03125, so 0.05 is under two
        // quanta: tight enough that a real divergence (measured at 32 with two
        // models loaded concurrently on this 24 GiB box) is still an order of
        // magnitude outside it, loose enough that rounding is not.
        assert!(
            worst < 0.05,
            "the batched head diverges from the sequential path by {worst} at row {} col {}",
            worst_at.0,
            worst_at.1
        );
    }
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target"]
fn batched_verify_head_carries_signal_on_every_row() {
    // A second, independent guard: `head_rows_matrix` returning the same row 16
    // times (a stride bug in the read-back) would make every verify decide from
    // one row's logits and pass any comparison that only checks the diagonal.
    let path = model_path();
    let dir = Path::new(&path);
    let mut model = TextModel::load_from_dir(dir).expect("target model");
    let vocab = model.vocab_size();

    let tokens: Vec<u32> = (0..40u32).map(|index| 1 + (index * 104_729) % 8192).collect();
    model.reset_gpu_caches();
    let positions: Vec<u32> = (0..16u32).collect();
    let batched = model
        .forward_verify(&tokens[..16], &positions)
        .expect("batched verify");
    assert_eq!(batched.len(), 16 * vocab);

    let mut differing = 0usize;
    let mut compared = 0usize;
    for row in 0..16usize {
        let next = (row + 1) % 16;
        for column in 0..vocab {
            compared += 1;
            let a = half::f16::from_bits(batched[row * vocab + column]).to_f32();
            let b = half::f16::from_bits(batched[next * vocab + column]).to_f32();
            if (a - b).abs() > 1e-6 {
                differing += 1;
            }
        }
    }
    eprintln!("adjacent rows differ on {differing}/{compared} vocabulary entries");
    // Adjacent positions genuinely have different distributions. If the rows
    // were a stride bug they would be identical everywhere.
    assert!(
        differing > compared / 100,
        "adjacent verify rows are nearly identical ({differing}/{compared}); the \
         read-back stride is probably wrong"
    );
}
