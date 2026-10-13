//! V0b: optimistic DFlash verifier ceiling using a correct target continuation.
//!
//! Ignored by default because this loads the target model and runs thousands
//! of target decode/verify steps. The timed path is the full target verify
//! (body + all-row head), with zero drafting cost. `KAPLUMBA_GPU_PROFILE=1`
//! adds the disjoint GPU body/head kernel attribution.

use std::path::Path;
use std::time::Instant;

use kaplumba_engine::model::TextModel;

const CONTEXTS: [usize; 4] = [32, 1024, 2048, 8192];
const PROPOSAL_ROWS: [usize; 4] = [2, 4, 8, 16];
const REPEATS: usize = 3;
const CANDIDATES_PER_CASE: usize = (2 + 4 + 8 + 16) * REPEATS;

fn argmax(logits: &[u16]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            half::f16::from_bits(**left)
                .to_f32()
                .total_cmp(&half::f16::from_bits(**right).to_f32())
        })
        .map(|(index, _)| index as u32)
        .expect("nonempty logits")
}

fn prompt(context: usize, seed: u32) -> Vec<u32> {
    (0..context)
        .map(|index| 1 + ((index as u32 * 7919 + seed * 1237) % 4096))
        .collect()
}

fn positions(start: usize, rows: usize) -> Vec<u32> {
    (start..start + rows)
        .map(|position| position as u32)
        .collect()
}

#[test]
#[ignore = "V0b: loads target and runs 3 prompts × 4 contexts × 3 interleaved verifier rounds"]
fn ideal_target_continuations_bound_verify_throughput() {
    let model_dir = std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL");
    let mut model = TextModel::load_from_dir(Path::new(&model_dir)).expect("target model");
    let vocab = model.vocab_size();

    for seed in 0..3u32 {
        for context in CONTEXTS {
            let prefix = prompt(context, seed);
            let prefix_positions = positions(0, context);

            // Save a known greedy target continuation. The measured verifier
            // later consumes this exact sequence, grouped into interleaved
            // 2/4/8/16-row blocks. Ground-truth generation is outside timing.
            model.reset_gpu_caches();
            let mut logits = model
                .forward_buffer(&prefix, &prefix_positions)
                .expect("ground-truth prefill");
            let mut continuation = Vec::with_capacity(CANDIDATES_PER_CASE + 1);
            for step in 0..=CANDIDATES_PER_CASE {
                let token = argmax(&logits);
                continuation.push(token);
                logits = model
                    .forward_buffer(&[token], &[(context + step) as u32])
                    .expect("ground-truth greedy step");
            }
            assert!(continuation.iter().all(|token| (*token as usize) < vocab));

            // Replay the same prefix once, then measure a single interleaved
            // A/B schedule. Each verify block appends exactly the known target
            // tokens, so all shapes start from a valid sequential continuation
            // and do not require repeated context prefills.
            model.reset_gpu_caches();
            let mut previous_logits = model
                .forward_buffer(&prefix, &prefix_positions)
                .expect("timed prefix prefill");
            let mut cursor = 0usize;
            let mut samples = Vec::with_capacity(PROPOSAL_ROWS.len() * REPEATS);

            for repeat in 0..REPEATS {
                for rows in PROPOSAL_ROWS {
                    assert_eq!(
                        argmax(&previous_logits),
                        continuation[cursor],
                        "saved continuation diverged before prompt {seed}, context {context}, M={rows}, repeat={repeat}"
                    );
                    let input = &continuation[cursor..cursor + rows];
                    let row_positions = positions(context + cursor, rows);
                    let started = Instant::now();
                    let verified = model
                        .forward_verify(input, &row_positions)
                        .expect("ideal target verification");
                    let elapsed = started.elapsed().as_secs_f64() * 1e3;
                    assert_eq!(verified.len(), rows * vocab);

                    for row in 0..rows {
                        assert_eq!(
                            argmax(&verified[row * vocab..(row + 1) * vocab]),
                            continuation[cursor + row + 1],
                            "verify row {row} disagrees with saved continuation at prompt {seed}, context {context}, M={rows}, repeat={repeat}"
                        );
                    }
                    previous_logits = verified[(rows - 1) * vocab..rows * vocab].to_vec();
                    samples.push((rows, elapsed));
                    eprintln!(
                        "V0b prompt={seed} context={context} M={rows} repeat={repeat} full_verify_ms={elapsed:.3} processed_ceiling_tok_s={:.3} unprocessed_ceiling_tok_s={:.3}",
                        (rows + 1) as f64 * 1000.0 / elapsed,
                        rows as f64 * 1000.0 / elapsed,
                    );
                    cursor += rows;
                }
            }
            assert_eq!(cursor, CANDIDATES_PER_CASE);

            for rows in PROPOSAL_ROWS {
                let mut timings: Vec<f64> = samples
                    .iter()
                    .filter_map(|(sample_rows, ms)| (*sample_rows == rows).then_some(*ms))
                    .collect();
                timings.sort_by(f64::total_cmp);
                let median = timings[timings.len() / 2];
                eprintln!(
                    "V0b MEDIAN prompt={seed} context={context} M={rows} verify_ms={median:.3} processed_ceiling_tok_s={:.3} unprocessed_ceiling_tok_s={:.3}",
                    (rows + 1) as f64 * 1000.0 / median,
                    rows as f64 * 1000.0 / median,
                );
            }
        }
    }
}
