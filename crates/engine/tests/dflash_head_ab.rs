//! T11: target verify-head timing and greedy identity at M=16.
//!
//! Run against two builds in alternating order to compare the legacy rowwise
//! head with the batched GEMM. This is ignored because it loads the target.

use std::path::Path;
use std::time::Instant;

use kaplumba_engine::model::TextModel;

fn argmax(row: &[u16]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            half::f16::from_bits(**left)
                .to_f32()
                .total_cmp(&half::f16::from_bits(**right).to_f32())
        })
        .map(|(index, _)| index as u32)
        .expect("nonempty logits")
}

fn prompt(seed: u32) -> Vec<u32> {
    (0..32)
        .map(|index| 1 + ((index * 7919 + seed * 1237) % 4096))
        .collect()
}

fn positions(start: usize, rows: usize) -> Vec<u32> {
    (start..start + rows)
        .map(|position| position as u32)
        .collect()
}

#[test]
#[ignore = "T11: loads target; compare rowwise and batched verify heads in alternating builds"]
fn m16_verify_head_is_greedy_exact_and_timed() {
    let model_dir = std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL");
    let mut model = TextModel::load_from_dir(Path::new(&model_dir)).expect("target model");
    let vocab = model.vocab_size();

    for seed in 0..3u32 {
        model.reset_gpu_caches();
        let prefix = prompt(seed);
        let mut logits = model
            .forward_buffer(&prefix, &positions(0, prefix.len()))
            .expect("prefix prefill");
        let mut continuation = Vec::with_capacity(49);
        for index in 0..49 {
            let token = argmax(&logits);
            continuation.push(token);
            logits = model
                .forward_buffer(&[token], &[(prefix.len() + index) as u32])
                .expect("greedy continuation");
        }

        model.reset_gpu_caches();
        let _ = model
            .forward_buffer(&prefix, &positions(0, prefix.len()))
            .expect("replay prefix");
        for repeat in 0..3 {
            let cursor = repeat * 16;
            let started = Instant::now();
            let verified = model
                .forward_verify(
                    &continuation[cursor..cursor + 16],
                    &positions(prefix.len() + cursor, 16),
                )
                .expect("M=16 verify");
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(verified.len(), 16 * vocab);
            for row in 0..16 {
                assert_eq!(
                    argmax(&verified[row * vocab..(row + 1) * vocab]),
                    continuation[cursor + row + 1],
                    "prompt={seed} repeat={repeat} verify row {row}"
                );
            }
            eprintln!("T11 prompt={seed} repeat={repeat} M=16 verify_ms={elapsed_ms:.3}");
        }
    }
}
