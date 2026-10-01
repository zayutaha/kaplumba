// Standalone: how long does `Sampler::sample` take on a 262144-wide logit row,
// and how much of that is the full-vocabulary sort?
//
// This exists because chat at temperature 1.0 measured 12.2 tok/s against
// 15.9 greedy, and the two paths differ in exactly two ways: the sampler runs on
// the host, and the logits row (262144 fp16) is read back and converted. Which of
// those costs the 3.7 tok/s is not guessable -- so it is measured.
use std::time::Instant;

use kaplumba_engine::sampler::{Sampler, SamplerConfig};

fn main() {
    let n = 262_144usize;
    // Pseudo-random but deterministic, so the sort cannot be degenerate.
    let mut state = 0x9E3779B9u64;
    // Exponent 12..17 keeps every value normal and finite. A full random u16
    // includes NaN and inf patterns, and `Sampler::sample` rejects those -- which
    // is correct behaviour, and not what this is measuring.
    let raw: Vec<u16> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let exponent = 12u16 + ((state >> 48) % 6) as u16;
            let mantissa = ((state >> 32) & 0x03FF) as u16;
            (exponent << 10) | mantissa
        })
        .collect();
    let logits: Vec<f32> = raw
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();

    let mut sampler =
        Sampler::new(SamplerConfig { temperature: 1.0, top_k: 64, top_p: 0.95 }, 1).unwrap();

    for _ in 0..3 {
        let _ = sampler.sample(&logits).unwrap();
    }

    let iters = 30;
    let mut sample_total = 0.0;
    for _ in 0..iters {
        let started = Instant::now();
        let _ = sampler.sample(&logits).unwrap();
        sample_total += started.elapsed().as_secs_f64() * 1e3;
    }
    let sample_ms = sample_total / iters as f64;

    // The conversion the caller does per token before sampling: read the fp16 row
    // back off the GPU and widen it. Measured separately because it is not inside
    // `sample` and is easy to forget when attributing the cost.
    let mut convert_total = 0.0;
    for _ in 0..iters {
        let started = Instant::now();
        let widened: Vec<f32> = raw.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
        std::hint::black_box(&widened);
        convert_total += started.elapsed().as_secs_f64() * 1e3;
    }
    let convert_ms = convert_total / iters as f64;

    // A decode step is 1/15.9 s = 62.9 ms. These two are the host-side work added
    // to every token on the sampling path.
    println!("logit row:      {n} fp16 ({:.0} KiB)", n * 2 / 1024);
    println!("sample():       {sample_ms:.2} ms/token");
    println!("fp16 -> f32:    {convert_ms:.2} ms/token");
    println!("total host add: {:.2} ms/token", sample_ms + convert_ms);
    println!(
        "greedy step is 62.9 ms, so this is {:.1}% of a step",
        100.0 * (sample_ms + convert_ms) / 62.9
    );
    println!(
        "implied sampling rate if the GPU part is unchanged: {:.2} tok/s",
        1000.0 / (62.9 + sample_ms + convert_ms)
    );
}