//! Throughput probes for checkpoint loading. Ignored by default.
//!
//! These measure the two candidate costs in the load path so the load strategy
//! can be chosen from numbers rather than guesses:
//!   * fresh-destination parallel copy (page faults + write bandwidth)
//!   * warm-destination parallel copy (write bandwidth only)
//!
//! Run with:
//!   cargo test --release -p kaplumba-engine --test load_perf -- --ignored --nocapture

use std::time::Instant;

use kaplumba_engine::metal::MetalContext;
use metal::MTLResourceOptions;
use rayon::prelude::*;

const TOTAL_BYTES: u64 = 12 * 1024 * 1024 * 1024;
const CHUNK: usize = 8 * 1024 * 1024;

fn parallel_copy(destinations: &[metal::Buffer], source: &[u8]) -> f64 {
    let started = Instant::now();
    let written: u64 = destinations
        .par_iter()
        .map(|buffer| {
            let length = buffer.length() as usize;
            let base = buffer.contents() as *mut u8;
            let mut offset = 0_usize;
            while offset < length {
                let take = CHUNK.min(length - offset);
                let from = offset % source.len();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source.as_ptr().add(from),
                        base.add(offset),
                        take,
                    );
                }
                offset += take;
            }
            length as u64
        })
        .sum();
    let seconds = started.elapsed().as_secs_f64();
    written as f64 / 1_000_000_000.0 / seconds
}

#[test]
#[ignore = "throughput probe; allocates many GiB"]
fn probes_copy_ceiling() {
    let context = MetalContext::new().expect("metal");
    let device = context.device();
    let source = vec![0x5a_u8; 256 * 1024 * 1024];

    let count = 24_usize;
    let each = TOTAL_BYTES / count as u64;
    let fresh: Vec<metal::Buffer> = (0..count)
        .map(|_| device.new_buffer(each, MTLResourceOptions::StorageModeShared))
        .collect();

    let cold = parallel_copy(&fresh, &source);
    let warm = parallel_copy(&fresh, &source);
    println!("destination bytes: {:.2} GB", TOTAL_BYTES as f64 / 1e9);
    println!("cold parallel copy: {cold:.2} GB/s");
    println!("warm parallel copy: {warm:.2} GB/s");
    println!(
        "projected full 14.88 GB load: cold {:.2}s / warm {:.2}s",
        14.88 / cold,
        14.88 / warm
    );
}
