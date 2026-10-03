//! B3 probe: `top_k_order` must not allocate a vocabulary-sized index vector any
//! more. **CPU-only. Opens no checkpoint, dispatches no Metal.**
//!
//! # Why this is a separate binary
//!
//! It installs a `#[global_allocator]`, and an allocation counter is global to the
//! whole test binary. Measuring one call means reading the counter twice with
//! nothing but the call in between, so any *other* test in the same binary
//! allocating concurrently would inflate the reading. One test per binary is the
//! only way to make the number mean anything, hence this file rather than another
//! `#[test]` in `probe_topk_equality.rs`.
//!
//! # What it pins
//!
//! The change this belongs to removed a `(0..logits.len()).collect::<Vec<_>>()` —
//! 8 B x 262,144 = **2 MiB, written in full on every sampled token** — from
//! `top_k_order`. What replaced it allocates a fixed 256 KiB counting table
//! (`RADIX = 1 << 16` u32 counters) plus at most a few hundred indices, none of
//! which depends on the vocabulary width.
//!
//! So this asserts a *budget*, not an implementation: peak bytes per call must
//! stay far below anything vocabulary-sized. It is the gate that fails if someone
//! reinstates the index vector, and unlike the equality gates it cannot be
//! satisfied by a selection that returns the right answer the expensive way.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use kaplumba_engine::sampler::top_k_order;

struct Counting;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            let live = LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(pointer, layout);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = System.realloc(pointer, layout, new_size);
        if !moved.is_null() {
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            let live = LIVE_BYTES.fetch_add(new_size, Ordering::Relaxed) + new_size;
            PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Everything allocated between these two reads is `body`'s. `println!` and the
/// panic machinery allocate, so the measurement may not touch them.
fn peak_of<T>(body: impl FnOnce() -> T) -> (usize, T) {
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_BYTES.store(0, Ordering::Relaxed);
    let result = body();
    (PEAK_BYTES.load(Ordering::Relaxed), result)
}

const VOCAB: usize = 262_144;

/// The budget: one 65,536-entry `u32` counting table (256 KiB) plus the winner and
/// candidate vectors, which together hold `top_k + bucket` indices. Stated at
/// 4 MiB — **half** the 2 MiB index vector plus slack — so the gate is a wide
/// margin around the intended shape rather than a fingerprint of it, but still
/// an order of magnitude below anything `vocab`-sized at 4 x the width.
const BUDGET: usize = 4 << 20;

#[test]
fn top_k_order_does_not_allocate_a_vocabulary_sized_index_vector() {
    // A row with a wide peak, so the selection takes its worst allocation path: a
    // populated counting table and a bucket wide enough to need the second one.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut logits: Vec<f32> = (0..VOCAB)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 11) as f64 / (1_u64 << 53) as f64) as f32;
            // 20 distinct head values inside one bucket: the refinement path.
            24.0 - (index % 20) as f32 * 0.0009 - noise * 9.0
        })
        .collect();
    logits[0] = 31.0;

    for top_k in [1_usize, 64, 1024] {
        let (peak, order) = peak_of(|| top_k_order(&logits, top_k));
        assert_eq!(order.len(), top_k, "the selection returned the wrong count");
        assert!(
            peak < BUDGET,
            "top_k_order allocated {peak} B for a {VOCAB}-wide row at top_k={top_k}, \
             over the {BUDGET} B budget. A vocabulary-sized index vector is 2 MiB, so \
             this is the allocation the counting select was supposed to remove"
        );
        println!("top_k={top_k:<6} peak {peak:>9} B ({:.1} KiB)", peak as f64 / 1024.0);
    }

    // The width must not matter, which is the property the budget is really about:
    // at 4x the vocabulary the same code must not allocate 4x anything.
    let (peak, _) = peak_of(|| top_k_order(&logits, 64));
    let wide: Vec<f32> = logits.iter().copied().chain(logits.iter().copied()).collect();
    let (wide_peak, order) = peak_of(|| top_k_order(&wide, 64));
    assert_eq!(order.len(), 64);
    println!(
        "vocab {VOCAB} peak {peak} B vs vocab {} peak {wide_peak} B",
        wide.len()
    );
    assert!(
        wide_peak <= peak + (1 << 10),
        "doubling the vocabulary raised the peak from {peak} B to {wide_peak} B, so \
         something in the selection still scales with the row width"
    );
}