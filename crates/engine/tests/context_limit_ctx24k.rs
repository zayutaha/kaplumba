//! T5 (`spec/prefill-120tps`): context extension to 24k -- requirement R4,
//! acceptance criteria 6 and 9.
//!
//! What is *not* here, deliberately: the generic KV ladder policy (walk to the
//! policy limit, clamp, refuse past the bound) is already pinned by
//! `tests/kv_growth_policy.rs`. This file pins the three things T5 actually
//! changed or validated and that the generic file does not see:
//!
//! 1. **The staged-token buffer covers every index a legal run can produce.**
//!    `host_tokens` is indexed by *decode step within a run*
//!    (`generate.rs:1013`, `chat.rs:470`), and the largest legal step is
//!    `max_new_tokens - 1` where only `prompt + max_new <= context_limit` is
//!    checked (`generate.rs:131-145`). With the old 8192-slot buffer a short
//!    prompt with `--max-new-tokens 10000` was a legal run whose GPU writes
//!    (`host_token_out[step_idx]`) and host reads (`read_host_token`, an
//!    unchecked `*ptr.add`) went **out of bounds**. Sized to the policy window,
//!    the buffer covers the index space by construction.
//! 2. **A 24k prompt fits the policy window with generation headroom**
//!    (AC9: 24k prefill demonstrated; the arithmetic that admits it).
//! 3. **The KV cache reaches 24k at the model's real full-attention shape and
//!    keeps its contents across the growth**, and sliding-window layers stay
//!    capped at their 1024 window while ringing through 24k positions
//!    (`model.rs:1811` only calls `ensure_capacity` for `full` layers).
//!
//! No model, no checkpoint: `GpuKvCache` and `ForwardScratch` allocate their
//! own Metal buffers.

use kaplumba_engine::chat::fits_in_context;
use kaplumba_engine::generate::DEFAULT_CONTEXT_LIMIT;
use kaplumba_engine::gpu_cache::{GpuCacheError, GpuKvCache, MAX_POLICY_CAPACITY};
use kaplumba_engine::model::{ForwardScratch, HOST_TOKEN_SLOTS};
use metal::Device;

/// `'Explain the theory of relativity. ' * 3990`, measured: T0 and run A both
/// report `prompt_tokens=23954`.
const PROMPT_24K: usize = 23954;

/// What `model.rs:3584` puts in `LayerConfig::max_sequence_len` for this
/// checkpoint (the *bound*; the policy clamp below is the real limit).
const MAX_POSITION_EMBEDDINGS: usize = 262_144;

/// `model.rs:1077`, `INITIAL_FULL_CACHE`.
const INITIAL_FULL_CACHE: usize = 512;

/// 24k, the acceptance criterion's own figure (AC9 says ">= 24k").
const AC9_SLOTS: usize = 24 * 1024;

fn device() -> Device {
    Device::system_default().expect("a Metal device")
}

/// AC9 + the out-of-bounds fix: the buffer that decode steps are staged into
/// must cover every `step_idx` a legal run can produce, and AC9 asks for at
/// least 24k slots outright.
#[test]
fn the_staged_token_buffer_covers_the_whole_legal_step_index_space() {
    assert!(
        HOST_TOKEN_SLOTS >= AC9_SLOTS,
        "AC9 requires HOST_TOKEN_SLOTS >= 24k, got {HOST_TOKEN_SLOTS}"
    );
    assert!(
        HOST_TOKEN_SLOTS >= DEFAULT_CONTEXT_LIMIT,
        "HOST_TOKEN_SLOTS ({HOST_TOKEN_SLOTS}) must cover every decode step index \
         a legal run can produce: generate.rs bounds only `prompt + max_new` by \
         DEFAULT_CONTEXT_LIMIT ({DEFAULT_CONTEXT_LIMIT}), so step_idx reaches \
         {DEFAULT_CONTEXT_LIMIT} - 1 and host_tokens[step_idx] would be written \
         and read OUT OF BOUNDS at the old size"
    );
    assert_eq!(
        HOST_TOKEN_SLOTS, MAX_POLICY_CAPACITY,
        "the staged-token buffer and the KV growth bound are both indexed by \
         position-adjacent run state and must be the same window; two constants \
         that can drift apart is how the 8192/32768 split happened in the first place"
    );
}

/// The same invariant through the allocation itself: the buffer really is
/// `HOST_TOKEN_SLOTS` u32 wide, and that width covers AC9's 24k. If someone
/// decouples the allocation from the constant (the historical failure mode),
/// the constant's promise becomes a lie while every `HOST_TOKEN_SLOTS` assert
/// still passes -- so the buffer length is checked too.
#[test]
fn the_host_tokens_buffer_is_really_slots_wide() {
    let Ok(scratch) = ForwardScratch::new(
        /* max_tokens = */ 8,
        /* hidden = */ 16,
        /* q_max = */ 16,
        /* kv_max = */ 16,
        /* inter = */ 16,
        /* vocab = */ 16,
        /* attn_partials = */ 16,
        /* attn_pass_state = */ 16,
    ) else {
        eprintln!("no Metal device; skipping");
        return;
    };
    assert_eq!(
        scratch.host_tokens.length(),
        (HOST_TOKEN_SLOTS * std::mem::size_of::<u32>()) as u64,
        "host_tokens length must be HOST_TOKEN_SLOTS * 4 bytes"
    );
    assert!(
        scratch.host_tokens.length() >= (AC9_SLOTS * std::mem::size_of::<u32>()) as u64,
        "host_tokens is {} bytes, fewer than 24k slots (AC9)",
        scratch.host_tokens.length()
    );
}

/// The arithmetic that admits the 24k run: 23954 prompt tokens + 8 generated
/// fits the policy window, exactly at the window fits, one over does not --
/// and the error carries the window it refused against.
#[test]
fn a_24k_prompt_fits_the_policy_window_with_generation_headroom() {
    let budget = DEFAULT_CONTEXT_LIMIT - PROMPT_24K;
    assert!(
        budget >= 8,
        "the policy window {DEFAULT_CONTEXT_LIMIT} leaves only {budget} tokens of \
         generation after a 24k prompt"
    );
    assert!(fits_in_context(PROMPT_24K, 0, 8, DEFAULT_CONTEXT_LIMIT).is_ok());
    // Exactly at the limit fits, one over does not (the contract chat.rs:102
    // documents, restated at 24k rather than at the old 8192 literal).
    assert!(fits_in_context(PROMPT_24K, 0, budget, DEFAULT_CONTEXT_LIMIT).is_ok());
    assert!(fits_in_context(PROMPT_24K, 0, budget + 1, DEFAULT_CONTEXT_LIMIT).is_err());
    match fits_in_context(PROMPT_24K, 0, budget + 1, DEFAULT_CONTEXT_LIMIT) {
        Err(kaplumba_engine::chat::ChatError::ContextOverflow { used, need, limit }) => {
            assert_eq!((used, need, limit), (PROMPT_24K, budget + 1, DEFAULT_CONTEXT_LIMIT));
        }
        other => panic!("expected ContextOverflow, got {other:?}"),
    }
    // The KV growth bound must admit the same position count (model.rs:3584 ->
    // gpu_cache.rs:228 clamp). 24k is reachable iff the clamp is the policy,
    // not the checkpoint's 262144 -- and it must not be *below* the policy.
    assert!(
        MAX_POLICY_CAPACITY >= AC9_SLOTS,
        "KV growth is capped at {MAX_POLICY_CAPACITY}, below the 24k AC9 requires"
    );
    assert!(
        MAX_POSITION_EMBEDDINGS >= AC9_SLOTS,
        "the checkpoint declares {MAX_POSITION_EMBEDDINGS} positions, below 24k"
    );
}

/// Full-attention layer at the model's real shape (4 KV heads x 512 head dim =
/// 2048-wide stream, 8196 B per position): grow from the 512-position initial
/// allocation to the 24k prompt's 23981st position, with live contents that
/// must survive every rung of the ladder.
#[test]
fn the_full_attention_cache_grows_to_24k_and_keeps_its_contents() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, INITIAL_FULL_CACHE, 4, 512).unwrap();
    assert_eq!(cache.stream_len(), 2048, "full-attention layer stream width");

    // Three live positions, with markers in all three buffers: growth that
    // dropped or reordered the prefix is the failure this catches.
    assert_eq!(cache.reserve(0, 3, true), Ok(0));
    cache.advance(3);
    let keys = cache.keys();
    let values = cache.values();
    let positions = cache.positions();
    unsafe {
        let k = keys.contents() as *mut u16;
        let v = values.contents() as *mut u16;
        let p = positions.contents() as *mut u32;
        *k = 0xBEEF;
        *v = 0xF00D;
        *p = 42;
    }

    // Exactly what model.rs:1814 asks for per chunk: needed = offset + tokens,
    // bound = max_position_embeddings.
    let needed = PROMPT_24K + 1 + 8; // prompt + first decode token + 8 more
    cache
        .ensure_capacity(&device, needed, MAX_POSITION_EMBEDDINGS)
        .unwrap_or_else(|error| panic!("ensure_capacity({needed}) failed: {error}"));
    assert!(
        cache.capacity() >= needed,
        "capacity {} does not fit the 24k run's {} positions",
        cache.capacity(),
        needed,
    );
    assert!(
        cache.capacity() <= MAX_POLICY_CAPACITY,
        "capacity {} exceeds the policy window {MAX_POLICY_CAPACITY}",
        cache.capacity(),
    );
    assert_eq!(
        cache.next_position(),
        3,
        "growth must not move the write cursor"
    );

    unsafe {
        let k = cache.keys().contents() as *const u16;
        let v = cache.values().contents() as *const u16;
        let p = cache.positions().contents() as *const u32;
        assert_eq!(*k, 0xBEEF, "keys prefix lost across the growth ladder");
        assert_eq!(*v, 0xF00D, "values prefix lost across the growth ladder");
        assert_eq!(*p, 42, "positions prefix lost across the growth ladder");
        assert_eq!(
            *p.add(3),
            kaplumba_engine::gpu_cache::EMPTY_POSITION,
            "slot 3 must read empty after growth"
        );
    }

    // The 24k position itself is reservable -- the reserve that would have
    // failed with the old hardcoded 8192 growth bound ("position 9215 exceeds
    // capacity 8192", model.rs:3569-3573). Walk the cursor up first: `reserve`
    // requires the exact next position, as every forward chunk does.
    let at = PROMPT_24K - 3;
    cache.advance(at - cache.next_position());
    assert_eq!(
        cache.reserve(at, 1, true),
        Ok(at),
        "position {at} of a 24k run must be reservable"
    );
}

/// One past the window is still refused at the real shape, naming the clamped
/// bound. (The generic version of this lives in `kv_growth_policy.rs` with a
/// 1-wide stream; this pins it at the shape the 24k run actually allocates.)
#[test]
fn growth_past_the_policy_is_refused_at_the_real_shape() {
    let device = device();
    let mut cache = GpuKvCache::new(&device, INITIAL_FULL_CACHE, 4, 512).unwrap();
    let before = cache.capacity();
    assert_eq!(
        cache.ensure_capacity(&device, MAX_POLICY_CAPACITY + 1, MAX_POSITION_EMBEDDINGS),
        Err(GpuCacheError::PositionOutOfRange {
            position: MAX_POLICY_CAPACITY,
            capacity: MAX_POLICY_CAPACITY,
        }),
    );
    assert_eq!(cache.capacity(), before, "a refused growth must not reallocate");
}

/// Sliding-window layers (50 of 60) never grow: `model.rs:1811` calls
/// `ensure_capacity` only `if full`, and the ring serves 24k positions out of
/// a 1024-slot window. Walk the whole 24k run through a sliding cache and the
/// capacity must still be the window at the end.
#[test]
fn sliding_window_layers_stay_capped_at_1024_through_24k_positions() {
    let device = device();
    const WINDOW: usize = 1024;
    // 16 KV heads x 256 head dim, the sliding-layer shape.
    let mut cache = GpuKvCache::new(&device, WINDOW, 16, 256).unwrap();

    for position in 0..(PROMPT_24K + 8) {
        let slot = cache
            .reserve(position, 1, false)
            .unwrap_or_else(|error| panic!("sliding reserve at position {position}: {error}"));
        assert_eq!(slot, position % WINDOW, "ring slot mapping at {position}");
        cache.advance(1);
    }

    assert_eq!(
        cache.capacity(),
        WINDOW,
        "sliding capacity grew to {}; sliding layers must ring at their window",
        cache.capacity(),
    );
    assert_eq!(
        cache.key_count(false),
        WINDOW,
        "a wrapped ring scans exactly its window"
    );
    assert_eq!(
        cache.next_position(),
        PROMPT_24K + 8,
        "the cursor must reach the end of the run"
    );
}
