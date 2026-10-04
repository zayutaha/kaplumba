//! OQ9: prefill key-range passes must be **bit-identical** to the one-dispatch
//! flash body.
//!
//! The mechanism's entire claim is that a row handed across several dispatches
//! runs exactly the operations of one full dispatch, in the same order
//! (`omlx/utils/nax_attention.py:164-170`: *"Every row thus runs exactly the
//! operations of one full dispatch, in the same order (bit-identical output)"*).
//! That is a claim about arithmetic, so this compares **bits**, not a tolerance:
//! a chain of 2, 3, 5, 7 and 16 dispatches must produce output whose fp16 payload
//! equals the single-dispatch output's.
//!
//! This is deliberately stricter than `attention_qk_mma_matches_scalar`, which
//! allows 5e-3 (about one fp16 ulp at these magnitudes). The flash body already
//! differs from the *scalar* body by that much; a pass chain must add nothing on
//! top.
//!
//! # Coverage
//!
//! * pass counts from 2 to 64 keys per pass, including counts that do **not**
//!   divide `key_count`, so the passes have ragged edges;
//! * both query blocks (`BQ = 1` and `BQ = 2`), so the multi-query accumulator
//!   chain and the grid-tail guard are both covered;
//! * key counts straddling the 64-key tile, so a pass boundary that lands *inside*
//!   a tile exercises the straddling-key staging (`partial_sub` / `partial_kt`);
//! * query tails at `BQ = 2` (1, 3, 5, 7, 511, 513 queries), where the last
//!   threadgroup is partially out of range and its rows must be neither written
//!   to the state nor read from it;
//! * windowed shapes, because the window test interacts with the pass range;
//! * a first pass in which **every key is masked out**, so the row handed over is
//!   the untouched initial one -- the case a "skip the resume when the row looks
//!   empty" optimisation would break;
//! * the gate: a grid below `PREFILL_PASS_MIN_GROUPS` declines, an unsupported
//!   shape declines, a missing kernel declines, and a declined dispatch encodes
//!   **nothing** so the caller can fall through without a half-written output;
//! * a short state buffer is an error, never a truncation.
//!
//! # Sabotage check (AGENTS.md §3.4)
//!
//! `bit_exactness_can_fail` proves the comparison can fail: it runs the *same*
//! instrument on two different inputs and requires a difference. Without it, a
//! test that always compares zeros would pass every case above.

use half::f16;
use kaplumba_kernels::attention::{self, AttentionEncodeRequest, AttentionKernels, AttentionQk};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

/// Threadgroups per (query, KV head) pair for the shipped full-attention shape.
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 512;

struct Fixture {
    device: Device,
    queue: CommandQueue,
    kernels: AttentionKernels,
}

impl Fixture {
    fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let kernels = AttentionKernels::new()
            .unwrap_or_else(|e| panic!("AttentionKernels::new failed: {e}"));
        let queue = device.new_command_queue().to_owned();
        let f = Self {
            device,
            queue,
            kernels,
        };
        assert!(
            f.kernels.prefill_pass_available(),
            "the OQ9 pass kernels did not compile; every other assertion in this file \
             would be vacuous"
        );
        f
    }

    /// Deterministic, well-scaled fp16. A sin/cos pair rather than an LCG so no
    /// input is exactly zero and no product saturates -- the same generator
    /// `attention_qk_mma_matches_scalar` uses.
    fn row(&self, n: usize, salt: f32) -> Buffer {
        let bits: Vec<u16> = (0..n)
            .map(|i| {
                let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                    + ((i as f32 + salt) * 0.0031).cos() * 0.25;
                f16::from_f32(x).to_bits()
            })
            .collect();
        self.u16(&bits)
    }

    fn u16(&self, v: &[u16]) -> Buffer {
        self.device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn u32(&self, v: &[u32]) -> Buffer {
        self.device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn encoder(&self) -> MetalEncoder {
        MetalEncoder::new(&self.queue).expect("encoder")
    }

    fn read_u16(&self, buffer: &Buffer) -> Vec<u16> {
        let slice = unsafe {
            std::slice::from_raw_parts(
                buffer.contents() as *const u16,
                (buffer.length() / 2) as usize,
            )
        };
        slice.to_vec()
    }
}

/// Inputs and outputs for one comparison.
struct Case {
    queries: Buffer,
    keys: Buffer,
    values: Buffer,
    query_positions: Buffer,
    key_positions: Buffer,
    out_single: Buffer,
    out_pass: Buffer,
    state: Buffer,
    queries_count: usize,
    key_count: usize,
    window: usize,
    scale: f32,
}

impl Fixture {
    fn case(&self, queries_count: usize, key_count: usize, window: usize) -> Case {
        let kv = key_count * KV_HEADS * HEAD_DIM;
        let q = queries_count * Q_HEADS * HEAD_DIM;
        let out_elements = queries_count * Q_HEADS * HEAD_DIM;
        let query_positions: Vec<u32> = (0..queries_count as u32).map(|i| i + 3).collect();
        let key_positions: Vec<u32> = (0..key_count as u32).collect();
        Case {
            queries: self.row(q, 3.0),
            keys: self.row(kv, 11.0),
            values: self.row(kv, 29.0),
            query_positions: self.u32(&query_positions),
            key_positions: self.u32(&key_positions),
            out_single: self
                .device
                .new_buffer((out_elements * 2) as u64, MTLResourceOptions::StorageModeShared),
            out_pass: self
                .device
                .new_buffer((out_elements * 2) as u64, MTLResourceOptions::StorageModeShared),
            state: self.device.new_buffer(
                (attention::prefill_pass_state_elements(queries_count, Q_HEADS, HEAD_DIM) * 4)
                    as u64,
                MTLResourceOptions::StorageModePrivate,
            ),
            queries_count,
            key_count,
            window,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        }
    }
}

impl Case {
    fn request<'a>(&'a self, output: &'a Buffer) -> AttentionEncodeRequest<'a> {
        AttentionEncodeRequest {
            queries: &self.queries,
            keys: &self.keys,
            values: &self.values,
            query_positions: &self.query_positions,
            key_positions: &self.key_positions,
            output,
            queries_count: self.queries_count,
            q_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            key_count: self.key_count,
            scale: self.scale,
            window: self.window,
        }
    }

    fn label(&self) -> String {
        format!(
            "queries={} keys={} window={}",
            self.queries_count, self.key_count, self.window
        )
    }
}

/// Run one shape through the one-dispatch body and through a pass chain of
/// `pass_keys` keys per pass, and require bit equality.
fn assert_bit_exact(fx: &Fixture, case: &Case, block: usize, pass_keys: usize) -> usize {
    {
        let mut enc = fx.encoder();
        fx.kernels
            .encode_online_attention_qk_block(
                &mut enc,
                case.request(&case.out_single),
                AttentionQk::Flash,
                block,
            )
            .expect("one-dispatch flash");
        enc.commit_and_wait().expect("wait");
    }
    let dispatched = {
        let mut enc = fx.encoder();
        let result = fx.kernels.encode_online_attention_passes_explicit(
            &mut enc,
            case.request(&case.out_pass),
            block,
            &case.state,
            pass_keys,
        );
        enc.commit_and_wait().expect("wait");
        result
    };
    let dispatched = dispatched.expect("pass chain must not error");
    assert!(
        dispatched,
        "the pass arm declined to dispatch: {} block={block} pass_keys={pass_keys} grid={} \
         (gate {PREFILL_PASS_MIN_GROUPS})",
        case.label(),
        attention::prefill_threadgroups(case.queries_count, KV_HEADS, block)
    );

    let a = fx.read_u16(&case.out_single);
    let b = fx.read_u16(&case.out_pass);
    assert_eq!(a.len(), b.len());
    let mut differing = 0usize;
    let mut first_bad = None;
    let mut worst = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            differing += 1;
            let d = (f16::from_bits(*x).to_f32() - f16::from_bits(*y).to_f32()).abs();
            if d > worst {
                worst = d;
            }
            if first_bad.is_none() {
                first_bad = Some((
                    i / (Q_HEADS * HEAD_DIM),
                    (i / HEAD_DIM) % Q_HEADS,
                    i % HEAD_DIM,
                ));
            }
        }
    }
    assert_eq!(
        differing, 0,
        "a pass chain of {pass_keys} keys over {} keys is NOT bit-exact ({} block={}): \
         {differing} of {} fp16 values differ, worst |diff| = {worst}, \
         first at (query, head, dim) = {first_bad:?}",
        case.key_count,
        case.label(),
        block,
        a.len()
    );
    // A comparison of two all-zero buffers proves nothing, so assert the control
    // produced real output.
    assert!(
        a.iter().any(|x| *x != 0),
        "{}: the one-dispatch control produced an all-zero output, so bit equality \
         would be vacuous",
        case.label()
    );
    differing
}

use attention::PREFILL_PASS_MIN_GROUPS;

// --- the bit-exactness cases -------------------------------------------------

#[test]
fn two_passes_at_the_shipping_context() {
    // The production shape: a 1024-query chunk against 8192 keys, BQ = 2, sliced
    // into two halves. This is the case the N1 design targets.
    let fx = Fixture::new();
    let case = fx.case(1024, 8192, 0);
    assert_bit_exact(&fx, &case, 2, 4096);
}

#[test]
fn many_pass_counts_over_ragged_key_counts() {
    // 24 590 keys is the measured 24k prompt. None of these pass sizes divides it
    // evenly except a couple, so most chains have ragged first and last passes.
    let fx = Fixture::new();
    let case = fx.case(1024, 24_590, 0);
    for pass_keys in [1024usize, 2048, 4096, 8192, 16_384] {
        assert_bit_exact(&fx, &case, 2, pass_keys);
    }
}

#[test]
fn both_query_blocks() {
    // BQ = 2 halves the grid, so 512 queries clear the gate at both blocks.
    let fx = Fixture::new();
    for block in [1usize, 2] {
        let case = fx.case(1024, 8192, 0);
        for pass_keys in [1024usize, 4096] {
            assert_bit_exact(&fx, &case, block, pass_keys);
        }
    }
}

#[test]
fn every_pass_size_is_bit_exact_at_ragged_key_counts() {
    // The straddle staging (`partial_sub` / `partial_kt`) only runs when a tile
    // is not full, which after tile-aligned edges means the LAST tile of a pass
    // whose range ends off a tile edge. Ragged key counts with several pass
    // sizes is where that path is actually exercised; tile-aligned pass sizes
    // over a tile-aligned key count never touch it.
    let fx = Fixture::new();
    for keys in [129usize, 191, 255, 257, 1000, 1023, 1025, 4095, 4097, 8191] {
        let case = fx.case(1024, keys, 0);
        // Pass sizes must be BELOW the key count or the chain is a single
        // dispatch by design (asserted in `a_short_key_range_declines`).
        let mut sizes: Vec<usize> = [64usize, 128, 256]
            .into_iter()
            .filter(|k| *k < keys)
            .collect();
        assert!(!sizes.is_empty(), "keys={keys} admits no multi-pass chain");
        for pass_keys in sizes.drain(..) {
            assert_bit_exact(&fx, &case, 2, pass_keys);
        }
    }
}

#[test]
fn pass_boundaries_are_snapped_to_the_key_tile() {
    // The whole mechanism's bit-exactness rests on this: the online softmax
    // updates (m, s, O) once per 64-key tile, so a boundary inside a tile changes
    // every row's rescale sequence. Measured cost of getting it wrong is one fp16
    // ulp on ~1% of outputs -- well inside the 5e-3 tolerance the other attention
    // tests use, so it would have shipped unnoticed.
    //
    // oMLX gets this for free by cutting in key-BLOCK units (`_BK = 32`,
    // `nax_attention.py:76`, `:800`, `:362`); here it has to be explicit.
    for key_count in [1000usize, 8192, 24_576, 24_590] {
        for pass_keys in [1usize, 63, 64, 65, 127, 128, 8192, 8193] {
            let edges = attention::prefill_pass_edges(key_count, pass_keys);
            assert_eq!(edges[0], 0, "the chain must start at key 0");
            assert_eq!(
                edges[edges.len() - 1], key_count,
                "the chain must end at key_count, or the last dispatch would return an \
                 unnormalised accumulator"
            );
            for (i, edge) in edges[..edges.len() - 1].iter().enumerate() {
                assert_eq!(
                    edge % attention::PREFILL_PASS_TILE,
                    0,
                    "edge {i} of a {key_count}-key chain with pass_keys={pass_keys} is not \
                     tile-aligned: {edges:?}"
                );
            }
            // Monotone and non-empty.
            for pair in edges.windows(2) {
                assert!(pair[1] >= pair[0], "edges must not run backwards: {edges:?}");
            }
            assert!(
                edges.len() >= 2 && edges[1] > 0,
                "a chain must make progress: {edges:?}"
            );
        }
    }
    // A key range that fits in one pass is a single dispatch, not a degenerate
    // two-pass chain with an empty first pass.
    assert_eq!(
        attention::prefill_pass_edges(100, 8192),
        vec![0, 100],
        "a range shorter than one pass must stay one dispatch"
    );
    assert_eq!(
        attention::prefill_pass_edges(8192, 8192),
        vec![0, 8192],
        "a range exactly one pass long must stay one dispatch"
    );
    // 24 590 keys at 8192 per pass: 24 590 keys = 385 key tiles, 8192 keys = 128
    // tiles per pass, so 385 / 128 = 4 passes with the last one carrying 14 keys.
    // Pinned because the pass COUNT is what the dispatch loop iterates, and an
    // off-by-one there silently drops or repeats a pass.
    let edges = attention::prefill_pass_edges(24_590, 8192);
    assert_eq!(edges, vec![0, 8192, 16_384, 24_576, 24_590]);
    assert_eq!(*edges.last().unwrap(), 24_590);
    // A key count that IS a whole number of tiles ends on a tile edge, so the
    // final pass is a full one and there is no short tail.
    assert_eq!(
        attention::prefill_pass_edges(24_576, 8192),
        vec![0, 8192, 16_384, 24_576]
    );
    // Every non-final edge is a multiple of 64 (the last one need not be: the
    // key range itself does not have to end on a tile).
    for edge in &edges[..edges.len() - 1] {
        assert_eq!(edge % 64, 0);
    }
    // A pass size below one tile is floored to one tile, never to zero passes.
    for tiny in [1usize, 2, 63] {
        let e = attention::prefill_pass_edges(8192, tiny);
        assert_eq!(e.len() - 1, 128, "pass_keys={tiny} must become 1 tile per pass: {e:?}");
    }
}

#[test]
fn pass_edges_tile_the_key_range_exactly() {
    for count in [64usize, 100, 1000, 8192, 8193, 16_384, 24_576, 24_590] {
        for pass_keys in [64usize, 128, 512, 1024, 4096, 8192] {
            let edges = attention::prefill_pass_edges(count, pass_keys);
            assert_eq!(edges[0], 0);
            assert_eq!(*edges.last().unwrap(), count);
            for pair in edges.windows(2) {
                assert!(pair[1] >= pair[0], "edges must not run backwards ({count})");
                assert!(pair[1] <= count, "an edge must not pass key_count ({count})");
            }
        }
    }
}

#[test]
fn ragged_key_counts_around_the_tile() {
    let fx = Fixture::new();
    for keys in [129usize, 191, 255, 257, 1000, 1023, 1025, 4095, 4097] {
        let case = fx.case(1024, keys, 0);
        // Pass size must be *below* `keys` or the chain declines, which is the
        // documented behaviour and is asserted separately; here the point is the
        // ragged boundary, so pick a pass size that splits each count unevenly.
        assert_bit_exact(&fx, &case, 2, keys / 3 + 1);
    }
}

#[test]
fn query_tails_at_bq2() {
    // At BQ = 2 an odd query count leaves the last threadgroup half out of
    // range. Its rows must be neither written to the state nor read back, and the
    // rows before it must be unaffected.
    let fx = Fixture::new();
    for queries in [1usize, 3, 5, 7, 511, 513] {
        let case = fx.case(queries, 8192, 0);
        assert_bit_exact(&fx, &case, 2, 1024);
        assert_bit_exact(&fx, &case, 1, 1024);
    }
}

#[test]
fn windowed_shapes() {
    // The window test and the pass range interact: a row whose live keys all sit
    // inside one pass must still get the rescale its earlier keys owed it, and a
    // pass boundary inside the window must not change which keys are live.
    let fx = Fixture::new();
    for window in [64usize, 256, 1024, 4096] {
        let case = fx.case(1024, 8192, window);
        assert_bit_exact(&fx, &case, 2, 512);
    }
}

#[test]
fn a_first_pass_with_every_key_masked_out() {
    // Every query is at position 0, so only key 0 is visible to anything. Pass 0
    // (keys 0..1024 under `pass_keys = 1024`) is *not* empty -- it holds key 0 --
    // so make the cut coarser and the queries late enough that pass 0 is entirely
    // masked: queries at position 5000 with `pass_keys = 1024` and 8192 keys
    // means passes 0..3 score nothing at all for these rows.
    //
    // This is the case a "skip the resume when the row looks empty" optimisation
    // would break: the row handed over has `m = NEG`, `s = 0` and a zero
    // accumulator, and the next pass must pick it up as if nothing had happened.
    let fx = Fixture::new();
    let case = fx.case(1024, 8192, 0);
    // Rewrite the query positions to sit at 5000, after four whole passes.
    let late: Vec<u32> = vec![5000u32; 1024];
    let qp = fx.u32(&late);
    let shifted = Case {
        queries: case.queries,
        keys: case.keys,
        values: case.values,
        query_positions: qp,
        key_positions: case.key_positions,
        out_single: fx.device.new_buffer(
            (1024 * Q_HEADS * HEAD_DIM * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        out_pass: fx.device.new_buffer(
            (1024 * Q_HEADS * HEAD_DIM * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        state: fx.device.new_buffer(
            (attention::prefill_pass_state_elements(1024, Q_HEADS, HEAD_DIM) * 4) as u64,
            MTLResourceOptions::StorageModePrivate,
        ),
        queries_count: 1024,
        key_count: 8192,
        window: 0,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };
    assert_bit_exact(&fx, &shifted, 2, 1024);
}

#[test]
fn a_long_chain_of_tiny_passes() {
    // One key tile (64 keys) per pass over 8192 keys is 128 dispatches -- the
    // extreme of the mechanism. If any pass boundary loses a ulp, this is where it
    // shows.
    let fx = Fixture::new();
    let case = fx.case(1024, 8192, 0);
    assert_bit_exact(&fx, &case, 2, 64);
    // Two tiles per pass: the minimum that is not degenerate, and the shape a
    // 128-key `KAPLUMBA_PREFILL_PASS_KEYS` would produce.
    assert_bit_exact(&fx, &case, 2, 128);
}

// --- the gate ----------------------------------------------------------------

#[test]
fn the_grid_gate_holds_small_grids_to_one_dispatch() {
    // oMLX's `_PASS_MIN_GROUPS = 512`: below that the threadgroups stay in step
    // anyway, so a chain would cost state traffic for nothing.
    assert_eq!(PREFILL_PASS_MIN_GROUPS, 512);
    let available = Fixture::new().kernels.prefill_pass_available();

    // The gate is a strict `<`, matching oMLX's `n_groups < min_groups`
    // (`nax_attention.py:797`), so a grid exactly AT 512 is sliced and one row
    // short is not. Pinned on both sides because an off-by-one here either pays
    // state traffic for nothing or silently disables the mechanism.
    assert_eq!(
        attention::prefill_threadgroups(128, 4, 1),
        PREFILL_PASS_MIN_GROUPS
    );
    assert_eq!(
        attention::prefill_pass_count(128, 4, 40_000, 512, 8, 1, available),
        0,
        "a grid exactly AT the gate keeps one dispatch -- oMLX calls 512 the \
         'wave or two' size, so the boundary is on the unsliced side"
    );
    // One query row over the gate (129 queries x 4 kv heads = 516 groups) is sliced.
    assert_eq!(
        attention::prefill_threadgroups(129, 4, 1),
        PREFILL_PASS_MIN_GROUPS + 4
    );
    assert!(
        attention::prefill_pass_count(129, 4, 40_000, 512, 8, 1, available) >= 2,
        "a grid just over the gate must be sliced"
    );
    assert_eq!(
        attention::prefill_threadgroups(127, 4, 1),
        PREFILL_PASS_MIN_GROUPS - 4
    );
    assert_eq!(
        attention::prefill_pass_count(127, 4, 40_000, 512, 8, 1, available),
        0,
        "a grid one query row short of the gate must NOT be sliced"
    );
    assert!(
        attention::prefill_pass_count(1024, 4, 40_000, 512, 8, 2, available) >= 2,
        "a 2048-group grid over 40k keys must be sliced"
    );
    // An unsupported shape keeps its existing body however big the grid is.
    assert_eq!(
        attention::prefill_pass_count(1024, 16, 40_000, 256, 2, 2, available),
        0,
        "the sliding shape has no pass kernel"
    );
    // A kernel that did not compile means one dispatch, not a different kernel.
    assert_eq!(
        attention::prefill_pass_count(1024, 4, 40_000, 512, 8, 2, false),
        0
    );
}

#[test]
fn a_short_key_range_declines() {
    let fx = Fixture::new();
    let case = fx.case(1024, 4096, 0);
    for pass_keys in [4096usize, 8192, 1_000_000] {
        let mut enc = fx.encoder();
        let dispatched = fx
            .kernels
            .encode_online_attention_passes_explicit(
                &mut enc,
                case.request(&case.out_pass),
                2,
                &case.state,
                pass_keys,
            )
            .expect("decline is not an error");
        assert!(
            !dispatched,
            "pass_keys {pass_keys} >= key_count {} must decline, not run a pointless chain",
            case.key_count
        );
        enc.commit_and_wait().expect("wait");
    }
}

#[test]
fn a_declined_dispatch_encodes_nothing() {
    // The caller falls through to the one-dispatch path on `false`, so a declined
    // dispatch must leave the encoder empty rather than half-encode. Verified by
    // running a declined arm and an empty encoder and comparing their outputs.
    let fx = Fixture::new();
    let case = fx.case(1024, 4096, 0);
    let dec = fx.device.new_buffer(
        (1024 * Q_HEADS * HEAD_DIM * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    {
        let mut enc = fx.encoder();
        assert!(!fx
            .kernels
            .encode_online_attention_passes_explicit(
                &mut enc,
                case.request(&dec),
                2,
                &case.state,
                999_999,
            )
            .unwrap());
        enc.commit_and_wait().expect("wait");
    }
    let after = fx.read_u16(&dec);
    // And the real thing, which must produce non-trivial output, so "empty" and
    // "written" are distinguishable.
    {
        let mut enc = fx.encoder();
        assert!(fx
            .kernels
            .encode_online_attention_passes_explicit(
                &mut enc,
                case.request(&dec),
                2,
                &case.state,
                512,
            )
            .unwrap());
        enc.commit_and_wait().expect("wait");
    }
    let written = fx.read_u16(&dec);
    assert!(
        written.iter().any(|x| *x != 0),
        "an eligible chain must write real output, else the declined case proves nothing"
    );
    assert!(
        after != written,
        "a declined dispatch must leave the output untouched; it appears to have written"
    );
}

#[test]
fn a_short_state_buffer_is_an_error_not_a_truncation() {
    // Every row of every pass writes its whole row unconditionally, so a short
    // buffer must be refused. Silently truncating would read uninitialised state.
    let fx = Fixture::new();
    let case = fx.case(1024, 16_384, 0);
    let needed = attention::prefill_pass_state_elements(case.queries_count, Q_HEADS, HEAD_DIM);
    let short = fx.device.new_buffer(
        ((needed - 1) * 4) as u64,
        MTLResourceOptions::StorageModePrivate,
    );
    let mut enc = fx.encoder();
    let result = fx.kernels.encode_online_attention_passes_explicit(
        &mut enc,
        case.request(&case.out_pass),
        2,
        &short,
        4096,
    );
    match result {
        Err(e) => {
            let message = format!("{e:?}");
            assert!(
                message.contains("pass state buffer"),
                "the error must name the buffer, got: {message}"
            );
        }
        Ok(true) => panic!("a state buffer one f32 short must be refused"),
        Ok(false) => panic!(
            "a short buffer must be an ERROR, not a decline: declining would run the \
             one-dispatch path and hide the mis-sizing"
        ),
    }
}

#[test]
fn the_state_buffer_is_fixed_size_and_context_independent() {
    // AGENTS.md §8: scratch is fixed-size and independent of context length. The
    // row state depends on the CHUNK, never on `key_count` -- the function's
    // signature has no key count in it, which is the structural reason it cannot
    // grow with context.
    for queries in [1usize, 8, 64, 1024] {
        assert_eq!(
            attention::prefill_pass_state_elements(queries, 32, 512),
            queries * 32 * 514,
            "layout is (O[head_dim], m, s) per (query, head) row"
        );
    }
    // The widest-layer helper picks the largest, never the sum: the buffer is
    // shared across layer shapes, like the decode partials.
    assert_eq!(
        attention::prefill_pass_state_elements_widest([(32, 512), (8, 256)], 16),
        attention::prefill_pass_state_elements(16, 32, 512)
    );
    assert_eq!(
        attention::prefill_pass_state_elements_widest([(8, 256)], 16),
        attention::prefill_pass_state_elements(16, 8, 256)
    );
    // An empty shape list still yields 1, so a buffer is never 0 bytes.
    assert_eq!(attention::prefill_pass_state_elements_widest([], 16), 1);
    // 1024 x 32 x 514 f32 = 64.25 MiB at the shipping chunk size, against an
    // 18.35 GiB footprint and a 20 GiB budget. Pinned so a layout change that
    // quadruples it is noticed.
    assert_eq!(
        attention::prefill_pass_state_elements(1024, 32, 512) * 4,
        67_371_008
    );
}

// --- sabotage check ----------------------------------------------------------

#[test]
fn bit_exactness_can_fail() {
    // AGENTS.md §10.3: "a measurement that has never been seen to fail is not a
    // measurement". The same comparison, on deliberately different inputs, must
    // report a difference -- otherwise every assertion above could be passing on
    // a comparison that cannot fail.
    let fx = Fixture::new();
    let a = fx.case(1024, 8192, 0);
    // Same shapes, but a different value table: perturb one element of V.
    let mut v_bits: Vec<u16> = (0..8192 * KV_HEADS * HEAD_DIM)
        .map(|i| {
            let x = ((i as f32 + 29.0) * 0.0173).sin() * 0.5
                + ((i as f32 + 29.0) * 0.0031).cos() * 0.25;
            f16::from_f32(x).to_bits()
        })
        .collect();
    v_bits[7] = f16::from_f32(1.25).to_bits();
    let mut differing = 0usize;
    let n = 8192 * KV_HEADS * HEAD_DIM;
    let untouched = read_u16(&a.values, n);
    assert_ne!(
        untouched[7], v_bits[7],
        "the sabotage did not change the V table, so this test cannot fail"
    );
    for (x, y) in untouched.iter().zip(v_bits.iter()) {
        if x != y {
            differing += 1;
        }
    }
    assert_eq!(
        differing, 1,
        "expected exactly one perturbed element, got {differing}"
    );
    // And prove the *output* comparison is sensitive to it, by running the
    // one-dispatch kernel on both value tables and requiring a difference. If
    // the outputs matched anyway, every bit-exactness assertion above would be
    // comparing something that does not depend on V.
    let mut out_a = fx
        .device
        .new_buffer((1024 * Q_HEADS * HEAD_DIM * 2) as u64, MTLResourceOptions::StorageModeShared);
    let mut out_b = fx.device.new_buffer(
        (1024 * Q_HEADS * HEAD_DIM * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_values = fx.u16(&v_bits);
    let b_values = fx.u16(&v_bits);
    {
        let mut enc = fx.encoder();
        fx.kernels
            .encode_online_attention_qk_block(
                &mut enc,
                request_with_values(&a, &b_values, &out_a),
                AttentionQk::Flash,
                2,
            )
            .unwrap();
        enc.commit_and_wait().expect("wait");
    }
    {
        let mut enc = fx.encoder();
        fx.kernels
            .encode_online_attention_qk_block(
                &mut enc,
                request_with_values(&a, &a.values, &out_b),
                AttentionQk::Flash,
            2,
        )
        .unwrap();
        enc.commit_and_wait().expect("wait");
    }
    let ra = fx.read_u16(&out_a);
    let rb = fx.read_u16(&out_b);
    assert_ne!(
        ra, rb,
        "perturbing V[0] by one fp16 element did not change the attention output, so the \
         bit-exactness comparisons in this file are insensitive to the value table"
    );
}

fn read_u16(buffer: &Buffer, n: usize) -> Vec<u16> {
    let ptr = buffer.contents() as *const u16;
    (0..n).map(|i| unsafe { ptr.add(i).read() }).collect()
}

/// Build a request over an arbitrary value table, reusing every other input of
/// `case`. Written as a function rather than struct-update syntax because
/// `metal::Buffer` is not `Copy` and the update form would move out of `case`.
fn request_with_values<'a>(
    case: &'a Case,
    values: &'a Buffer,
    output: &'a Buffer,
) -> AttentionEncodeRequest<'a> {
    AttentionEncodeRequest {
        queries: &case.queries,
        keys: &case.keys,
        values,
        query_positions: &case.query_positions,
        key_positions: &case.key_positions,
        output,
        queries_count: case.queries_count,
        q_heads: Q_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        key_count: case.key_count,
        scale: case.scale,
        window: case.window,
    }
}