//! spec/dflash-correctness T7 (R5): export the target's post-layer hidden
//! states at the drafter's tap layers.
//!
//! Four properties, in increasing order of how easily a plausible bug survives
//! them:
//!
//! 1. **Shape.** One `tokens * hidden_size` buffer per requested layer, and it
//!    carries signal (not zeros).
//! 2. **The capture does not perturb the forward.** The same prompt must produce
//!    bitwise-identical logits with taps enabled and disabled. The blit runs
//!    inside the layer loop between two ping-pong buffers, so a wrong source, a
//!    missing barrier or an aliased buffer can silently corrupt the model's own
//!    output — which would then be blamed on the drafter.
//! 3. **Parity.** This is the sharp one. Layer `k` writes into `hidden_b` when
//!    `k` is even and `hidden_a` when `k` is odd, so the blit must select its
//!    source per layer. **With a hardcoded `hidden_a`, a tap at even layer 2
//!    captures `hidden_a` — which at that moment still holds layer 1's output —
//!    so `tap[2] == tap[1]` exactly.** The DFlash taps `[1, 12, 23, 35, 46, 57]`
//!    are four odd, two even: the bug is invisible for four of the six layers and
//!    silently wrong for two.
//! 4. **Order.** `enable_taps(&[a, b])` and `enable_taps(&[b, a])` must return the
//!    layers in the requested order, because the drafter's `fc` was trained on
//!    the config's exact `target_layer_ids` concatenation.
//!
//! Ignored by default: it loads the 16 GB target and runs the real GPU forward.
//!
//! ```text
//! KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked \
//!   cargo test -p kaplumba-engine --release --test dflash_taps -- --ignored --nocapture
//! ```

use kaplumba_engine::model::TextModel;

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required")
}

fn prompt() -> Vec<u32> {
    vec![
        2_u32, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
    ]
}

fn differing(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; runs the full GPU forward path"]
fn taps_capture_the_layers_they_claim_without_perturbing_the_forward() {
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_path()))
        .expect("load target model");
    let hidden = model.hidden_size();
    let tokens = prompt();
    let positions: Vec<u32> = (0..tokens.len() as u32).collect();
    let rows = tokens.len();

    // --- 2. baseline: no taps at all -------------------------------------
    model.reset_gpu_caches();
    let without = model
        .forward_buffer(&tokens, &positions)
        .expect("forward without taps");
    // Logits come back as f16 bit patterns, so a non-zero bit pattern is the
    // "there is signal here" check (+0 and -0 are the only zero encodings).
    assert!(
        without.iter().any(|bits| (*bits & 0x7fff) != 0),
        "the baseline forward produced all-zero logits"
    );

    // --- 1. capture two adjacent layers, one odd and one even ------------
    // Layer 1 is odd (writes hidden_a) and layer 2 is even (writes hidden_b),
    // which is exactly the pair the parity bug collapses.
    model.enable_taps(&[1, 2]).expect("enable taps");
    assert_eq!(model.tap_layers(), &[1, 2]);
    model.reset_gpu_caches();
    let with = model
        .forward_buffer(&tokens, &positions)
        .expect("forward with taps");

    let taps = model.take_taps(rows).expect("take taps");
    assert_eq!(taps.len(), 2, "one buffer per requested layer");
    for (slot, values) in taps.iter().enumerate() {
        assert_eq!(
            values.len(),
            rows * hidden,
            "tap {slot} is {} values, expected {rows} rows of {hidden}",
            values.len()
        );
        assert!(
            values.iter().any(|value| *value != 0),
            "tap {slot} is all zeros: the blit never ran or wrote the wrong buffer"
        );
    }

    // --- 2. the forward is unchanged by capturing ------------------------
    assert_eq!(
        without, with,
        "enabling tap capture changed the model's own logits, so the blit is \
         stealing from or racing the layer it sits beside"
    );

    // --- 3. parity: layer 2's tap must not be layer 1's hidden state -----
    let layer_one = &taps[0];
    let layer_two = &taps[1];
    let same = differing(layer_one, layer_two);
    eprintln!(
        "tap[1] (odd, hidden_a) vs tap[2] (even, hidden_b): {same} of {} elements differ",
        layer_one.len()
    );
    assert!(
        same > layer_one.len() / 2,
        "tap[1] and tap[2] agree on {} of {} elements. A tap at an even layer is \
         nearly identical to the previous odd layer's, which is the signature of \
         the source being hardcoded to one ping-pong buffer instead of selected \
         per layer",
        layer_one.len() - same,
        layer_one.len()
    );

    // --- 4. request order is honoured ------------------------------------
    model.enable_taps(&[2, 1]).expect("enable taps reversed");
    assert_eq!(model.tap_layers(), &[2, 1]);
    model.reset_gpu_caches();
    model
        .forward_buffer(&tokens, &positions)
        .expect("forward with reversed taps");
    let reversed = model.take_taps(rows).expect("take reversed taps");
    assert_eq!(
        &reversed[0], layer_two,
        "tap[2] captured first should be the same buffer as tap[2] captured second"
    );
    assert_eq!(
        &reversed[1], layer_one,
        "tap[1] captured second should be the same buffer as tap[1] captured first"
    );
}
