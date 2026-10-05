use kaplumba_kernels::attention::decode_splits;

/// ADVERSARY ARTIFACT (spec/prefill-24k-90tps) -- makes the "two rules agree at
/// short context" claim checkable instead of asserted.
///
/// `decode_split_rule()` is cached per process, so this prints the table under
/// whatever `KAPLUMBA_DECODE_SPLIT_RULE` the environment selects; run it twice.
///
/// What it is for:
///  * AC3's short-context row and N6's "five reference prompts byte-identical
///    between arms" can only be vacuous if the two rules really do evaluate to
///    the SAME split count at those key counts. This prints that.
///  * the sliding layers' key count is `min(position, ring capacity)` =
///    `min(position, 1024)`, so they are never affected by the flip -- the reason
///    is the ring, not the wave budget. `key_count = 1024` here shows that.

#[test]
fn print_the_two_rules_over_the_key_counts_the_spec_uses() {
    println!(
        "rule = {:?} (KAPLUMBA_DECODE_SPLIT_RULE)",
        kaplumba_kernels::attention::decode_split_rule()
    );
    for (label, q_heads, kv_heads) in [
        ("10 full  (32q/4kv/d512, 256-thread groups)", 32usize, 4usize),
        ("50 sliding (32q/16kv/d256, 64-thread groups)", 32, 16),
    ] {
        println!("-- {label} --");
        for key_count in [
            120usize, 626, 658, 832, 896, 1024, 2048, 4096, 8204, 8236, 16394, 16397, 24_590,
        ] {
            println!(
                "  key_count={key_count:<6} splits={:<3} grid={}",
                decode_splits(q_heads, kv_heads, key_count),
                decode_splits(q_heads, kv_heads, key_count) * kv_heads
            );
        }
    }
}