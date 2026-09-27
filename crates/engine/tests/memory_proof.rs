//! Resident-memory proof.
//!
//! The point of this test is that `ru_maxrss` is the wrong instrument on macOS:
//! it reports resident pages only and misses Metal buffer memory, reading
//! ~7.0 GiB for an engine the kernel actually charges ~15.7 GiB. The figure the
//! 18 GB budget is about is `phys_footprint`, which only the OS will tell you.
//!
//! This test therefore shells out to `/usr/bin/time -l`, which reports Apple's
//! own "peak memory footprint", and cross-checks it against what the engine
//! says it deliberately allocated.

use kaplumba_engine::metrics::measure_rss;
use kaplumba_engine::model::TextModel;

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL")
        .expect("set KAPLUMBA_TEST_MODEL to the Gemma 4 checkpoint directory")
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and /usr/bin/time; resident memory proof"]
fn resident_memory_proof() {
    let path = model_path();

    // 1. What the engine says it deliberately keeps resident.
    let model = TextModel::load_from_dir(std::path::Path::new(&path)).unwrap();
    let (weights, kv, scratch) = model.allocation_breakdown();
    let engine_total = model.allocated_bytes();
    let gib = |b: u64| b as f64 / 1_073_741_824.0;

    println!("MEM weights_resident   = {:>8.2} GiB", gib(weights));
    println!("MEM kv_cache_resident  = {:>8.2} GiB", gib(kv));
    println!("MEM scratch_resident   = {:>8.3} GiB", gib(scratch));
    println!("MEM engine_allocated   = {:>8.2} GiB", gib(engine_total));
    const BUDGET: u64 = 18 * 1_073_741_824;
    assert!(weights > 14_000_000_000, "weights should be ~14.9 GiB, got {weights}");
    assert!(engine_total < BUDGET, "engine allocation must fit 18 GiB budget");
    drop(model);

    // 2. What the kernel charges, per Apple. Run the real binary so the number
    //    includes Metal buffers, the KV cache, and everything else.
    let exe = std::env::var("KAPLUMBA_BIN").unwrap_or_else(|_| {
        // cargo runs tests with cwd = the crate dir, so anchor to the workspace.
        let manifest = env!("CARGO_MANIFEST_DIR");
        let workspace = std::path::Path::new(manifest)
            .ancestors()
            .find(|dir| dir.join("Cargo.toml").is_file() && dir.join("crates").is_dir())
            .expect("workspace root")
            .to_path_buf();
        workspace.join("target/release/kaplumba").to_string_lossy().into_owned()
    });
    let out = std::process::Command::new("/usr/bin/time")
        .arg("-l")
        .arg(&exe)
        .args(["--model", &path, "--prompt", "Hi", "--max-new-tokens", "8", "--temperature", "0"])
        .output()
        .unwrap_or_else(|e| panic!("failed to run {exe}: {e}"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed: {stderr}");

    let footprint = stderr
        .lines()
        .find_map(|line| line.split_whitespace().last())
        .and_then(|_| {
            stderr
                .lines()
                .find(|line| line.contains("peak memory footprint"))
                .and_then(|line| line.split_whitespace().next())
                .and_then(|value| value.parse::<u64>().ok())
        })
        .expect("no 'peak memory footprint' line from /usr/bin/time -l");

    println!("MEM ru_maxrss          = {:>8.2} GiB", gib(measure_rss().unwrap()));
    println!("MEM phys_footprint     = {:>8.2} GiB   <- the real figure", gib(footprint));
    println!(
        "MEM metal_hidden       = {:>8.2} GiB   (footprint - ru_maxrss)",
        gib(footprint.saturating_sub(measure_rss().unwrap()))
    );
    println!("MEM budget_18GiB       = used {:.2} GiB of 18.00 GiB", gib(footprint));

    assert!(footprint > 0);
    assert!(
        footprint < BUDGET,
        "footprint {:.2} GiB exceeds the 18 GiB budget",
        gib(footprint)
    );
}
