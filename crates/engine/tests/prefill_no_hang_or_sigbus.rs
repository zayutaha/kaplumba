use std::process::Command;

#[test]
#[ignore = "requires live model in ~/.kaplumba/models/Gxma-repacked"]
fn prefill_2k_runs_cleanly_without_hang_or_sigbus() {
    let model_path = std::path::PathBuf::from(
        std::env::var("HOME").unwrap(),
    ).join(".kaplumba/models/Gxma-repacked");

    if !model_path.exists() {
        eprintln!("Model path {:?} does not exist, skipping.", model_path);
        return;
    }

    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root_dir.join("target/release/kaplumba");

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(200);

    let output = Command::new(&bin_path)
        .current_dir(root_dir)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            &text,
            "--metrics",
            "--max-new-tokens",
            "1",
            "--temperature",
            "0",
        ])
        .output()
        .expect("failed to execute kaplumba binary");

    assert!(
        output.status.success(),
        "kaplumba exited with non-zero status: {:?}, stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("prefill_tok_s="),
        "metrics line missing from stdout: {}",
        stdout
    );

    // Extract prefill_tok_s and assert it is >= 85.0
    let mut rate_found = false;
    for part in stdout.split_whitespace() {
        if let Some(val_str) = part.strip_prefix("prefill_tok_s=") {
            let rate: f64 = val_str.parse().expect("valid prefill_tok_s float");
            assert!(
                rate >= 85.0,
                "prefill rate {:.2} tok/s fell below 85.0 floor",
                rate
            );
            rate_found = true;
            break;
        }
    }
    assert!(rate_found, "did not find prefill_tok_s in stdout: {}", stdout);
}
