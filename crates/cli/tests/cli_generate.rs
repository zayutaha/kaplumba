//! CLI end-to-end tests.
//!
//! The `generate` run is gated on the artifact **and** `KAPLUMBA_RUN_SLOW=1`: it
//! loads 16 GB and runs the naive CPU path (~3.2s per layer per token in
//! release, far worse unoptimised), so the default `cargo test` must stay fast.
//! It uses `--layers 2`; text quality is irrelevant — this proves the plumbing:
//! load → tokenize → generate → stream → `kaplumba-bench/1` report.
//!
//! Run it with:
//! `KAPLUMBA_RUN_SLOW=1 cargo test --release -p kaplumba-cli --test cli_generate`

use std::path::PathBuf;
use std::process::Command;

fn model_dir() -> Option<PathBuf> {
    let dir = match std::env::var("KAPLUMBA_MODEL_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let h = std::env::var("HOME").ok()?;
            PathBuf::from(format!("{h}/.kaplumba/models/Gxma/Gxma"))
        }
    };
    dir.join("config.json").is_file().then_some(dir)
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_kaplumba-engine")
}

#[test]
fn help_is_usage_and_exit_0() {
    for flag in ["--help", "-h"] {
        let out = Command::new(bin()).arg(flag).output().expect("run cli");
        assert!(out.status.success(), "{flag} should exit 0");
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("generate"), "usage mentions generate");
        assert!(text.contains("metal-smoke"), "usage mentions metal-smoke");
    }
}

#[test]
fn no_args_is_usage_on_stderr_exit_2() {
    let out = Command::new(bin()).output().expect("run cli");
    assert_eq!(out.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("usage:"),
        "usage goes to stderr"
    );
}

#[test]
fn unknown_flag_and_command_exit_2() {
    let bad_command = Command::new(bin()).arg("frobnicate").output().expect("run");
    assert_eq!(bad_command.status.code(), Some(2));

    let bad_flag = Command::new(bin())
        .args(["generate", "--nope"])
        .output()
        .expect("run");
    assert_eq!(bad_flag.status.code(), Some(2));
    let err = String::from_utf8_lossy(&bad_flag.stderr);
    assert!(err.contains("unknown flag"), "got: {err}");
    assert!(err.contains("usage:"), "usage follows the error");

    // Missing prompt is a usage error, not a crash (model load never starts).
    let no_prompt = Command::new(bin())
        .args(["generate", "--model", "/nonexistent", "--max-tokens", "1"])
        .output()
        .expect("run");
    assert_ne!(no_prompt.status.code(), Some(0));
}

#[test]
fn metal_smoke_runs_on_this_host() {
    let out = Command::new(bin())
        .arg("metal-smoke")
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        // No Metal device (non-macOS CI) is not a failure of the CLI contract.
        eprintln!(
            "skip: metal-smoke unavailable: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }
    assert!(stdout.contains("metal-smoke: OK"), "got: {stdout}");
}

#[test]
fn generate_emits_bench_report_and_streams_text() {
    if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
        eprintln!("skip: set KAPLUMBA_RUN_SLOW=1 (loads the 16 GB artifact)");
        return;
    }
    let Some(dir) = model_dir() else {
        eprintln!("skip: no artifact (KAPLUMBA_MODEL_DIR)");
        return;
    };
    let out = Command::new(bin())
        .args([
            "generate",
            "--model",
            dir.to_str().unwrap(),
            "--prompt",
            "The quick brown fox",
            "--max-tokens",
            "1",
            "--layers",
            "2",
            "--temp",
            "0",
            "--json",
            "-",
            "--quiet",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let report: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout is valid JSON when --json -");
    assert_eq!(report["schema"], "kaplumba-bench/1");
    assert_eq!(report["engine"]["layers_run"], 2);
    assert_eq!(report["engine"]["layers"], 60);
    assert_eq!(report["engine"]["kv_cache"], false);
    assert_eq!(report["engine"]["naive_recompute"], true);
    assert_eq!(report["sampling"]["greedy"], true);
    assert_eq!(report["mtp"], false);
    assert_eq!(report["decode"]["gen_tokens"], 1);
    assert_eq!(report["tokens"].as_array().map(|a| a.len()), Some(1));
    assert!(report["ttft"][0]["prompt_tokens"].as_u64().unwrap() >= 4);
    assert!(report["load_s"].as_f64().unwrap() > 0.0);
    assert!(report["mem"]["weights_bytes"].as_u64().unwrap() > 10_000_000_000);
    assert!(report["mem"]["peak_rss_bytes"].as_u64().unwrap() > 1_000_000);
    assert!(report["mem"]["metal_alloc_bytes"].as_u64().unwrap() > 0);
    // Roofline is the DE2 model of record: 273 / 15.5.
    assert_eq!(report["roofline"]["weights_gb"], 15.5);
    assert_eq!(report["roofline"]["tg_tps_ceiling"], 17.613);
    // Text streamed to stderr because the report owns stdout.
    assert!(
        !String::from_utf8_lossy(&out.stderr).contains("error"),
        "no errors on stderr"
    );
    let tok = report["tokens"][0].as_u64().unwrap();
    assert!(tok < 262144, "token id in vocab: {tok}");
}
