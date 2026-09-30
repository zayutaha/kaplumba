//! Generated text must not change when weights are mapped instead of copied.
//!
//! This is the guard for the whole offload design. A dropped offset, a truncated
//! byte offset, or a wrong BF16 widening does not raise an error: it produces
//! fluent-looking *noise*, which is exactly the failure mode that made every
//! earlier zero-copy build look "almost working". Grepping logs will not catch
//! it; comparing text will.
//!
//! Expected strings were captured from the all-resident build (commit 78b3198)
//! at `--temperature 0`, where generation is deterministic. The real CLI binary
//! is exercised rather than the library, because the CLI is what users run and it
//! covers tokenisation, sampling, detokenisation, and streaming together.

use std::process::Command;

const CASES: &[(&str, &str)] = &[
    ("Hi", "Hello! How can I help you today?"),
    (
        "What is the capital of France?",
        "The capital of France is Paris.",
    ),
    ("Count from one to five.", "1, 2, 3, 4, 5"),
    (
        "Name three primary colors.",
        "The three primary colors are red, blue, and yellow.",
    ),
    (
        "Write one sentence about the ocean.",
        "The ocean is a vast, mysterious expanse of blue that covers most of our \
         planet and holds countless undiscovered secrets.",
    ),
];

fn binary() -> String {
    std::env::var("KAPLUMBA_BIN").unwrap_or_else(|_| {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .find(|dir| dir.join("Cargo.toml").is_file() && dir.join("crates").is_dir())
            .expect("workspace root")
            .to_path_buf();
        workspace
            .join("target/release/kaplumba")
            .to_string_lossy()
            .into_owned()
    })
}

/// First line the CLI prints for `prompt`, generated greedily.
fn generate(prompt: &str) -> String {
    let model = std::env::var("KAPLUMBA_TEST_MODEL").unwrap();
    let out = Command::new(binary())
        .args([
            "--model",
            &model,
            "--prompt",
            prompt,
            "--max-new-tokens",
            "32",
            "--temperature",
            "0",
        ])
        .output()
        .expect("failed to run the kaplumba binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "CLI failed for {prompt:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout.lines().next().unwrap_or("").trim().to_string()
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and a release binary; five greedy decodes"]
fn mapped_weights_produce_identical_text() {
    let mut failures = Vec::new();
    for (prompt, expected) in CASES {
        let actual = generate(prompt);
        if actual != *expected {
            failures.push(format!(
                "prompt {prompt:?}\n  expected: {expected}\n  actual:   {actual}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "generated text changed with mapped weights:\n{}",
        failures.join("\n")
    );
}

/// The escape hatch must produce the same text too; only the residency differs.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and a release binary; five greedy decodes"]
fn resident_weights_produce_the_same_text() {
    let mut failures = Vec::new();
    for (prompt, expected) in CASES {
        let model = std::env::var("KAPLUMBA_TEST_MODEL").unwrap();
        let out = Command::new(binary())
            .args([
                "--model",
                &model,
                "--prompt",
                prompt,
                "--max-new-tokens",
                "32",
                "--temperature",
                "0",
            ])
            .env("KAPLUMBA_RESIDENT_WEIGHTS", "1")
            .output()
            .expect("failed to run the kaplumba binary");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let actual = stdout.lines().next().unwrap_or("").trim().to_string();
        if actual != *expected {
            failures.push(format!(
                "prompt {prompt:?}\n  expected: {expected}\n  actual:   {actual}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "all-resident text diverged from the recorded baseline:\n{}",
        failures.join("\n")
    );
}
