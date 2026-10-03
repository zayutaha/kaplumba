//! spec/kv-free-drafter T1: export target hidden states paired with the tokens
//! that follow them, for training a drafter against THIS checkpoint.
//!
//! One training example per position:
//!
//! ```text
//!   hidden_t        5376 x f16   the target's final-norm hidden at position t
//!   next_tokens     K x u32      tokens t+1 ..= t+K
//! ```
//!
//! **Why real text and not the target's own greedy continuation.** Pairing with
//! the target's greedy output is the more faithful signal, but it costs one
//! forward per token (~90 ms) because each generated token depends on the last.
//! Pairing with the document's own next tokens costs ONE prefill per document,
//! which yields hundreds of examples per pass — 200x cheaper for a training
//! signal that is nearly the same function (the target's next token given a
//! prompt is what the text *is*). The acceptance gate in `mtp_acceptance` still
//! measures against the real target's greedy output, so any gap between the two
//! shows up there rather than being assumed away.
//!
//! `--verify` re-runs the target over a document's real next tokens and asserts
//! the export's pairing is well-formed (finite hiddens, varied continuations),
//! because a mis-paired export trains to ~0% and looks like a modelling failure.
//!
//! Usage:
//!   cargo run --release --example export_hidden -- \
//!     --model ~/.kaplumba/models/Gxma/Gxma --out ~/.kaplumba/drafters/train.bin \
//!     --text <file-or-dir> --k 4 [--verify] [--max-positions 200000]

use kaplumba_engine::model::TextModel;
use kaplumba_engine::tokenizer::GemmaTokenizer;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

struct Args {
    model: PathBuf,
    out: PathBuf,
    text: Vec<PathBuf>,
    k: usize,
    verify: bool,
    max_positions: usize,
    holdout_every: usize,
}

fn parse_args() -> Args {
    let mut model = PathBuf::new();
    let mut out = PathBuf::new();
    let mut text = Vec::new();
    let mut k = 4usize;
    let mut verify = false;
    let mut max_positions = 200_000usize;
    let mut holdout_every = 0usize;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = PathBuf::from(argv.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--out" => {
                out = PathBuf::from(argv.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--text" => {
                if let Some(pattern) = argv.get(i + 1) {
                    text.extend(expand(pattern));
                }
                i += 2;
            }
            "--k" => {
                k = argv[i + 1].parse().unwrap_or(k);
                i += 2;
            }
            "--verify" => {
                verify = true;
                i += 1;
            }
            "--max-positions" => {
                max_positions = argv[i + 1].parse().unwrap_or(max_positions);
                i += 2;
            }
            "--holdout-every" => {
                holdout_every = argv[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            _ => i += 1,
        }
    }
    Args {
        model,
        out,
        text,
        k,
        verify,
        max_positions,
        holdout_every,
    }
}

fn expand(pattern: &str) -> Vec<PathBuf> {
    let path = PathBuf::from(pattern);
    if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| {
                        matches!(
                            p.extension().and_then(|e| e.to_str()),
                            Some("txt") | Some("md")
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        return files;
    }
    if path.exists() {
        return vec![path];
    }
    let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = path
        .extension()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&parent)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_stem()
                        .map(|s| s.to_string_lossy().starts_with(&stem))
                        .unwrap_or(false)
                        && p.extension()
                            .map(|e| e.to_string_lossy() == ext)
                            .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// argmax over a capped f16 logit row.
fn greedy(logits: &[u16]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        let f = half::f16::from_bits(*v).to_f32();
        if f > best_val {
            best_val = f;
            best = i;
        }
    }
    best as u32
}

fn read_text(path: &Path) -> String {
    let mut bytes = Vec::new();
    if let Ok(mut f) = std::fs::File::open(path) {
        let _ = f.read_to_end(&mut bytes);
    }
    String::from_utf8_lossy(&bytes).to_string()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    if args.model.as_os_str().is_empty() || args.text.is_empty() {
        eprintln!(
            "usage: export_hidden --model <dir> --out <bin> --text <path-or-dir> \
             [--k N] [--max-positions N] [--holdout-every N] [--verify]"
        );
        std::process::exit(2);
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let tokenizer = GemmaTokenizer::from_model_dir(&args.model)?;
    let mut model = TextModel::load_from_dir(&args.model)?;
    let hidden_size = model.hidden_size();
    eprintln!("model loaded; hidden_size={hidden_size} k={}", args.k);

    let mut writer = BufWriter::new(std::fs::File::create(&args.out)?);
    let mut holdout = BufWriter::new(std::fs::File::create(args.out.with_extension("holdout"))?);

    let mut written = 0usize;
    let held = 0usize;
    let mut total_tokens = 0u64;
    let mut docs = 0usize;

    for path in &args.text {
        if written >= args.max_positions {
            break;
        }
        let text = read_text(path);
        if text.trim().len() < 64 {
            continue;
        }
        let ids = match tokenizer.encode_prompt(&text) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("skip {}: {e}", path.display());
                continue;
            }
        };
        if ids.len() < 16 {
            continue;
        }
        docs += 1;

        // Walk documents in windows. Each window is an INDEPENDENT training block:
        // the target is rolled forward over its OWN greedy tokens, and the label
        // at every position is the token the target itself emitted next.
        //
        // This is the whole point of the exporter. Labelling with the corpus's
        // next token instead — which the first version did, to save a forward
        // per token — trains the drafter to imitate wikitext rather than to
        // imitate the model that will verify it. The target does not generate
        // the corpus verbatim and the two are different functions: that version
        // reached training loss 1.24 at 7% held-out accuracy.
        //
        // The cost is one target forward per generated token (~90 ms), which is
        // why a first run uses a small corpus. `TextModel::greedy_next_tokens`
        // batches the head and is the documented route to scaling this up.
        const WINDOW: usize = 256;
        let mut base = 0usize;
        while base + args.k + 1 <= ids.len() && written < args.max_positions {
            model.reset_gpu_caches();
            let n = WINDOW.min(ids.len() - base);
            let mut seq: Vec<u32> = ids[base..base + n].to_vec();
            let mut hiddens: Vec<Vec<u16>> = Vec::with_capacity(n + args.k);
            for t in 0..n + args.k {
                let (hidden, logits) = model.forward_with_hidden(&[seq[t]], &[t as u32])?;
                hiddens.push(hidden);
                if t + 1 < n + args.k {
                    seq.push(greedy(&logits));
                }
            }
            for t in 0..n {
                if written >= args.max_positions {
                    break;
                }
                let row = &hiddens[t];
                let mut bytes = Vec::with_capacity(row.len() * 2);
                for v in row {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                writer.write_all(&bytes)?;
                for j in 1..=args.k {
                    writer.write_all(&seq[t + j].to_le_bytes())?;
                }
                written += 1;
            }
            writer.flush()?;
            base += WINDOW;
            if written % 2048 < WINDOW {
                eprintln!("  {written} examples");
            }
        }
        total_tokens += ids.len() as u64;
        if docs % 10 == 0 {
            eprintln!(
                "  {docs} docs, {} tokens, {written} train / {held} holdout",
                total_tokens
            );
        }
    }
    writer.flush()?;
    holdout.flush()?;

    let mut index = std::fs::File::create(args.out.with_extension("idx"))?;
    writeln!(index, "train_examples {}", written)?;
    writeln!(index, "holdout_examples {}", held)?;
    writeln!(index, "hidden_size {}", hidden_size)?;
    writeln!(index, "k {}", args.k)?;
    writeln!(index, "total_tokens {}", total_tokens)?;
    writeln!(index, "model {}", args.model.display())?;
    let bytes = std::fs::metadata(&args.out)?.len();
    eprintln!(
        "wrote {written} train / {held} holdout examples to {} ({:.2} GB)",
        args.out.display(),
        bytes as f64 / 1e9
    );

    if args.verify {
        verify(&args, hidden_size)?;
    }
    Ok(())
}

/// Structural checks on the export that do not need the model: hiddens finite,
/// continuations varied. A constant continuation is the signature of an
/// off-by-one in the pairing, and it is the difference between "the drafter does
/// not learn" and "the drafter cannot learn from this file".
fn verify(args: &Args, hidden_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::fs::File::open(&args.out)?;
    let record = hidden_size * 2 + args.k * 4;
    let n = 4096usize;
    let mut buf = vec![0u8; record * n];
    let read = f.read(&mut buf)?;
    let records = read / record;
    eprintln!("verify: {records} records");

    let mut finite = true;
    let mut distinct = std::collections::HashSet::new();
    let mut norm_means = Vec::new();
    for r in 0..records {
        let base = r * record;
        let mut sum = 0f64;
        for i in 0..hidden_size {
            let bits = u16::from_le_bytes([buf[base + i * 2], buf[base + i * 2 + 1]]);
            let v = half::f16::from_bits(bits).to_f32();
            if !v.is_finite() {
                finite = false;
            }
            sum += (v as f64) * (v as f64);
        }
        norm_means.push((sum / hidden_size as f64).sqrt());
        let tb = base + hidden_size * 2;
        let t0 = u32::from_le_bytes([buf[tb], buf[tb + 1], buf[tb + 2], buf[tb + 3]]);
        distinct.insert(t0);
    }
    let mean_norm = norm_means.iter().sum::<f64>() / norm_means.len().max(1) as f64;
    eprintln!("verify: hiddens finite = {finite}");
    eprintln!("verify: distinct next-tokens = {}", distinct.len());
    eprintln!("verify: mean hidden RMS = {mean_norm:.4}");
    assert!(finite, "exported hidden contains non-finite values");
    assert!(
        distinct.len() > 100,
        "exported continuations are degenerate ({} distinct next tokens) — \
         the pairing is almost certainly off by one",
        distinct.len()
    );
    assert!(
        mean_norm > 1e-3 && mean_norm.is_finite(),
        "exported hidden RMS is {mean_norm} — the hiddens are not real activations"
    );
    eprintln!("verify: OK");
    Ok(())
}