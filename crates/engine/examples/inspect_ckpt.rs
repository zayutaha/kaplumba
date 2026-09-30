//! CPU-only checkpoint inspection: runs every metadata check in
//! `inspect_model_dir` without acquiring the GPU lock or touching Metal.
//!
//! `Model::load_from_dir` takes the kernel `flock` *before* it inspects, so a
//! metadata failure on a second checkpoint cannot be diagnosed while another
//! agent holds the GPU -- the run exits 75 and never reaches the check. This
//! entry point calls the same function directly, so the whole validation chain
//! is testable on a busy box.
//!
//! Usage: `cargo run --release --example inspect_ckpt -- <model_dir>`

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(arg) = std::env::args().nth(1) else {
        eprintln!("usage: inspect_ckpt <model_dir>");
        return ExitCode::from(2);
    };
    let dir = PathBuf::from(arg);
    match kaplumba_engine::safetensors::inspect_model_dir(&dir) {
        Ok(metadata) => {
            let q = metadata.config;
            println!("OK  {}", dir.display());
            println!("  quant: {} bpw linears, {} bpw head, codebook {}",
                q.quantization.bits, q.quantization.head_bits, q.quantization.codebook);
            println!("  layout: {:?}", metadata.quantization.layout);
            println!(
                "  tensor_storage modules: {}",
                metadata.quantization.tensor_storage.len()
            );
            println!("  text tensors: {}", metadata.selection.tensors.len());
            println!("  vision tensors excluded: {}", metadata.selection.excluded_vision.len());
            println!("  text bytes: {}", metadata.selection.text_bytes);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("FAIL {}\n  {error}", dir.display());
            ExitCode::FAILURE
        }
    }
}
