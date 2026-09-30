//! Offline tool to repack model weights into Column-Major coalesced format.
//!
//! Reorders all 3 bpw trellis tensors from [tiles_k, tiles_n, 48] to [tiles_n, tiles_k, 48].
//! Dequantization is 100% bit-for-bit exact; only tile order along the K dimension is made contiguous.
//!
//! Rule compliance:
//! - Single disk copy: Deletes existing destination before repacking.
//! - Leaves original model files untouched.

use kaplumba_engine::exl3::reorder_tensor_to_colmajor;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::Instant;

fn expand_tilde(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(&path[2..]);
        }
    }
    PathBuf::from(path)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let src_dir = if args.len() > 1 {
        expand_tilde(&args[1])
    } else {
        expand_tilde("~/.kaplumba/models/Gxma/Gxma")
    };
    let dst_dir = if args.len() > 2 {
        expand_tilde(&args[2])
    } else {
        expand_tilde("~/.kaplumba/models/Gxma-repacked")
    };

    println!("=== REPACK MODEL TO COLUMN-MAJOR ===");
    println!("Source:      {}", src_dir.display());
    println!("Destination: {}", dst_dir.display());

    if !src_dir.exists() {
        return Err(format!("Source directory {} does not exist", src_dir.display()).into());
    }

    // AGENTS.md §8 & user directive: You can only have 1 copy at a time. Delete old copy before creating new one.
    if dst_dir.exists() {
        println!("Deleting existing copy at {}...", dst_dir.display());
        fs::remove_dir_all(&dst_dir)?;
    }
    fs::create_dir_all(&dst_dir)?;

    let started = Instant::now();

    // 1. Copy config files and tokenizer
    println!("Copying config files...");
    for entry in fs::read_dir(&src_dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name_str = file_name.to_string_lossy();
        if name_str.ends_with(".safetensors") {
            continue; // Will be repacked below
        }
        let src_file = entry.path();
        let dst_file = dst_dir.join(&file_name);
        if name_str == "quantization_config.json" {
            let content = fs::read_to_string(&src_file)?;
            let mut val: Value = serde_json::from_str(&content)?;
            if let Some(obj) = val.as_object_mut() {
                obj.insert("layout".to_string(), json!("colmajor"));
            }
            fs::write(&dst_file, serde_json::to_string_pretty(&val)?)?;
            println!("  Created updated quantization_config.json (layout: colmajor)");
        } else if src_file.is_file() {
            fs::copy(&src_file, &dst_file)?;
            println!("  Copied {}", name_str);
        }
    }

    // 2. Read index to find all shards
    let index_path = src_dir.join("model.safetensors.index.json");
    let shards: Vec<String> = if index_path.exists() {
        let index_content = fs::read_to_string(&index_path)?;
        let index_json: Value = serde_json::from_str(&index_content)?;
        let mut shard_set = BTreeMap::new();
        if let Some(map) = index_json.get("weight_map").and_then(|v| v.as_object()) {
            for (_tensor, shard) in map {
                if let Some(shard_name) = shard.as_str() {
                    shard_set.insert(shard_name.to_string(), ());
                }
            }
        }
        shard_set.into_keys().collect()
    } else {
        vec!["model.safetensors".to_string()]
    };

    println!("Found {} shard(s) to process: {:?}", shards.len(), shards);

    let mut total_bytes_written = 0u64;

    for shard_name in shards {
        let src_shard = src_dir.join(&shard_name);
        let dst_shard = dst_dir.join(&shard_name);
        println!("\nRepacking shard {}...", shard_name);

        let shard_started = Instant::now();
        let mut src_file = BufReader::new(File::open(&src_shard)?);

        // Read header length
        let mut len_buf = [0u8; 8];
        src_file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;

        // Read header JSON
        let mut header_bytes = vec![0u8; header_len];
        src_file.read_exact(&mut header_bytes)?;
        let header_str = std::str::from_utf8(&header_bytes)?;
        let mut header_json: Value = serde_json::from_str(header_str)?;

        let header_obj = header_json
            .as_object_mut()
            .ok_or_else(|| "Header is not a JSON object")?;

        // Collect tensors sorted by start offset
        struct TensorEntry {
            name: String,
            shape: Vec<usize>,
            start: u64,
            end: u64,
        }

        let mut tensor_entries = Vec::new();
        for (name, val) in header_obj.iter() {
            if name == "__metadata__" {
                continue;
            }
            let shape: Vec<usize> = val
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
                .unwrap_or_default();
            let offsets = val
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .unwrap();
            let start = offsets[0].as_u64().unwrap();
            let end = offsets[1].as_u64().unwrap();
            tensor_entries.push(TensorEntry {
                name: name.clone(),
                shape,
                start,
                end,
            });
        }
        tensor_entries.sort_by_key(|t| t.start);

        // Update header JSON with modified shapes for colmajor trellis tensors
        for entry in &mut tensor_entries {
            if entry.name.ends_with(".trellis") && entry.shape.len() == 3 && entry.shape[2] == 48 {
                let tiles_k = entry.shape[0];
                let tiles_n = entry.shape[1];
                let new_shape = vec![tiles_n, tiles_k, 48];
                if let Some(val) = header_obj.get_mut(&entry.name) {
                    if let Some(obj) = val.as_object_mut() {
                        obj.insert("shape".to_string(), json!(new_shape));
                    }
                }
            }
        }

        // Serialize new header
        let new_header_bytes = serde_json::to_vec(&header_json)?;
        let new_header_len = new_header_bytes.len() as u64;

        // Open destination file
        let mut dst_file = BufWriter::with_capacity(16 * 1024 * 1024, File::create(&dst_shard)?);
        dst_file.write_all(&new_header_len.to_le_bytes())?;
        dst_file.write_all(&new_header_bytes)?;

        let payload_base = 8 + header_len as u64;
        let mut shard_repacked_tensors = 0;

        for entry in &tensor_entries {
            let tensor_len = (entry.end - entry.start) as usize;
            src_file.seek(SeekFrom::Start(payload_base + entry.start))?;

            if entry.name.ends_with(".trellis") && entry.shape.len() == 3 && entry.shape[2] == 48 {
                let tiles_k = entry.shape[0];
                let tiles_n = entry.shape[1];
                let in_f = tiles_k * 16;
                let out_f = tiles_n * 16;

                let mut raw_bytes = vec![0u8; tensor_len];
                src_file.read_exact(&mut raw_bytes)?;

                let mut u16_slice = Vec::with_capacity(tensor_len / 2);
                for chunk in raw_bytes.chunks_exact(2) {
                    u16_slice.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                }

                let colmajor_u16 = reorder_tensor_to_colmajor(&u16_slice, in_f, out_f)?;
                let mut out_bytes = Vec::with_capacity(tensor_len);
                for w in colmajor_u16 {
                    out_bytes.extend_from_slice(&w.to_le_bytes());
                }
                dst_file.write_all(&out_bytes)?;
                shard_repacked_tensors += 1;
            } else {
                // Copy in 8MB chunks
                let mut remaining = tensor_len;
                let mut buf = vec![0u8; 8 * 1024 * 1024];
                while remaining > 0 {
                    let to_read = remaining.min(buf.len());
                    src_file.read_exact(&mut buf[..to_read])?;
                    dst_file.write_all(&buf[..to_read])?;
                    remaining -= to_read;
                }
            }
        }

        dst_file.flush()?;
        let shard_size = dst_shard.metadata()?.len();
        total_bytes_written += shard_size;
        println!(
            "  Finished {} in {:.1}s: {:.2} GB written ({} trellis tensors repacked)",
            shard_name,
            shard_started.elapsed().as_secs_f64(),
            shard_size as f64 / 1e9,
            shard_repacked_tensors
        );
    }

    println!(
        "\nRepacking COMPLETE in {:.1}s! Total written: {:.2} GB at {}",
        started.elapsed().as_secs_f64(),
        total_bytes_written as f64 / 1e9,
        dst_dir.display()
    );

    Ok(())
}
