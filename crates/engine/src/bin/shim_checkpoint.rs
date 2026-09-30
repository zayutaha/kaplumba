//! Offline tool: make an EXL3 checkpoint directory loadable by writing the
//! `quantization_config.json` the engine requires, into a NEW directory.
//!
//! # Why this exists
//!
//! Some EXL3 exports ship a `config.json` whose `quantization_config` block
//! carries only the *summary* (`bits`, `head_bits`, `codebook`, `calibration`,
//! `out_scales`) and no per-module `tensor_storage` map, and ship no
//! `quantization_config.json` at all. The engine's
//! `inspect_model_dir` (crates/engine/src/safetensors.rs) requires both:
//! `validate_quantization_metadata` walks `tensor_storage` to cross-check every
//! per-tensor shape / byte count / dtype / bitrate against the shard header, and
//! `validate_text_graph_tensors` looks up a storage entry for all 411 quantized
//! modules and rejects any text tensor it did not expect. An absent file fails
//! the read; an empty map fails the lookup at
//! `safetensors.rs:712` ("missing quantization storage <module>").
//!
//! # What it does
//!
//! Reads the source directory's `config.json`, `model.safetensors.index.json`
//! and every shard's safetensors **header** (8-byte length prefix + JSON; the
//! payload is never read), then synthesizes a complete `quantization_config.json`
//! whose `tensor_storage` is derived from the headers rather than guessed. It
//! links the shards and the small JSON/tokenizer files into the destination, so
//! no weight bytes are copied and no source file is touched.
//!
//! The MCG multiplier is READ from each module's `.mcg` tensor and checked
//! against `kaplumba_engine`'s constant. That check is the point: the multiplier
//! is a codebook constant, so a checkpoint carrying a different one would decode
//! to plausible-looking noise, and a shim that hardcoded ours would write a
//! config that lies about the data.
//!
//! # Safety
//!
//! - Refuses to run when source and destination resolve to the same directory.
//! - Refuses to write into a destination that already has content (pass
//!   `--force` to replace a destination this tool created).
//! - Opens the source read-only. The destination must not be the source.
//! - Refuses if any required input is missing rather than guessing.
//!
//! # Usage
//!
//! ```text
//! shim_checkpoint <src_dir> <dst_dir> [--force]
//! ```

use kaplumba_engine::config::MCG_MULTIPLIER;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Suffixes that make up one quantized module, in the order the engine's
/// `TensorStorage::validate` looks for them.
const COMPONENT_SUFFIXES: &[&str] = &[".trellis", ".suh", ".su", ".svh", ".sv", ".mcg"];

/// The non-shard files the engine actually opens, from `inspect_model_dir`
/// (safetensors.rs), `GemmaTokenizer::from_model_dir` (tokenizer.rs) and the CLI
/// (crates/cli/src/main.rs). `quantization_config.json` is listed because it is
/// REQUIRED to exist -- the point of this tool is to write it -- but the
/// destination copy is synthesised rather than copied, so the tool skips it here.
const AUX_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "quantization_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// `foo.json` -> the first existing `foo(1).json`, `foo(2).json`, ...
fn numbered_variant(dir: &Path, stem: &str) -> Option<String> {
    for n in 1..100 {
        let candidate = format!("{stem}({n}).json");
        if dir.join(&candidate).is_file() {
            return Some(candidate);
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (src, dst, force) = parse_args()?;
    run(&src, &dst, force)
}

fn parse_args() -> Result<(PathBuf, PathBuf, bool), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if positional.len() != 2 {
        return Err(
            "usage: shim_checkpoint <src_dir> <dst_dir> [--force]".to_string().into(),
        );
    }
    let force = args.iter().any(|a| a == "--force");
    Ok((
        PathBuf::from(positional[0]),
        PathBuf::from(positional[1]),
        force,
    ))
}

fn run(src: &Path, dst: &Path, force: bool) -> Result<(), Box<dyn std::error::Error>> {
    // The refuse-if-same-path guard comes first, before anything is created, so
    // a mistaken invocation cannot truncate the source it was pointed at.
    let src_real = src
        .canonicalize()
        .map_err(|e| format!("source {} is unreadable: {e}", src.display()))?;
    if let Ok(dst_real) = dst.canonicalize() {
        if dst_real == src_real {
            return Err(format!(
                "refusing to write into the source directory {} (in-place edit)",
                src.display()
            )
            .into());
        }
    }
    if dst.exists() {
        let mut entries = fs::read_dir(dst)?;
        if entries.next().is_some() && !force {
            return Err(format!(
                "destination {} is not empty; refusing to overwrite (pass --force \
                 to replace a directory this tool created)",
                dst.display()
            )
            .into());
        }
        if force {
            fs::remove_dir_all(dst)?;
        }
    }

    println!("=== SHIM EXL3 CHECKPOINT ===");
    println!("source:      {}", src_real.display());
    println!("destination: {}", dst.display());

    let config_bytes = fs::read(src_real.join("config.json"))?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    let summary = config
        .get("quantization_config")
        .ok_or("config.json has no quantization_config block")?
        .clone();
    let bits = summary
        .get("bits")
        .and_then(Value::as_f64)
        .ok_or("quantization_config.bits is missing or not a number")?;
    let head_bits = summary
        .get("head_bits")
        .and_then(Value::as_u64)
        .ok_or("quantization_config.head_bits is missing")?;
    let codebook = summary
        .get("codebook")
        .and_then(Value::as_str)
        .ok_or("quantization_config.codebook is missing")?;
    if codebook != "mcg" {
        return Err(format!(
            "codebook {codebook} is not supported by this shim; it only knows how to \
             cross-check the MCG multiplier"
        )
        .into());
    }
    if bits.fract() != 0.0 || !(1.0..=8.0).contains(&bits) {
        return Err(format!("bits {bits} is not an integer rate in 1..=8").into());
    }
    if head_bits == 0 || head_bits > 8 {
        return Err(format!("head_bits {head_bits} is out of range 1..=8").into());
    }
    println!(
        "quantization: {bits} bpw linears, {head_bits} bpw head, codebook {codebook}, \
         version {}",
        summary
            .get("version")
            .and_then(Value::as_str)
            .unwrap_or("(none)")
    );

    // Read the index so the shard set and total_size come from the file the
    // engine will read, not from a directory listing.
    let index: Value =
        serde_json::from_slice(&fs::read(src_real.join("model.safetensors.index.json"))?)?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or("model.safetensors.index.json has no weight_map")?;

    // Shard headers only: shapes, dtypes and byte ranges. The payload is never
    // touched, so this is a few hundred KB of reads against 20 GB of weights.
    let mut headers: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
    let mut payload_starts: BTreeMap<String, u64> = BTreeMap::new();
    let mut shards: Vec<String> = weight_map
        .values()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    shards.sort();
    shards.dedup();
    for shard in &shards {
        let mut reader = BufReader::new(File::open(src_real.join(shard))?);
        let mut len_buf = [0u8; 8];
        reader.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf);
        let mut header_bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut header_bytes)?;
        let parsed: Value = serde_json::from_slice(&header_bytes)?;
        let object = parsed
            .as_object()
            .ok_or_else(|| format!("shard {shard} header is not a JSON object"))?;
        let mut map = BTreeMap::new();
        for (name, entry) in object {
            if name == "__metadata__" {
                continue;
            }
            map.insert(name.clone(), entry.clone());
        }
        headers.insert(shard.clone(), map);
        payload_starts.insert(shard.clone(), 8 + header_len);
    }
    println!("read {} shard header(s)", headers.len());

    // Build one storage entry per module. A module is identified by its
    // `.trellis` tensor; the sign vectors and the MCG scalar are its siblings.
    let mut modules: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    let mut index_of: BTreeMap<String, String> = BTreeMap::new();
    for (name, shard) in weight_map {
        let shard = shard
            .as_str()
            .ok_or_else(|| format!("weight_map[{name}] is not a string"))?;
        index_of.insert(name.clone(), shard.to_string());
    }

    let mut mcg_mismatches: Vec<String> = Vec::new();
    for name in index_of.keys() {
        let Some(module) = name.strip_suffix(".trellis") else {
            continue;
        };
        let entry = modules.entry(module.to_string()).or_default();
        for suffix in COMPONENT_SUFFIXES {
            let component = format!("{module}{suffix}");
            let Some(shard_name) = index_of.get(&component) else {
                continue;
            };
            let tensor = headers
                .get(shard_name)
                .and_then(|h| h.get(&component))
                .ok_or_else(|| {
                    format!("index names {component} but {shard_name} does not contain it")
                })?;
            let shape = tensor
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{component} has no shape"))?;
            let offsets = tensor
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{component} has no data_offsets"))?;
            let n_bytes = offsets[1]
                .as_u64()
                .ok_or_else(|| format!("{component} has a non-numeric data_offsets[1]"))?
                - offsets[0]
                    .as_u64()
                    .ok_or_else(|| format!("{component} has a non-numeric data_offsets[0]"))?;
            let dtype = safetensors_dtype_to_torch(
                tensor
                    .get("dtype")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{component} has no dtype"))?,
            )?;
            entry.insert(
                component.clone(),
                json!({ "shape": shape, "n_bytes": n_bytes, "dtype": dtype }),
            );
        }
        // Cross-check the MCG multiplier against the engine's constant. This is
        // a READ of the 4-byte scalar, not a guess: the value is the codebook,
        // and a mismatch would decode to noise that every other check here
        // would pass.
        if codebook == "mcg" {
            let mcg_name = format!("{module}.mcg");
            let shard_name = index_of
                .get(&mcg_name)
                .ok_or_else(|| format!("{module} has no .mcg tensor"))?;
            let actual = read_mcg_multiplier(
                &src_real.join(shard_name),
                payload_starts[shard_name],
                headers
                    .get(shard_name)
                    .and_then(|h| h.get(&mcg_name))
                    .ok_or_else(|| format!("{mcg_name} missing from {shard_name}"))?,
            )?;
            if actual != MCG_MULTIPLIER {
                mcg_mismatches.push(format!(
                    "{mcg_name} = {actual} (0x{actual:08X}), engine expects \
                     {MCG_MULTIPLIER} (0x{MCG_MULTIPLIER:08X})"
                ));
            }
        }
    }
    if !mcg_mismatches.is_empty() {
        return Err(format!(
            "MCG multiplier mismatch in {} module(s); this is a different codebook and \
             the engine cannot decode it:\n  {}",
            mcg_mismatches.len(),
            mcg_mismatches
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n  ")
        )
        .into());
    }
    println!("built {} module storage entries", modules.len());

    // Assemble the output. `layout` is deliberately ABSENT: the engine sets
    // `permutation_mode = 1` when it sees `"colmajor"`, and the only repacker in
    // the tree (`exl3::reorder_tensor_to_colmajor`) hardcodes 48 u16 per tile,
    // which is 3 bpw. A 4 or 8 bpw tensor cannot be transposed by it, and
    // setting the flag would make the engine dispatch a colmajor kernel over
    // row-major weights -- the silent-gibberish failure recorded in
    // spec/decode-19tok-repacked-250gb.
    let mut tensor_storage = Map::new();
    for (module, entry) in &modules {
        let module_bits = if module == "lm_head" {
            head_bits as f64
        } else {
            bits
        };
        let mut storage = Map::new();
        storage.insert("stored_tensors".to_string(), Value::Object(entry.clone()));
        storage.insert("quant_format".to_string(), json!("exl3"));
        storage.insert("bits_per_weight".to_string(), json!(module_bits));
        if codebook == "mcg" {
            storage.insert("mcg_multiplier".to_string(), json!(MCG_MULTIPLIER));
        }
        tensor_storage.insert(module.clone(), Value::Object(storage));
    }
    if !modules.contains_key("lm_head") {
        return Err("no lm_head module found; this is not a text checkpoint".into());
    }

    let mut out = Map::new();
    for key in [
        "quant_method",
        "version",
        "bits",
        "head_bits",
        "calibration",
        "out_scales",
        "codebook",
    ] {
        let value = summary
            .get(key)
            .ok_or_else(|| format!("config.json quantization_config is missing {key}"))?;
        out.insert(key.to_string(), value.clone());
    }
    out.insert("tensor_storage".to_string(), Value::Object(tensor_storage));

    // Write the destination: links for the weights, real files for the JSON.
    fs::create_dir_all(dst)?;
    for shard in &shards {
        std::os::unix::fs::symlink(src_real.join(shard), dst.join(shard))?;
    }
    // An ALLOWLIST, not "every file that is not a shard". The source is often a
    // download folder: the first version of this tool copied all 262 non-shard
    // files beside the checkpoint, including a 14 GB .gguf, for a destination
    // that needs six small JSON files. Anything the loader does not read is
    // copied for no reason and can be arbitrarily large.
    for name in AUX_FILES {
        // A browser that de-duplicated a second download writes `name(1).ext`.
        // Accept that spelling for the tokenizer, which is the one file the
        // engine opens by exact name and the one a re-download renames.
        let source = if src_real.join(name).is_file() {
            src_real.join(name)
        } else if let Some(stem) = name.strip_suffix(".json")
            && let Some(numbered) = numbered_variant(&src_real, stem)
        {
            src_real.join(numbered)
        } else {
            println!("  (absent, skipped) {name}");
            continue;
        };
        if *name == "quantization_config.json" {
            // Synthesised below. A source copy would be the very file this tool
            // exists to replace, and leaving it in place on an early return would
            // make the destination look complete when it is not.
            continue;
        }
        fs::copy(&source, dst.join(name))?;
        println!("  copied {name}");
    }
    fs::write(
        dst.join("quantization_config.json"),
        serde_json::to_string_pretty(&Value::Object(out))?,
    )?;

    // Re-derive the total the engine will compute and compare it with the
    // index's `total_size`, so a header this tool mis-parsed is caught here
    // rather than at load time.
    let index_total = index
        .get("metadata")
        .and_then(|m| m.get("total_size"))
        .and_then(Value::as_u64);
    let mut all_bytes = 0u64;
    for header in headers.values() {
        for tensor in header.values() {
            let offsets = tensor["data_offsets"].as_array().expect("checked above");
            all_bytes += offsets[1].as_u64().expect("checked above")
                - offsets[0].as_u64().expect("checked above");
        }
    }
    if let Some(expected) = index_total
        && expected != all_bytes
    {
        return Err(format!(
            "index total_size {expected} does not match header tensor bytes {all_bytes}; \
             refusing to leave a half-consistent destination at {}",
            dst.display()
        )
        .into());
    }
    println!("header tensor bytes: {all_bytes} (index agrees)");
    println!(
        "\nWrote {}. {} shards are symlinked; no weight bytes were copied.",
        dst.join("quantization_config.json").display(),
        shards.len()
    );
    Ok(())
}

/// Read the 4-byte MCG multiplier of one module.
///
/// `data_offsets` are relative to the *payload*, which begins after the 8-byte
/// length prefix and the header JSON. Seeking to the raw offset reads the
/// header text instead and yields a plausible-looking u32 that is not the
/// multiplier — which is why the caller treats a mismatch as fatal rather than
/// defaulting to the engine's constant.
fn read_mcg_multiplier(
    shard_path: &Path,
    payload_start: u64,
    tensor: &Value,
) -> Result<u32, Box<dyn std::error::Error>> {
    let offsets = tensor["data_offsets"].as_array().expect("checked by caller");
    let start = offsets[0].as_u64().expect("checked by caller");
    let end = offsets[1].as_u64().expect("checked by caller");
    if end - start != 4 {
        return Err(format!(
            "MCG tensor in {} occupies {} bytes, expected 4",
            shard_path.display(),
            end - start
        )
        .into());
    }
    let mut file = File::open(shard_path)?;
    file.seek(SeekFrom::Start(payload_start + start))?;
    let mut raw = [0u8; 4];
    file.read_exact(&mut raw)?;
    Ok(u32::from_le_bytes(raw))
}

fn safetensors_dtype_to_torch(name: &str) -> Result<&'static str, Box<dyn std::error::Error>> {
    Ok(match name {
        "BOOL" => "torch.bool",
        "U8" => "torch.uint8",
        "I8" => "torch.int8",
        "U16" => "torch.uint16",
        "I16" => "torch.int16",
        "U32" => "torch.uint32",
        "I32" => "torch.int32",
        "U64" => "torch.uint64",
        "I64" => "torch.int64",
        "F16" => "torch.float16",
        "BF16" => "torch.bfloat16",
        "F32" => "torch.float32",
        "F64" => "torch.float64",
        other => return Err(format!("unsupported safetensors dtype {other}").into()),
    })
}
