//! Dump every text tensor in the checkpoint: name, dtype, shape, bytes.
//!
//! Purpose is narrow: two questions the byte total cannot answer on its own.
//!
//! * **Q1 — conditional reads.** bytes/token assumes *every* text tensor is read
//!   on every decode step. If some large tensor is read only in prefill, or only
//!   for some layer kinds, bytes/token is overstated. This prints every tensor
//!   with its byte length sorted descending, so anything big enough to matter is
//!   visible by eye.
//! * **Q2 — dtype/param arithmetic.** bf16 tensors must satisfy
//!   `bytes == prod(shape) * 2`. One did not, and an unexplained residue in a
//!   bucket that feeds the load-bearing total is exactly the kind of thing that
//!   should not be waved through.
//!
//! Header-only, no GPU.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    match std::env::var_os("KAPLUMBA_PROBE_MODEL_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".kaplumba/models/Gxma/Gxma"),
    }
}

fn main() {
    let dir = model_dir();
    let idx_txt = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_str(&idx_txt).unwrap();
    let mut shards: Vec<String> = idx["weight_map"]
        .as_object()
        .unwrap()
        .values()
        .filter_map(|v| v.as_str())
        .map(|s| s.to_string())
        .collect();
    shards.sort();
    shards.dedup();

    let mut rows: Vec<(String, String, Vec<u64>, u64)> = Vec::new();
    for s in &shards {
        let mut f = File::open(dir.join(s)).unwrap();
        let mut lb = [0u8; 8];
        f.read_exact(&mut lb).unwrap();
        let n = u64::from_le_bytes(lb) as usize;
        let mut h = vec![0u8; n];
        f.read_exact(&mut h).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&h).unwrap();
        for (name, e) in v.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let shape: Vec<u64> = e["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect();
            let b = e["data_offsets"][1].as_u64().unwrap() - e["data_offsets"][0].as_u64().unwrap();
            rows.push((name.clone(), e["dtype"].as_str().unwrap().to_string(), shape, b));
        }
    }

    // ---- Q1: the biggest text tensors ------------------------------------
    let is_vision = |n: &str| n.starts_with("model.vision_tower.") || n.starts_with("model.embed_vision.");
    let text: Vec<_> = rows.iter().filter(|(n, ..)| !is_vision(n)).collect();
    let mut by_size: Vec<_> = text.clone();
    by_size.sort_by_key(|(_, _, _, b)| std::cmp::Reverse(*b));
    println!("=== 40 largest TEXT tensors (a conditionally-read tensor would show here) ===");
    println!("{:<62} {:>7} {:>22} {:>14}", "tensor", "dtype", "shape", "bytes");
    for (n, d, s, b) in by_size.iter().take(40) {
        let shp = s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x");
        println!("{:<62} {:>7} {:>22} {:>14}", n, d, shp, b);
    }

    // Group the whole text set by name-with-index-stripped, so the shape
    // variety inside a family is visible.
    let mut fam: BTreeMap<String, (usize, u64, Vec<String>)> = BTreeMap::new();
    for (n, d, s, b) in &text {
        let stripped: String = {
            let mut out = String::new();
            let mut prev_digit = false;
            for c in n.chars() {
                if c.is_ascii_digit() {
                    if !prev_digit {
                        out.push('N');
                    }
                    prev_digit = true;
                } else {
                    out.push(c);
                    prev_digit = false;
                }
            }
            out
        };
        let e = fam.entry(stripped).or_insert((0, 0, Vec::new()));
        e.0 += 1;
        e.1 += b;
        let shp = format!("{}:{:?}", d, s);
        if !e.2.contains(&shp) {
            e.2.push(shp);
        }
    }
    println!("\n=== text tensors by family (index-stripped) ===");
    println!("{:<62} {:>5} {:>16}  dtypes/shapes", "family", "count", "total bytes");
    let mut famv: Vec<_> = fam.iter().collect();
    famv.sort_by_key(|(_, (_, b, _))| std::cmp::Reverse(*b));
    for (name, (cnt, b, shapes)) in &famv {
        println!("{:<62} {:>5} {:>16}  {}", name, cnt, b, shapes.join(" "));
    }

    // ---- Q2: dtype arithmetic --------------------------------------------
    println!("\n=== Q2: dtype vs byte-length arithmetic (Q2 must be empty) ===");
    let mut bad = 0;
    for (n, d, s, b) in &text {
        let params: u64 = s.iter().product();
        let expect = match d.as_str() {
            "BF16" | "F16" => params * 2,
            "F32" | "I32" => params * 4,
            "I16" | "U16" => params * 2,
            "I8" | "U8" => params,
            _ => {
                println!("  UNKNOWN dtype {d} on {n}");
                bad += 1;
                continue;
            }
        };
        if expect != *b {
            println!("  MISMATCH {n} dtype={d} shape={s:?} params={params} bytes={b} expected={expect}");
            bad += 1;
        }
    }
    println!("  mismatches: {bad}");

    // ---- the bucket that did not add up, resolved explicitly --------------
    println!("\n=== bf16 norm/scalar accounting ===");
    let mut norms_params = 0u64;
    let mut norms_bytes = 0u64;
    for (n, d, s, b) in &text {
        let is_normish = n.contains("norm") || n.ends_with("layer_scalar");
        if is_normish && (d == "BF16" || d == "F16") {
            norms_params += s.iter().product::<u64>();
            norms_bytes += b;
            println!("  {n:<62} {:>6} {:>16} bytes={}", d, s.iter().product::<u64>(), b);
        }
    }
    println!(
        "  TOTAL params {norms_params} x 2 = {} vs bytes {norms_bytes}  delta {}",
        norms_params * 2,
        norms_bytes as i64 - (norms_params * 2) as i64
    );

    println!("\n=== totals ===");
    let text_bytes: u64 = text.iter().map(|(_, _, _, b)| b).sum();
    let vision_bytes: u64 = rows.iter().filter(|(n, ..)| is_vision(n)).map(|(_, _, _, b)| b).sum();
    println!("  text tensors {} bytes {} ({:.4} GB)", text.len(), text_bytes, text_bytes as f64 / 1e9);
    println!("  vision tensors {} bytes {} ({:.4} GB)", rows.len() - text.len(), vision_bytes, vision_bytes as f64 / 1e9);
}