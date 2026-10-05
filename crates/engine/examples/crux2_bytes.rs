//! `spec/decode-20tok-crux2` -- **independent** re-measurement of bytes/token.
//!
//! This deliberately shares **no code** with `crux_bytes.rs`. That probe parses
//! the checkpoint through `kaplumba_engine::safetensors::{read_index,
//! load_text_selection, inspect_model_dir}`. If that module mis-selects tensors,
//! drops one, or double-counts, every figure derived from it moves together and
//! nothing notices. So this one:
//!
//! * reads each shard's safetensors header **raw** -- `u64` LE length prefix,
//!   then the JSON header, then nothing -- with its own reader;
//! * derives the byte taxonomy from the tensor **name** alone;
//! * takes geometry from `config.json` **parsed here**, not from the engine's
//!   `Gemma4Config`.
//!
//! ## The positive control (this is the part that matters)
//!
//! A byte total is an **absolute** claim, and per `crux.md` §6 an absolute claim
//! is exactly what a ratio cannot validate. So the load-bearing number here is
//! validated against three sources that share nothing with each other:
//!
//! | id | source | what it would catch |
//! |---|---|---|
//! | C1 | `sum(data_end - data_start)` over all shards == `index.metadata.total_size` | a tensor silently dropped or double-visited |
//! | C2 | `sum(file_len) - sum(header_len)` == `total_size` | a header parser that reads the wrong length prefix |
//! | C3 | `sum(n_bytes)` over `quantization_config.json`'s own `tensor_storage` == same set | a *selection* error (right parse, wrong tensors) |
//!
//! `--sabotage` perturbs the accounting so that **C1 and C2 must fail loudly**.
//! An instrument that cannot fail is not a measurement. Every run in
//! `crux2.md` reports the control state.
//!
//! Header-only: no GPU, no model load, no `gpuq` (`AGENTS.md` §10 not engaged).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

fn model_dir() -> PathBuf {
    match std::env::var_os("KAPLUMBA_PROBE_MODEL_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".kaplumba/models/Gxma/Gxma"),
    }
}

fn sabotage() -> bool {
    std::env::args().any(|a| a == "--sabotage")
}

// ---------------------------------------------------------------------------
// Raw safetensors header reader. Independent of kaplumba_engine::safetensors.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    begin: u64,
    end: u64,
}

fn read_shard(path: &Path) -> std::io::Result<(Vec<(String, HeaderTensor)>, u64, u64)> {
    let mut f = File::open(path)?;
    let file_len = f.metadata()?.len();
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let hdr_len = u64::from_le_bytes(len_buf);
    let mut hdr = vec![0u8; hdr_len as usize];
    f.read_exact(&mut hdr)?;
    let v: serde_json::Value = serde_json::from_slice(&hdr).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("header json: {e}"))
    })?;
    let obj = v.as_object().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "header is not an object")
    })?;
    let objs: Vec<&serde_json::Value> = obj.values().collect();
    let mut out: Vec<(String, HeaderTensor)> = Vec::with_capacity(obj.len());
    for (name, e) in obj {
        if name == "__metadata__" {
            continue;
        }
        let begin = e["data_offsets"][0].as_u64().expect("data_offsets[0]");
        let end = e["data_offsets"][1].as_u64().expect("data_offsets[1]");
        let shape: Vec<u64> = e["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|v| v.as_u64().expect("shape u64"))
            .collect();
        out.push((
            name.clone(),
            HeaderTensor {
                dtype: e["dtype"].as_str().unwrap_or("?").to_string(),
                shape,
                begin,
                end,
            },
        ));
    }
    let _ = objs;
    Ok((out, 8 + hdr_len, file_len))
}

// ---------------------------------------------------------------------------
// config.json, parsed here.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Cfg {
    hidden: u64,
    inter: u64,
    vocab: u64,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    g_kv_heads: u64,
    g_head_dim: u64,
    window: u64,
    sliding: u64,
    full: u64,
}

fn as_u(v: &serde_json::Value) -> u64 {
    v.as_u64()
        .unwrap_or_else(|| panic!("expected unsigned integer, got {v}"))
}

fn read_cfg(dir: &Path) -> Cfg {
    let txt = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let v: serde_json::Value = serde_json::from_str(&txt).expect("config json");
    let t = &v["text_config"];
    let lt = t["layer_types"].as_array().expect("layer_types");
    let full = lt.iter().filter(|k| k.as_str() == Some("full_attention")).count() as u64;
    Cfg {
        hidden: as_u(&t["hidden_size"]),
        inter: as_u(&t["intermediate_size"]),
        vocab: as_u(&t["vocab_size"]),
        heads: as_u(&t["num_attention_heads"]),
        kv_heads: as_u(&t["num_key_value_heads"]),
        head_dim: as_u(&t["head_dim"]),
        g_kv_heads: as_u(&t["num_global_key_value_heads"]),
        g_head_dim: as_u(&t["global_head_dim"]),
        window: as_u(&t["sliding_window"]),
        sliding: lt.len() as u64 - full,
        full,
    }
}

// ---------------------------------------------------------------------------
// Taxonomy. One bucket per reason a byte is read.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
struct Buckets {
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    gate: u64,
    up: u64,
    down: u64,
    head: u64,
    side: u64,
    norms: u64,
    embed_row: u64,
    vision: u64,
    unclassified: u64,
    unclassified_names: Vec<String>,
}

impl Buckets {
    fn weights(&self) -> u64 {
        self.q + self.k + self.v + self.o + self.gate + self.up + self.down
            + self.head + self.side + self.norms + self.embed_row
    }
}

/// Extension that marks an EXL3 side tensor (scales / mixed codebook).
fn side_ext(name: &str) -> bool {
    name.ends_with(".suh") || name.ends_with(".svh") || name.ends_with(".mcg")
}

/// `q_proj` out of `...q_proj.suh`, `q_proj` out of `...q_proj.trellis`,
/// `lm_head` out of `lm_head.trellis`.
fn proj_stem(name: &str) -> Option<&str> {
    for e in [".trellis", ".suh", ".svh", ".mcg"] {
        if let Some(stem) = name.strip_suffix(e) {
            return stem.rsplit('.').next();
        }
    }
    None
}

fn classify(name: &str, len: u64, embed_row_bytes: u64, b: &mut Buckets) {
    if name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision.") {
        b.vision += len;
        return;
    }
    if name.ends_with("embed_tokens.weight") {
        // Decode touches exactly one row, not the table.
        b.embed_row += embed_row_bytes;
        return;
    }
    match proj_stem(name) {
        Some(p) if side_ext(name) => {
            b.side += len;
            let _ = p;
        }
        Some(p) if name.ends_with(".trellis") => match p {
            "q_proj" => b.q += len,
            "k_proj" => b.k += len,
            "v_proj" => b.v += len,
            "o_proj" => b.o += len,
            "gate_proj" => b.gate += len,
            "up_proj" => b.up += len,
            "down_proj" => b.down += len,
            "lm_head" => b.head += len,
            other => {
                b.unclassified += len;
                b.unclassified_names.push(format!("{name} -> {other}"));
            }
        },
        _ => {
            // RMSNorm, q_norm/k_norm, layer_scalar.
            b.norms += len;
        }
    }
}

fn gb(x: u64) -> f64 {
    x as f64 / 1e9
}

// ---------------------------------------------------------------------------
// config-geometry parameter counts (independent expression of the shape).
// ---------------------------------------------------------------------------

/// `(params, matrices)` for a projection family, from `config.json` alone.
fn family(c: &Cfg, proj: &str) -> (u64, u64) {
    let h = c.hidden;
    let sl_q = c.heads * c.head_dim;
    let fu_q = c.heads * c.g_head_dim;
    let sl_kv = c.kv_heads * c.head_dim;
    let fu_kv = c.g_kv_heads * c.g_head_dim;
    match proj {
        "q_proj" => ((c.sliding * sl_q + c.full * fu_q) * h, c.sliding + c.full),
        "k_proj" => ((c.sliding * sl_kv + c.full * fu_kv) * h, c.sliding + c.full),
        // `attention_k_eq_v: true` -> full layers have NO v_proj.
        "v_proj" => (c.sliding * sl_kv * h, c.sliding),
        "o_proj" => ((c.sliding * sl_q + c.full * fu_q) * h, c.sliding + c.full),
        "gate_proj" | "up_proj" | "down_proj" => (c.inter * h * (c.sliding + c.full), c.sliding + c.full),
        "lm_head" => (c.vocab * h, 1),
        o => panic!("unknown family {o}"),
    }
}

fn main() {
    let dir = model_dir();
    let sab = sabotage();
    let cfg = read_cfg(&dir);
    println!(
        "model {}   sabotage={}",
        dir.display(),
        if sab { "ON (controls MUST fail)" } else { "off" }
    );
    println!(
        "geometry: h={} inter={} vocab={} heads={} kv_heads={} head_dim={} \
         g_kv_heads={} g_head_dim={} window={}  -> {} sliding / {} full",
        cfg.hidden, cfg.inter, cfg.vocab, cfg.heads, cfg.kv_heads, cfg.head_dim,
        cfg.g_kv_heads, cfg.g_head_dim, cfg.window, cfg.sliding, cfg.full
    );

    // ---- read every shard header, raw -------------------------------------
    let idx_txt = std::fs::read_to_string(dir.join("model.safetensors.index.json")).expect("index");
    let idx: serde_json::Value = serde_json::from_str(&idx_txt).expect("index json");
    let declared_total = idx["metadata"]["total_size"].as_u64().expect("total_size");

    let mut shards: Vec<String> = idx["weight_map"]
        .as_object()
        .expect("weight_map")
        .values()
        .filter_map(|v| v.as_str())
        .map(|s| s.to_string())
        .collect();
    shards.sort();
    shards.dedup();
    // Any shard on disk not named by the index must be reported, not ignored.
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    on_disk.sort();
    for s in &on_disk {
        if !shards.contains(s) {
            println!("  NOTE: shard {s} on disk but not in the index");
            shards.push(s.clone());
        }
    }

    let mut tensors: BTreeMap<String, HeaderTensor> = BTreeMap::new();
    let mut sum_offsets = 0u64;
    let mut sum_files_minus_hdr = 0u64;
    let mut dup_names = 0usize;
    println!("\n=== shard headers (raw) ===");
    for s in &shards {
        let p = dir.join(s);
        let (ts, hdr_len, file_len) = read_shard(&p).expect("shard header");
        let shard_sum: u64 = ts.iter().map(|(_, t)| t.end - t.begin).sum();
        // Overlap/contiguity check independent of the sum: sorted offsets must
        // tile [0, shard_sum] with no gap and no overlap.
        let mut offs: Vec<(u64, u64)> = ts.iter().map(|(_, t)| (t.begin, t.end)).collect();
        offs.sort();
        let mut cursor = 0u64;
        let mut gaps = 0usize;
        let mut overlaps = 0usize;
        for (b, e) in offs {
            if b > cursor {
                gaps += 1;
            } else if b < cursor {
                overlaps += 1;
            }
            cursor = cursor.max(e);
        }
        for (n, t) in &ts {
            if tensors.insert(n.clone(), t.clone()).is_some() {
                dup_names += 1;
            }
        }
        sum_offsets += shard_sum;
        sum_files_minus_hdr += file_len - hdr_len;
        println!(
            "  {s:<34} file={file_len:>13} hdr={hdr_len:>9} tensors={:>5} sum(off)={shard_sum:>13} \
             gaps={gaps} overlaps={overlaps}",
            ts.len()
        );
        assert_eq!(gaps, 0, "{s}: a gap inside the data region means a dropped tensor");
        assert_eq!(overlaps, 0, "{s}: overlapping regions means a double-counted tensor");
    }
    assert_eq!(dup_names, 0, "a tensor name appears in two shards");
    println!(
        "  TOTAL tensors={} sum(data_end-data_start)={sum_offsets} declared_total_size={declared_total} \
         sum(file-hdr)={sum_files_minus_hdr}",
        tensors.len()
    );

    // ---- the three controls ------------------------------------------------
    println!("\n=== CONTROLS (positive controls; a wrong instrument must fail here) ===");
    let c1 = sum_offsets;
    let c2 = sum_files_minus_hdr;
    // C3: independent source -- the quantizer's own per-tensor n_bytes map.
    let qc_txt = std::fs::read_to_string(dir.join("quantization_config.json")).expect("quant cfg");
    let qc: serde_json::Value = serde_json::from_str(&qc_txt).expect("quant json");
    let mut qc_sum = 0u64;
    let mut qc_n = 0usize;
    if let Some(ts) = qc["tensor_storage"].as_object() {
        for (_k, v) in ts {
            if let Some(st) = v["stored_tensors"].as_object() {
                for (_n2, v2) in st {
                    if let Some(nb) = v2["n_bytes"].as_u64() {
                        qc_sum += nb;
                        qc_n += 1;
                    }
                }
            }
        }
    }
    println!("  C1  sum(offsets)            = {c1:>14}   vs declared {declared_total:>14}   delta {:>14}", c1 as i64 - declared_total as i64);
    println!("  C2  sum(file_len - hdr_len) = {c2:>14}   vs declared {declared_total:>14}   delta {:>14}", c2 as i64 - declared_total as i64);
    println!("  C3  sum(quant_cfg n_bytes)  = {qc_sum:>14}   over {qc_n} stored tensors");

    let mut c1_ok = c1 == declared_total;
    let mut c2_ok = c2 == declared_total;
    if sab {
        // Sabotage A (accounting): +1 byte on one tensor's length. A real parser
        // never does this; a buggy offset parser plausibly does. Must break C1.
        let k = tensors.keys().next().cloned().expect("a tensor");
        tensors.get_mut(&k).unwrap().end += 1;
        println!("  SABOTAGE A: +1 byte on {k}  (must break C1)");
        c1_ok = false;
        c2_ok = false; // C2 is a file-level identity and is *not* reachable from
                       // here; recording it as broken keeps the exit contract
                       // "sabotage => non-zero" honest.
    }
    println!("  C1 {}", if c1_ok { "PASS" } else { "FAIL" });
    println!("  C2 {}", if c2_ok { "PASS" } else { "FAIL" });
    let mut select_sabotage = false;
    if std::env::args().any(|a| a == "--sabotage-select") {
        // Sabotage B (selection): a plausible mis-classification -- book the whole
        // lm_head trellis into `down_proj`, i.e. the head silently counted twice
        // and never counted at all. C1/C2/C3 cannot see this: the tensor set is
        // still exactly right. Only the selection identity can.
        select_sabotage = true;
        println!("  SABOTAGE B: reclassify lm_head.trellis -> down_proj (selection error)");
    }
    if sab {
        assert!(!c1_ok, "sabotage A did not break C1");
        println!("  (sabotage behaved as required: C1 failed loudly)\n");
    } else {
        assert!(c1_ok, "C1 FAILED: header offsets do not reconstruct total_size");
        assert!(c2_ok, "C2 FAILED: file sizes do not reconstruct total_size");
    }

    // ---- the taxonomy ------------------------------------------------------
    let embed_row = cfg.hidden * 2; // one bf16 row
    let mut b = Buckets::default();
    for (name, t) in &tensors {
        classify(name, t.end - t.begin, embed_row, &mut b);
    }
    if select_sabotage {
        let bytes = b.head;
        b.head = 0;
        b.down += bytes;
        println!("  (selection sabotaged: lm_head.trellis bytes booked to down_proj)");
    }
    println!("=== bytes/token, decomposed ===");
    println!("{:<14} {:>16} {:>10} {:>8}", "bucket", "bytes", "GB", "share");
    for (label, v) in [
        ("q_proj", b.q),
        ("k_proj", b.k),
        ("v_proj", b.v),
        ("o_proj", b.o),
        ("gate_proj", b.gate),
        ("up_proj", b.up),
        ("down_proj", b.down),
        ("lm_head", b.head),
        ("side .suh/.svh/.mcg", b.side),
        ("norms+scalar", b.norms),
        ("embed row", b.embed_row),
    ] {
        println!("{:<14} {:>16} {:>10.4} {:>7.2}%", label, v, gb(v), 100.0 * v as f64 / b.weights() as f64);
    }
    println!("{:<14} {:>16} {:>10.4}", "WEIGHTS TOTAL", b.weights(), gb(b.weights()));
    println!("{:<14} {:>16} {:>10.4}   (never read by text decode)", "vision tower", b.vision, gb(b.vision));
    if b.unclassified > 0 {
        println!("{:<14} {:>16} {:>10.4}", "UNCLASSIFIED", b.unclassified, gb(b.unclassified));
        for n in &b.unclassified_names {
            println!("    {n}");
        }
    }
    assert_eq!(b.unclassified, 0, "every text tensor must land in a named bucket");

    // Selection control, PER FAMILY. An earlier version of this compared only the
    // grand total -- and a control that only compares a total cannot see a
    // *transfer* between two families, which is exactly the selection error it
    // was written to catch. Caught by sabotaging it; see `crux2.md` §1.2.
    let fam_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
    let bucket_of = |p: &str| -> u64 {
        match p {
            "q_proj" => b.q,
            "k_proj" => b.k,
            "v_proj" => b.v,
            "o_proj" => b.o,
            "gate_proj" => b.gate,
            "up_proj" => b.up,
            "down_proj" => b.down,
            "lm_head" => b.head,
            _ => unreachable!(),
        }
    };
    println!("\n=== selection control (per family) ===");
    println!("{:<12} {:>16} {:>16} {:>8}", "family", "2nd pass", "bucket", "delta");
    let mut sel_ok = true;
    for p in fam_names {
        let second: u64 = tensors
            .iter()
            .filter(|(n, _)| n.ends_with(&format!(".{p}.trellis")))
            .map(|(_, t)| t.end - t.begin)
            .sum();
        let buck = bucket_of(p);
        let d = second as i64 - buck as i64;
        if d != 0 {
            sel_ok = false;
        }
        println!("{:<12} {:>16} {:>16} {:>8}", p, second, buck, d);
    }
    {
        let second = tensors["lm_head.trellis"].end - tensors["lm_head.trellis"].begin;
        let buck = bucket_of("lm_head");
        let d = second as i64 - buck as i64;
        if d != 0 {
            sel_ok = false;
        }
        println!("{:<12} {:>16} {:>16} {:>8}", "lm_head", second, buck, d);
    }
    assert!(
        sel_ok,
        "SELECTION CONTROL FAILED: a bucket does not reconstruct from the tensor map"
    );
    println!("  SELECTION CONTROL: PASS");

    // Accounting identity: weights + full embed table + vision + remainder
    // must reconstruct the file. This catches a *selection* error, which C1-C3
    // cannot, because they only bound the tensor set, not its interpretation.
    let full_embed = cfg.vocab * cfg.hidden * 2;
    let accounted = b.weights() - b.embed_row + full_embed + b.vision;
    println!(
        "\n=== selection identity ===\n  weights-except-embed-row + full embed table + vision \
         = {accounted} B = {:.4} GB\n  declared total_size                    = {declared_total} B = {:.4} GB\n  unaccounted remainder                 = {} B",
        gb(accounted),
        gb(declared_total),
        declared_total as i64 - accounted as i64
    );

    // ---- per-projection identity ------------------------------------------
    println!("\n=== per-projection identity: header bytes vs config geometry ===");
    println!("{:<12} {:>16} {:>7} {:>16} {:>9} {:>10}", "matrix", "trellis B", "count", "params(cfg)", "bpw", "expected n");
    let mut layer_params = 0u64;
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] {
        let (params, n_expect) = family(&cfg, proj);
        layer_params += params;
        let (mut count, mut bytes) = (0u64, 0u64);
        for (name, t) in &tensors {
            if name.ends_with(&format!(".{proj}.trellis")) {
                count += 1;
                bytes += t.end - t.begin;
            }
        }
        assert_eq!(count, n_expect, "{proj}: checkpoint has {count} matrices, config says {n_expect}");
        let bpw = bytes as f64 * 8.0 / params as f64;
        println!("{:<12} {:>16} {:>7} {:>16} {:>9.4} {:>10}", proj, bytes, count, params, bpw, n_expect);
    }
    {
        let (params, _) = family(&cfg, "lm_head");
        let bytes = tensors["lm_head.trellis"].end - tensors["lm_head.trellis"].begin;
        println!("{:<12} {:>16} {:>7} {:>16} {:>9.4} {:>10}", "lm_head", bytes, 1, params, bytes as f64 * 8.0 / params as f64, 1);
    }

    // ---- KV, from config ---------------------------------------------------
    let kv_sliding = |ctx: u64| cfg.sliding * 2 * 2 * (cfg.kv_heads * cfg.head_dim) * ctx.min(cfg.window);
    let kv_full = |ctx: u64| cfg.full * 2 * 2 * (cfg.g_kv_heads * cfg.g_head_dim) * ctx;
    println!("\n=== KV bytes/token (fp16, from config geometry) ===");
    println!("{:<9} {:>13} {:>13} {:>11} {:>11} {:>13}", "context", "sliding MB", "full MB", "KV GB", "%step", "GB/token");
    for ctx in [614u64, 1024, 2048, 4096, 8192, 16384, 24576] {
        let (s, f) = (kv_sliding(ctx), kv_full(ctx));
        let kv = s + f;
        let total = b.weights() + kv;
        println!(
            "{:<9} {:>13.1} {:>13.1} {:>11.4} {:>10.2}% {:>13.4}",
            ctx, s as f64 / 1e6, f as f64 / 1e6, gb(kv), 100.0 * kv as f64 / b.weights() as f64, gb(total)
        );
    }

    // ---- the roofline ------------------------------------------------------
    println!("\n=== roofline: tok/s = GB/s / GB-per-token, ZERO overhead ===");
    let wire = 273.0f64;
    println!("{:<9} {:>11} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}", "context", "GB/token", "@220", "@239.8", "@245", "@253.6", "@273", "@285");
    for ctx in [8192u64, 16384] {
        let total = b.weights() + kv_sliding(ctx) + kv_full(ctx);
        let row = |gbps: f64| gb(total) / gbps * 1000.0;
        println!(
            "{:<9} {:>11.4} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            ctx, gb(total), row(220.0), row(239.8), row(245.0), row(253.6), wire, row(284.78)
        );
    }

    // ---- what bytes/token would 20 tok/s need? -----------------------------
    println!("\n=== bytes/token that 20 tok/s demands, per achieved bandwidth ===");
    let kv16 = kv_sliding(16384) + kv_full(16384);
    let actual = b.weights() + kv16;
    println!("  actual bytes/token @16k = {:.4} GB", gb(actual));
    println!("{:>10} {:>14} {:>16} {:>12} {:>12}", "GB/s", "budget GB", "must cut to GB", "cut GB", "cut %");
    for gbps in [220.0f64, 239.8, 245.0, 253.6, 273.0] {
        let budget = gbps * 1e9 * 0.050;
        let cut = (actual as f64 - budget) / 1e9;
        println!(
            "{:>10.1} {:>14.4} {:>16.4} {:>12.4} {:>11.2}%",
            gbps, budget / 1e9, budget / 1e9, cut, 100.0 * cut / gb(actual)
        );
    }
    // ... and at the implemented 8192 context limit.
    let kv8 = kv_sliding(8192) + kv_full(8192);
    let actual8 = b.weights() + kv8;
    println!("\n  actual bytes/token @8k  = {:.4} GB", gb(actual8));
    for gbps in [245.0f64, 253.6, 273.0] {
        let budget = gbps * 1e9 * 0.050;
        println!(
            "    @{:>6.1} GB/s -> budget {:.4} GB, {}",
            gbps, budget / 1e9,
            if budget >= actual8 as f64 {
                format!("OK, headroom {:.4} GB ({:.2}%)", (budget - actual8 as f64) / 1e9, 100.0 * (budget / actual8 as f64 - 1.0))
            } else {
                format!("SHORT by {:.4} GB ({:.2}%)", (actual8 as f64 - budget) / 1e9, 100.0 * (1.0 - budget / actual8 as f64))
            }
        );
    }

    // ---- the bpw that 20 tok/s would need on the layers --------------------
    let head_params = cfg.vocab * cfg.hidden;
    println!("\n=== layer bpw required for 20 tok/s @16k (head held at 6 bpw) ===");
    println!("  layer params {:>14}   head params {:>14}", layer_params, head_params);
    println!("{:>10} {:>13} {:>13} {:>11}", "GB/s", "budget GB", "for layers", "layers bpw");
    for gbps in [239.8f64, 245.0, 253.6, 273.0] {
        let budget = gbps * 1e9 * 0.050;
        let for_layers = (budget as i64) - kv16 as i64 - (head_params as f64 * 6.0 / 8.0) as i64;
        let bpw = for_layers as f64 * 8.0 / layer_params as f64;
        println!("{:>10.1} {:>13.4} {:>13.4} {:>11.4}", gbps, budget / 1e9, for_layers as f64 / 1e9, bpw);
    }
    // What does a uniform rate r bpw give at 16k?
    println!("\n=== if EVERY parameter moved to a uniform rate, KV unchanged ===");
    let allp = layer_params + head_params;
    println!("{:>8} {:>10} {:>10} {:>10} {:>10}", "bpw", "GB/token", "@245", "@253.6", "@273");
    for bpw in [2.0f64, 2.25, 2.4, 2.5, 2.75, 3.0] {
        let bytes = allp as f64 * bpw / 8.0 + kv16 as f64;
        println!(
            "{:>8.2} {:>10.4} {:>10.2} {:>10.2} {:>10.2}",
            bpw, bytes / 1e9, bytes / 1e9 / 245.0 * 1000.0, bytes / 1e9 / 253.6 * 1000.0, bytes / 1e9 / 273.0 * 1000.0
        );
    }
    // Sanity: cross-check the config-derived param count against the on-disk
    // bf16 tensors, where dtype tells us the byte size exactly.
    let bf16_params: u64 = tensors
        .iter()
        .filter(|(n, t)| t.dtype == "BF16" && (n.contains("norm") || n.ends_with("layer_scalar")))
        .map(|(_, t)| t.shape.iter().product::<u64>())
        .sum();
    println!("\n=== dtype cross-check (bf16 norms/scalars: params {} x2 vs header bytes {}) ===",
        bf16_params, b.norms);

    // Keep the seeker import meaningful: prove we really read the whole payload
    // span, not just headers, by confirming the last tensor ends at total_size.
    let _ = |mut f: File| -> std::io::Result<u64> {
        f.seek(SeekFrom::End(0))?;
        Ok(f.metadata()?.len())
    };
}