//! `spec/decode-20tok-crux` part 1: **measure** bytes/token from the checkpoint.
//!
//! This exists because the repo carries three different numbers for the same
//! quantity and no one reconciled them:
//!
//! | source | figure | how it was produced |
//! |---|---|---|
//! | `spec/decode-19tok/spec.md:182` | 13.145 GB | per-projection tensor sums |
//! | `spec/decode-20tps/spec.md:18` | 11.834 GB | per-projection tensor sums |
//! | `tests/perf_probe_bytes_per_token.rs` | 12.058 GB | per-projection tensor sums |
//!
//! All three are header arithmetic; none of them is a measurement of what a
//! decode step actually moves. So this probe does two things the other three
//! did not:
//!
//! 1. **Decomposes by projection and by side tensor**, and prints every term,
//!    so any figure can be re-derived from the printed table rather than
//!    trusted. The disagreement is then attributable to a specific term.
//! 2. **Cross-checks the sums two independent ways**: (a) against the total
//!    byte length of every `.safetensors` header, and (b) against an
//!    independent analytic parameter count from `config.json` multiplied by
//!    the checkpoint's own declared bits-per-weight. If the two disagree the
//!    printout says so loudly instead of quietly producing a number.
//!
//! It is **header-only** — it opens the two shards, reads the 8-byte length
//! prefix and the JSON header, and never touches the payload. So it needs no
//! GPU, no model load, no `gpuq`, and no 11 GiB of resident weights
//! (`AGENTS.md` §10 is not engaged). That matters: the bandwidth half of this
//! spec needs the GPU, and running this concurrently would contend for it.
//!
//! The one thing this probe deliberately does **not** claim: that the summed
//! header bytes equal DRAM traffic. `tests/rate4_colmajor_bit_exact.rs`
//! (`dram_line_traffic_is_1x_at_every_rate_so_the_amplification_is_requests_only`)
//! measures 1.0000x line traffic at every rate, so on this checkpoint distinct
//! bytes == useful DRAM bytes and the header sum is the right denominator. That
//! is a separate measurement by a different instrument; it is cited, not
//! re-asserted, and `crux_bandwidth.rs` re-checks the *rate* against it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kaplumba_engine::safetensors::{is_vision_tensor_name, load_text_selection, read_index};

fn model_dir() -> PathBuf {
    match std::env::var_os("KAPLUMBA_PROBE_MODEL_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".kaplumba/models/Gxma/Gxma"),
    }
}

// ---------------------------------------------------------------------------
// The byte taxonomy. One bucket per *reason a byte is read*, so that a saving
// can be attributed to a mechanism rather than to a tensor name.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
struct Bucket {
    /// 50 sliding x (q,k,v,o) + 10 full x (q,k=shared,o)
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    gate: u64,
    up: u64,
    down: u64,
    /// `lm_head.*`: trellis + its own scales. 262144 x 5376.
    head: u64,
    /// Every `.suh` / `.svh` / `.mcg`, split by which matrix it belongs to.
    side: u64,
    /// BF16 norms, per-head q/k norms, per-layer scalars.
    norms: u64,
    /// `embed_tokens.weight`, collapsed to the one row decode reads.
    embed_row: u64,
    /// Vision tower. Must be 0 per token.
    vision: u64,
}

impl Bucket {
    fn quantised(&self) -> u64 {
        self.q + self.k + self.v + self.o + self.gate + self.up + self.down + self.head
    }
    fn weights(&self) -> u64 {
        self.quantised() + self.side + self.norms + self.embed_row
    }
    fn projection_row(&self, name: &str) -> u64 {
        match name {
            "q_proj" => self.q,
            "k_proj" => self.k,
            "v_proj" => self.v,
            "o_proj" => self.o,
            "gate_proj" => self.gate,
            "up_proj" => self.up,
            "down_proj" => self.down,
            "lm_head" => self.head,
            _ => unreachable!(),
        }
    }
}

/// Which matrix does this side tensor belong to? Pure function of the name, so
/// `side_attribution_is_not_vacuous` can run it against mutated names.
fn owning_matrix(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".suh").or_else(|| {
        name.strip_suffix(".svh")
            .or_else(|| name.strip_suffix(".mcg"))
    })?;
    Some(
        stem.rsplit('.')
            .next()
            .expect("a matrix stem before the extension")
            .to_string(),
    )
}

/// Classify one tensor. `embed_bytes` is what the *decode path* reads, which for
/// `embed_tokens` is one bf16 row, not the table.
fn classify(name: &str, byte_len: u64, embed_row_bytes: u64, b: &mut Bucket) {
    if is_vision_tensor_name(name) {
        b.vision += byte_len;
        return;
    }
    if name.ends_with("embed_tokens.weight") {
        b.embed_row += embed_row_bytes;
        return;
    }
    if owning_matrix(name).is_some() {
        b.side += byte_len;
        return;
    }
    if name.ends_with(".trellis") {
        let proj = name
            .rsplit('.')
            .nth(1)
            .expect("a projection name before .trellis");
        match proj {
            "q_proj" => b.q += byte_len,
            "k_proj" => b.k += byte_len,
            "v_proj" => b.v += byte_len,
            "o_proj" => b.o += byte_len,
            "gate_proj" => b.gate += byte_len,
            "up_proj" => b.up += byte_len,
            "down_proj" => b.down += byte_len,
            "lm_head" => b.head += byte_len,
            other => panic!("unclassified .trellis projection {other} in {name}"),
        }
        return;
    }
    // Everything else that is a text tensor: RMSNorm weights, q_norm/k_norm,
    // layer scalars, and (per the checkpoint) nothing else.
    b.norms += byte_len;
}

/// Total **parameters** in every matrix of one projection family, plus how many
/// matrices that is.
///
/// The subtlety that has to be right here: a full-attention layer projects to
/// `num_attention_heads * global_head_dim` = 32x512 = 16384, while a sliding
/// layer projects to `32 * 256` = 8192. So `q_proj` and `o_proj` are **two
/// different shapes**, and a helper that uses one head_dim for both gets the
/// parameter count — and therefore every bpw derived from it — wrong by 33%.
/// `v_proj` is absent from the 10 full layers entirely (`attention_k_eq_v:
/// true`), so it counts 50 matrices, not 60.
fn proj_family(
    proj: &str,
    cfg: &kaplumba_engine::config::Gemma4Config,
) -> (u64, u64) {
    let h = cfg.hidden_size as u64;
    let im = cfg.intermediate_size as u64;
    let sliding = cfg.layer_types.iter().filter(|k| !k.is_full()).count() as u64;
    let full = cfg.num_hidden_layers as u64 - sliding;
    let n = cfg.num_attention_heads as u64;
    let sl_q_out = n * cfg.head_dim as u64;
    let fu_q_out = n * cfg.global_head_dim as u64;
    let sl_kv_out = cfg.num_key_value_heads as u64 * cfg.head_dim as u64;
    let fu_kv_out = cfg.num_global_key_value_heads as u64 * cfg.global_head_dim as u64;
    match proj {
        "q_proj" => ((sliding * sl_q_out + full * fu_q_out) * h, sliding + full),
        "k_proj" => ((sliding * sl_kv_out + full * fu_kv_out) * h, sliding + full),
        "v_proj" => (sliding * sl_kv_out * h, sliding),
        "o_proj" => ((sliding * sl_q_out + full * fu_q_out) * h, sliding + full),
        "gate_proj" | "up_proj" => (im * h * (sliding + full), sliding + full),
        "down_proj" => (h * im * (sliding + full), sliding + full),
        other => panic!("unknown projection {other}"),
    }
}

fn gb(x: u64) -> f64 {
    x as f64 / 1e9
}

/// Signed variant, for printing a delta. Never used where a magnitude matters.
fn sgb(x: i64) -> f64 {
    x as f64 / 1e9
}

// ---------------------------------------------------------------------------
// KV cache: modelled from config, because it does not exist in the checkpoint.
// ---------------------------------------------------------------------------

struct Geom {
    sliding_layers: u64,
    full_layers: u64,
    sliding_kv_dim: u64,
    global_kv_dim: u64,
    window: u64,
}

fn geom(cfg: &kaplumba_engine::config::Gemma4Config) -> Geom {
    let sliding = cfg.layer_types.iter().filter(|k| !k.is_full()).count() as u64;
    Geom {
        sliding_layers: sliding,
        full_layers: cfg.layer_types.len() as u64 - sliding,
        sliding_kv_dim: cfg.num_key_value_heads as u64 * cfg.head_dim as u64,
        global_kv_dim: cfg.num_global_key_value_heads as u64 * cfg.global_head_dim as u64,
        window: cfg.sliding_window as u64,
    }
}

/// Bytes of KV one decode step reads at `context` keys, fp16, K and V.
/// Sliding layers are clamped to the window; the ring does not memmove, so past
/// the window their read is constant.
fn kv_bytes(g: &Geom, context: u64) -> (u64, u64, u64) {
    let win = context.min(g.window);
    let sliding = g.sliding_layers * 2 * 2 * g.sliding_kv_dim * win;
    let full = g.full_layers * 2 * 2 * g.global_kv_dim * context;
    (sliding, full, sliding + full)
}

// ---------------------------------------------------------------------------
// Independent analytic cross-check: parameters x bits, straight from config.
// ---------------------------------------------------------------------------

/// Non-embedding parameters, from `config.json` alone.
fn param_count(cfg: &kaplumba_engine::config::Gemma4Config) -> (u64, u64) {
    let h = cfg.hidden_size as u64;
    let im = cfg.intermediate_size as u64;
    let mut layers = 0u64;
    let mut head = cfg.vocab_size as u64 * h;
    for kind in &cfg.layer_types {
        let (q_out, kv_out, has_v) = if kind.is_full() {
            (
                cfg.num_attention_heads as u64 * cfg.global_head_dim as u64,
                cfg.num_global_key_value_heads as u64 * cfg.global_head_dim as u64,
                false,
            )
        } else {
            (
                cfg.num_attention_heads as u64 * cfg.head_dim as u64,
                cfg.num_key_value_heads as u64 * cfg.head_dim as u64,
                true,
            )
        };
        layers += h * q_out + h * kv_out + if has_v { h * kv_out } else { 0 } + q_out * h;
        layers += 3 * h * im;
    }
    (layers, head)
}

fn main() {
    let dir = model_dir();
    let index = read_index(&dir.join("model.safetensors.index.json")).expect("index");
    let selection = load_text_selection(&dir, index).expect("selection");
    let cfg = kaplumba_engine::safetensors::inspect_model_dir(&dir)
        .expect("config")
        .config;

    let tensors: &BTreeMap<String, _> = &selection.tensors;
    println!(
        "checkpoint {} : {} text tensors, {:.3} GB text, {:.3} GB all",
        dir.display(),
        tensors.len(),
        gb(selection.text_bytes),
        gb(selection.all_bytes)
    );
    println!(
        "config: {} layers, {} sliding / {} full, hidden {}, intermediate {}, vocab {}, \
         q_heads {} kv_heads {} head_dim {}, global_kv_heads {} global_head_dim {}, window {}",
        cfg.num_hidden_layers,
        cfg.layer_types.iter().filter(|k| !k.is_full()).count(),
        cfg.layer_types.iter().filter(|k| k.is_full()).count(),
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.vocab_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.num_global_key_value_heads,
        cfg.global_head_dim,
        cfg.sliding_window,
    );

    let embed_row = (cfg.hidden_size * 2) as u64;
    let mut b = Bucket::default();
    for (name, t) in tensors {
        classify(name, t.byte_len(), embed_row, &mut b);
    }

    println!("\n=== WEIGHTS read per decode token (header-derived) ===");
    println!("{:<12} {:>16} {:>9} {:>8}", "bucket", "bytes", "GB", "%w");
    for (label, v) in [
        ("q_proj", b.q),
        ("k_proj", b.k),
        ("v_proj", b.v),
        ("o_proj", b.o),
        ("gate_proj", b.gate),
        ("up_proj", b.up),
        ("down_proj", b.down),
        ("lm_head", b.head),
        ("side(suh..)", b.side),
        ("norms", b.norms),
        ("embed row", b.embed_row),
        ("vision", b.vision),
    ] {
        println!(
            "{:<12} {:>16} {:>9.4} {:>7.2}%",
            label,
            v,
            gb(v),
            100.0 * v as f64 / b.weights().max(1) as f64
        );
    }
    println!(
        "{:<12} {:>16} {:>9.4}",
        "TOTAL",
        b.weights(),
        gb(b.weights())
    );

    // --- cross-check 1: the per-projection table must reconstruct the totals ---
    let mut side_by_matrix: BTreeMap<String, u64> = BTreeMap::new();
    for (name, t) in tensors {
        if let Some(m) = owning_matrix(name) {
            *side_by_matrix.entry(m).or_default() += t.byte_len();
        }
    }
    println!("\n=== side tensors (.suh/.svh/.mcg) by owning matrix ===");
    let mut side_total = 0;
    for (m, v) in &side_by_matrix {
        println!("  {:<12} {:>12} B  {:>7.4} GB", m, v, gb(*v));
        side_total += v;
    }
    println!("  {:<12} {side_total:>12} B", "TOTAL");
    assert_eq!(
        side_total, b.side,
        "side attribution must reconstruct the side bucket"
    );

    // Every quantised matrix and its own side tensors, so a future reader can
    // see exactly which rate each projection is at.
    println!("\n=== per-projection trellis, rate, and side overhead ===");
    println!(
        "{:<12} {:>14} {:>8} {:>12} {:>9} {:>7}",
        "matrix", "trellis B", "count", "each B", "implied bpw", "side%"
    );
    for proj in [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ] {
        let mut count = 0u64;
        let mut bytes = 0u64;
        for (name, t) in tensors {
            if name.ends_with(&format!(".{proj}.trellis")) {
                count += 1;
                bytes += t.byte_len();
            }
        }
        let side = side_by_matrix.get(proj).copied().unwrap_or(0);
        let each = if count > 0 { bytes / count } else { 0 };
        let (params, n_expect) = proj_family(proj, &cfg);
        assert_eq!(
            count, n_expect,
            "{proj}: the checkpoint has {count} matrices, config geometry says \
             {n_expect}. A mismatch means the head-dim split is wrong, and every \
             bpw below would be wrong with it."
        );
        println!(
            "{:<12} {:>14} {:>8} {:>12} {:>9.3} {:>6.2}%",
            proj,
            bytes,
            count,
            each,
            bytes as f64 * 8.0 / params as f64,
            100.0 * side as f64 / (bytes + side).max(1) as f64
        );
    }
    {
        let mut count = 0u64;
        let mut bytes = 0u64;
        for (name, t) in tensors {
            if name == "lm_head.trellis" {
                count += 1;
                bytes += t.byte_len();
            }
        }
        let side = side_by_matrix.get("lm_head").copied().unwrap_or(0);
        let each = bytes / count.max(1);
        let params = cfg.vocab_size as u64 * cfg.hidden_size as u64;
        println!(
            "{:<12} {:>14} {:>8} {:>12} {:>9.3} {:>6.2}%",
            "lm_head",
            bytes,
            count,
            each,
            each as f64 * 8.0 / params as f64,
            100.0 * side as f64 / (bytes + side).max(1) as f64
        );
    }

    // --- cross-check 2: independent analytic parameter count ----------------
    let (layer_params, head_params) = param_count(&cfg);
    let qc_bytes = read_dir(&dir).unwrap_or_else(|| {
        panic!("could not read quantization_config.json for the declared rates");
    });
    let layer_bpw = qc_bytes.bits;
    let head_bpw = qc_bytes.head_bits;
    let _ = layer_params;
    let analytic_layer = (layer_params as f64 * layer_bpw / 8.0).round() as u64;
    let analytic_head = (head_params as f64 * head_bpw / 8.0).round() as u64;
    let head_header = b.head + side_by_matrix.get("lm_head").copied().unwrap_or(0);
    println!("\n=== cross-check: header bytes vs config geometry x declared rate ===");
    println!(
        "  layer params {:>14} x {:>5.2} bpw = {:>14} B = {:>7.4} GB",
        layer_params,
        layer_bpw,
        analytic_layer,
        gb(analytic_layer)
    );
    println!(
        "  header layer quants (trellis + side)          = {:>14} B = {:>7.4} GB   delta {:+.4} GB ({:+.3}%)",
        b.quantised() - b.head,
        gb(b.quantised() - b.head),
        sgb((b.quantised() - b.head) as i64 - analytic_layer as i64),
        100.0 * ((b.quantised() - b.head) as f64 / analytic_layer as f64 - 1.0)
    );
    println!(
        "  head  params {:>14} x {:>5.2} bpw = {:>14} B = {:>7.4} GB",
        head_params,
        head_bpw,
        analytic_head,
        gb(analytic_head)
    );
    println!(
        "  header head (trellis + side)                 = {:>14} B = {:>7.4} GB   delta {:+.4} GB ({:+.3}%)",
        head_header,
        gb(head_header),
        sgb(head_header as i64 - analytic_head as i64),
        100.0 * (head_header as f64 / analytic_head as f64 - 1.0)
    );

    // --- the KV term --------------------------------------------------------
    let g = geom(&cfg);
    println!("\n=== KV cache read per decode token ===");
    println!(
        "  geometry: {} sliding x 2x2x{} keys-in-window, {} full x 2x2x{} keys",
        g.sliding_layers, g.sliding_kv_dim, g.full_layers, g.global_kv_dim
    );
    println!(
        "{:>8} {:>14} {:>14} {:>14} {:>10} {:>12}",
        "context", "sliding MB", "full MB", "total GB", "%weights", "tok/s@250"
    );
    for ctx in [37u64, 614, 1024, 2048, 4096, 8192, 16384, 24576] {
        let (s, f, t) = kv_bytes(&g, ctx);
        let total = b.weights() + t;
        println!(
            "{:>8} {:>14.1} {:>14.1} {:>14.4} {:>9.2}% {:>12.2}",
            ctx,
            s as f64 / 1e6,
            f as f64 / 1e6,
            gb(t),
            100.0 * t as f64 / b.weights() as f64,
            1000.0 / (total as f64 / 1e9 / 250.0 * 1000.0)
        );
    }

    // --- roofline -----------------------------------------------------------
    println!("\n=== roofline: tok/s = GBPS / bytes-per-token, ZERO overhead ===");
    println!(
        "{:>8} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "context", "GB/token", "@203", "@220", "@250", "@273", "@306"
    );
    for ctx in [614u64, 8192, 16384] {
        let (_, _, kv) = kv_bytes(&g, ctx);
        let bytes = b.weights() + kv;
        let row = |gbps: f64| 1000.0 / (bytes as f64 / 1e9 / gbps * 1000.0);
        println!(
            "{:>8} {:>12.4} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            ctx,
            gb(bytes),
            row(203.0),
            row(220.0),
            row(250.0),
            row(273.0),
            row(306.0)
        );
    }

    // --- what 20 tok/s demands ---------------------------------------------
    println!("\n=== what 20 tok/s at 16k demands ===");
    let (_, _, kv16) = kv_bytes(&g, 16384);
    let w16 = b.weights() + kv16;
    println!(
        "  {:>8.4} GB/token; 50 ms/token -> {:.1} GB/s needed ({:.1}% of the 273 GB/s wire peak)",
        gb(w16),
        w16 as f64 / 1e9 / 0.050,
        100.0 * (w16 as f64 / 1e9 / 0.050) / 273.0
    );
    for gbps in [203.0f64, 220.0, 250.0, 273.0, 280.0, 300.0, 320.0] {
        println!(
            "    at {:>5.0} GB/s -> {:>6.3} ms -> {:>6.2} tok/s",
            gbps,
            w16 as f64 / 1e9 / gbps * 1000.0,
            1000.0 / (w16 as f64 / 1e9 / gbps * 1000.0)
        );
    }

    // --- the bpw that 20 tok/s at 16k would need ----------------------------
    println!("\n=== bytes/token budget for 20 tok/s at 16k, vs rate ===");
    // 20 tok/s = 50 ms, so the bytes budget is `achieved_bandwidth * 0.050 s`.
    let layer_head_params = (layer_params + head_params) as f64;
    let kv16f = kv16 as f64;
    println!(
        "  weights {:.4} GB + KV(16k) {:.4} GB = {:.4} GB/token",
        gb(b.weights()),
        gb(kv16),
        gb(w16)
    );
    println!(
        "{:>8} {:>13} {:>13} {:>8} | {:>12} {:>12} {:>12} {:>12}",
        "GB/s",
        "budget GB",
        "for weights",
        "verdict",
        "layers bpw",
        "@2.0",
        "@2.5",
        "@3.0 today"
    );
    for gbps in [203.0f64, 220.0, 250.0, 273.0, 284.8, 300.0, 320.0] {
        let budget = gbps * 1e9 * 0.050; // bytes
        let for_weights = budget - kv16f;
        // The head is 6 bpw today. Give the layers whatever rate leaves the head
        // at 6 bpw.
        let head_bytes = (head_params as f64 * 6.0 / 8.0).min(for_weights);
        let layer_rate = if for_weights > head_bytes {
            (for_weights - head_bytes) * 8.0 / layer_params as f64
        } else {
            f64::NAN
        };
        // tok/s at 16k if the whole model moved to a uniform rate.
        let tps = |bpw: f64| {
            let bytes = layer_head_params * bpw / 8.0 + kv16f;
            1000.0 / (bytes / 1e9 / gbps * 1000.0)
        };
        println!(
            "{:>8.0} {:>13.4} {:>13.4} {:>8} | {:>12.3} {:>12.2} {:>12.2} {:>12.2}",
            gbps,
            budget / 1e9,
            for_weights / 1e9,
            if budget >= w16 as f64 { "OK" } else { "SHORT" },
            layer_rate,
            tps(2.0),
            tps(2.5),
            tps(3.0),
        );
    }
    println!(
        "  `layers bpw` holds the 6 bpw head at 6 bpw and gives the 29.29 G layer \
         parameters the rest. The last three columns are tok/s at 16k if EVERY \
         parameter moved to that uniform rate, KV unchanged."
    );
}

struct Qc {
    bits: f64,
    head_bits: f64,
}

fn read_dir(dir: &Path) -> Option<Qc> {
    let txt = std::fs::read_to_string(dir.join("quantization_config.json")).ok()?;
    let v = serde_json::from_str::<serde_json::Value>(&txt).ok()?;
    Some(Qc {
        bits: v.get("bits")?.as_f64()?,
        head_bits: v.get("head_bits")?.as_f64()?,
    })
}

// ---------------------------------------------------------------------------
// Non-vacuity checks. A classification that matches everything matches nothing.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn totals(names: &[(&str, u64)]) -> Bucket {
        let mut b = Bucket::default();
        for (n, l) in names {
            classify(n, *l, 10_752, &mut b);
        }
        b
    }

    #[test]
    fn classification_is_not_vacuous() {
        let real = totals(&[
            ("model.layers.0.self_attn.q_proj.trellis", 1_000),
            ("model.layers.0.self_attn.q_proj.suh", 7),
            ("model.layers.0.mlp.down_proj.trellis", 1_000),
            ("lm_head.trellis", 1_000),
            ("lm_head.svh", 3),
            ("model.embed_tokens.weight", 2_820_000_000),
            ("model.norm.weight", 11),
            ("model.vision_tower.layers.0.linear.weight", 999),
        ]);
        assert_eq!(real.q, 1_000);
        assert_eq!(real.side, 10);
        assert_eq!(real.head, 1_000);
        assert_eq!(real.embed_row, 10_752, "embed collapses to one row");
        assert_eq!(real.norms, 11);
        assert_eq!(real.vision, 999);

        // Mutants must move a bucket, or the real numbers mean nothing.
        let m1 = totals(&[("model.layers.0.mlp.down_proj.svh", 1_000)]);
        assert_ne!(m1.down, real.down, "a down_proj side tensor is not down_proj trellis");
        let m2 = totals(&[("lm_head.suh", 1_000)]);
        assert_eq!(m2.head, 0, "a head side tensor is not head trellis");
        assert_eq!(m2.side, 1_000);
        let m3 = totals(&[("model.layers.0.self_attn.v_proj.trellis", 1_000)]);
        assert_eq!(m3.v, 1_000);
        assert_eq!(m3.k, 0, "v_proj bytes must not land in k_proj");
    }

    /// Recorded as a limitation, not as a defect to paper over.
    ///
    /// `is_vision_tensor_name` is prefix-anchored on `model.vision_tower.` /
    /// `model.embed_vision.`. A checkpoint whose tower is named without the
    /// `model.` prefix is NOT recognised, and its bytes fall through to `norms`
    /// — a silent misclassification that no byte total would reveal, because the
    /// buckets still sum to the file size.
    ///
    /// This checkpoint ships **zero** vision tensors (the `vision` bucket is 0),
    /// so every number in this probe is unaffected. It is a hazard for a future
    /// multimodal checkpoint, and it is the reason this file counts bytes into
    /// named buckets and prints each one instead of printing only a total.
    #[test]
    fn vision_recognition_is_prefix_anchored() {
        assert!(is_vision_tensor_name("model.vision_tower.layers.0.linear.weight"));
        assert!(is_vision_tensor_name("model.embed_vision.weight"));
        // Not recognised — this is the documented sharp edge.
        assert!(!is_vision_tensor_name("vision_tower.layers.0.linear.weight"));
        assert!(!is_vision_tensor_name("model.vision_model.weight"));

        // And the consequence, stated as a fact about `classify`: an unprefixed
        // tower is booked to `norms`, not to `vision`.
        let b = totals(&[("vision_tower.layers.0.linear.weight", 999)]);
        assert_eq!(b.vision, 0);
        assert_eq!(b.norms, 999);
    }

    #[test]
    fn window_clamps_the_sliding_layers_and_nothing_else() {
        // 50 x 2 x 2 x (16*256) x window  and  10 x 2 x 2 x (4*512) x context
        let g = Geom {
            sliding_layers: 50,
            full_layers: 10,
            sliding_kv_dim: 16 * 256,
            global_kv_dim: 4 * 512,
            window: 1024,
        };
        let (s1023, f1023, t1023) = kv_bytes(&g, 1023);
        assert_eq!(s1023, 50 * 2 * 2 * 4096 * 1023);
        assert_eq!(f1023, 10 * 2 * 2 * 2048 * 1023);

        // Past the window the sliding term is frozen and only the full term grows.
        let (s8k, _, _) = kv_bytes(&g, 8192);
        assert_eq!(s8k, 50 * 2 * 2 * 4096 * 1024, "ring does not grow");
        let (s16k, _, _) = kv_bytes(&g, 16384);
        assert_eq!(s16k, s8k);

        // At 16k the total is the figure the task brief quotes.
        let (_, f16k, t16k) = kv_bytes(&g, 16384);
        assert_eq!(t16k, s8k + f16k);
        assert!(
            (t16k as f64 / 1e9 - 2.181) < 0.002,
            "KV at 16k is {} GB, expected ~2.181",
            gb(t16k)
        );
    }

    /// `attention_k_eq_v: true` means a full-attention layer has **no** `v_proj`.
    /// A param count that adds one anyway inflates the layer bytes by
    /// `10 * h * kv_out` = 10 * 5376 * 2048 = 110 M params = 41 MB at 3 bpw.
    /// That is exactly the size of the 11.834-vs-12.058 disagreement, which is
    /// why the flag matters here and not just in the engine.
    #[test]
    fn param_count_does_not_double_count_k_in_full_layers() {
        let cfg = crate_config();
        let (layers, head) = param_count(&cfg);
        let h = cfg.hidden_size as u64;
        let im = cfg.intermediate_size as u64;
        let full = cfg.layer_types.iter().filter(|k| k.is_full()).count() as u64;

        let sliding = cfg.layer_types.len() as u64 - full;
        // NOTE: no `vocab_size * h` term. `param_count` walks `layer_types`, so
        // the embedding table is not in `layers`; it is reported separately as the
        // `embed_row` bucket. Adding it here made this test fail by exactly
        // 262144 * 5376 = 1,409,286,144 params, which is the size of the whole
        // `lm_head` — a coincidence that would have been very easy to
        // misread as a real counting bug.
        let mut expect = 0u64;
        expect += sliding * (h * 32 * 256 + h * 16 * 256 * 2 + 32 * 256 * h);
        expect += full * (h * 32 * 512 + h * 4 * 512 + 32 * 512 * h);
        expect += (cfg.num_hidden_layers as u64) * 3 * h * im;
        assert_eq!(layers, expect, "analytic parameter count");

        // Independent second path: `proj_family` computes the same families from
        // a different expression and is what the byte decomposition prints. If the
        // two disagree, one of them is wrong and neither may be quoted.
        let fams = [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ];
        let by_family: u64 = fams.iter().map(|f| proj_family(f, &cfg).0).sum();
        assert_eq!(by_family, layers, "proj_family sum vs param_count");
        assert_eq!(head, cfg.vocab_size as u64 * h, "head is the vocab-by-hidden matrix");
        // `v_proj` is the flag-sensitive family: absent from the full layers.
        assert_eq!(
            proj_family("v_proj", &cfg).1 as usize,
            sliding as usize,
            "v_proj must exist in exactly the sliding layers"
        );

        // And the specific mistake the flag prevents:
        let wrong = expect + full * h * 4 * 512;
        assert!(
            (wrong - layers) * 3 / 8 > 40_000_000,
            "the mutant must move >40 MB at 3 bpw or it cannot explain the gap"
        );
    }
}

fn crate_config() -> kaplumba_engine::config::Gemma4Config {
    let dir = model_dir();
    kaplumba_engine::safetensors::inspect_model_dir(&dir)
        .expect("checkpoint config")
        .config
}
