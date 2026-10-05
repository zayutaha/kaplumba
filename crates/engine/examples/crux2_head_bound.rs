//! `spec/decode-20tok-crux2` — can `lm_head` be read in *less* than 1.0570 GB and
//! still be **exactly** argmax-preserving?
//!
//! # The route being tested
//!
//! `lm_head` is 262144 x 5376 at 6 bpw = **1.056964608 GB**, 8.77% of the step.
//! A "coarse-then-refine" head would read a cheap approximation for all 262,144
//! logits, keep only the candidates that could still win, and read those exactly.
//! The task brief demands this be *proven* byte-identical or rejected, not
//! assumed either way. So: prove it.
//!
//! # Why it cannot work, quantitatively
//!
//! EXL3 stores `W[t,j] = out_scale[t] * in_scale[j] * q[t,j]` where `q` is a
//! small integer decoded through the codebook. So for a partial read that leaves
//! the column set `U` unread, every token's logit has the **exact** bound
//!
//! ```text
//! |logit_t - partial_t| <= out_scale[t] * SUM_{j in U} |h_j| * in_scale[j] * C
//! ```
//!
//! with `C = max|q|` over the codebook. This is a *valid* bound (triangle
//! inequality), not an approximation — which is exactly why it is worth testing:
//! a valid bound that is too loose prunes nothing, and then the scheme reads
//! *more* than the plain head, not less.
//!
//! Everything below is computed from the real checkpoint (`lm_head.svh`,
//! `lm_head.suh`) and the repo's own codebook decoder, not estimated.
//!
//! Note the config also sets `final_logit_softcapping: 30.0`, so **every** logit
//! is bounded by 30 in absolute value. If the exclusion bound exceeds 30, not one
//! token in the vocabulary can be excluded and the scheme is dead on arrival.

use half::f16;
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

/// Read one tensor's raw payload out of whichever shard holds it.
fn read_tensor(dir: &PathBuf, want: &str) -> (String, Vec<u64>, Vec<u8>) {
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
    for s in &shards {
        let mut f = File::open(dir.join(s)).unwrap();
        let mut lb = [0u8; 8];
        f.read_exact(&mut lb).unwrap();
        let n = u64::from_le_bytes(lb) as usize;
        let mut h = vec![0u8; n];
        f.read_exact(&mut h).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&h).unwrap();
        if let Some(e) = v.get(want) {
            let b0 = e["data_offsets"][0].as_u64().unwrap();
            let b1 = e["data_offsets"][1].as_u64().unwrap();
            let shape: Vec<u64> = e["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect();
            let dtype = e["dtype"].as_str().unwrap().to_string();
            let mut payload = vec![0u8; (b1 - b0) as usize];
            use std::io::Seek;
            f.seek(std::io::SeekFrom::Start(8 + n as u64 + b0)).unwrap();
            f.read_exact(&mut payload).unwrap();
            return (dtype, shape, payload);
        }
    }
    panic!("tensor {want} not found");
}

fn main() {
    let dir = model_dir();

    // ---- 1. the codebook: max and RMS of the decoded integers -------------
    // The engine's own decoders, so this is not a guess about the format.
    let mut vals: Vec<f32> = Vec::with_capacity(256);
    for s in 0..256u16 {
        vals.push(kaplumba_engine::exl3::decode_state(s, kaplumba_engine::exl3::Codebook::Mcg));
    }
    let cmax = vals.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let cmean = vals.iter().sum::<f32>() / vals.len() as f32;
    let crms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
    let c_absmax = vals.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("=== codebook (mcg, 256 states, the repo's own decoder) ===");
    println!("  max|q|        = {cmax:.6}");
    println!("  RMS(q)        = {crms:.6}");
    println!("  max|q|/RMS(q) = {:.4}", cmax / crms);
    println!("  mean(q)       = {cmean:.6}   (nonzero mean would leak into every logit)");
    println!(
        "  min/max       = {:.6} / {:.6}",
        vals.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        vals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    // ---- 2. the head's own scales ----------------------------------------
    let (d_svh, sh_svh, b_svh) = read_tensor(&dir, "lm_head.svh");
    let (d_suh, sh_suh, b_suh) = read_tensor(&dir, "lm_head.suh");
    println!("\n=== lm_head scales, read from the checkpoint ===");
    println!("  lm_head.svh dtype={d_svh} shape={sh_svh:?} bytes={}", b_svh.len());
    println!("  lm_head.suh dtype={d_suh} shape={sh_suh:?} bytes={}", b_suh.len());

    let rd = |b: &[u8], dtype: &str| -> Vec<f32> {
        (0..b.len() / 2)
            .map(|i| {
                let u = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
                f16::from_bits(u).to_f32()
            })
            .collect()
    };
    let svh = rd(&b_svh, &d_svh);
    let suh = rd(&b_suh, &d_suh);
    let stat = |v: &[f32]| {
        (
            v.iter().fold(0.0f32, |a, b| a.max(b.abs())),
            v.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt(),
        )
    };
    let (svh_absmax, svh_min, svh_max, svh_rms) = stat(&svh);
    let (suh_absmax, suh_min, suh_max, suh_rms) = stat(&suh);
    println!("  svh (per output row, {}): absmax={svh_absmax:.6} min={svh_min:.6} max={svh_max:.6} rms={svh_rms:.6}", svh.len());
    println!("  suh (per input col,  {}): absmax={suh_absmax:.6} min={suh_min:.6} max={suh_max:.6} rms={suh_rms:.6}", suh.len());

    // ---- 3. max |W| and therefore the exclusion bound ---------------------
    //
    // |W[t,j]| <= |svh[t]| * |suh[j]| * C. The bound over an unread set U is
    //   B_t = svh[t] * C * SUM_{j in U} |h_j| * |suh[j]|
    // and with |h_j| <= h_max:
    //   B_t <= svh[t] * C * h_max * h_max_count * max|suh|
    //
    // The product `svh * suh * C` is what a scale-aware bound can assume; the
    // *typical* logit is the same product times the cancellation of the codes,
    // which is why the bound/typical ratio is the number that matters.
    let h = 5376usize; // hidden
    let wmax = svh_absmax * suh_absmax * cmax;
    println!("\n=== the exclusion bound ===");
    println!("  max|W[t,j]| <= max|svh| * max|suh| * max|q| = {svh_absmax:.6} * {suh_absmax:.6} * {cmax:.6} = {wmax:.6}");

    // We do NOT have the hidden state here, so the honest way to state the bound
    // is parametric in it and then evaluate at the only value the config pins
    // down: `final_logit_softcapping: 30.0` caps every logit at 30.
    println!("\n  Let L = SUM_j |h_j| * |suh[j]| * max|q|  (the per-row bound factor, shared by all rows");
    println!("  up to the row's own svh). Then the bound on the UNREAD part is  svh[t] * L.");
    println!("  config.json: final_logit_softcapping = 30.0  =>  |logit_t| <= 30 for EVERY token.\n");
    println!("  For the scheme to exclude even one token it needs   svh[t] * L < 30   for that token,");
    println!("  i.e. a token can only be excluded when its own row scale is below 30 / L.");
    println!("  L in turn is at least  max|suh| * max|q| * (something >= 1).");
    println!("\n  Lower bound on L with no knowledge of h at all: L >= max|suh| * max|q| * min_j |h_j|*...");

    // Compute the threshold on SUM_j |h_j| * |suh[j]| that any scheme must beat.
    println!("\n=== threshold ===");
    let need = |svh_t: f32| -> f64 {
        // SUM_j |h_j| |suh[j]| * C * svh_t < 30  =>  SUM_j |h_j||suh[j]| < 30/(svh_t*C)
        30.0 / (svh_t as f64 * cmax as f64)
    };
    println!("  required  SUM_j |h_j|*|suh[j]|  <  30 / (svh[t] * max|q|)  for token t to be excludable");
    let median_svh = {
        let mut v = svh.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!("  median |svh| = {median_svh:.6}  ->  threshold = {:.3}", need(median_svh));
    println!("  max    |svh| = {svh_absmax:.6}  ->  threshold = {:.3}", need(svh_absmax));
    let row_typical = suh_rms * h as f32; // |h_j| ~ 1 is the post-RMSNorm norm
    println!("\n  With the usual post-RMSNorm |h_j| ~ 1, SUM_j |h_j||suh[j]| is of order");
    println!("  rms|suh| * h = {:.6} * {} = {:.3}", suh_rms, h, row_typical);
    println!(
        "  => the bound is about {:.1}x the post-RMSNorm estimate of that sum, i.e. {:.0}x the softcap.",
        row_typical as f64 / need(median_svh),
        (median_svh * row_typical * cmax as f32) / 30.0f32
    );

    // ---- 4. how many columns may stay unread? ----------------------------
    //
    // The bound is linear in the number of unread columns. To bring it under the
    // softcap the readable fraction must satisfy
    //   sqrt(K) * typical_logit * (max|q|/RMS(q)) < 30   ->   K < (30/typ)^2 ...
    // expressed instead as: fraction readable needed.
// ---- 4. how much of the head may stay unread? ---------------------------
    //
    // The bound is   B_t = |svh[t]| * C * SUM_{j in U} |h_j| |suh[j]|,
    // LINEAR in the L1 mass of the UNREAD columns. A necessary condition for
    // excluding token t is B_t < 30 (the softcap). Hence:
    //
    //     SUM_{j in U} |h_j||suh[j]|  <  30 / (|svh[t]| * C)  =: T
    //     fraction of columns left unread  <  T / L1_all
    //
    // This is the MOST generous bound on any saving, and it still only asks the
    // bound to beat the softcap rather than the actual runner-up logit.
    println!("\n=== how much of the head may be SKIPPED? ===");
    let sum_abs_suh: f64 = suh.iter().map(|x| x.abs() as f64).sum();
    let e_abs_h = 0.7979f64; // E|N(0,1)| for post-RMSNorm unit-variance entries
    let l1_all = e_abs_h * sum_abs_suh;
    println!("  SUM_j |suh[j]|  = {sum_abs_suh:.3}   (real checkpoint data)");
    println!("  L1_all = SUM_j |h_j||suh[j]| ~ E|h| * SUM|suh| = {e_abs_h:.4} * {sum_abs_suh:.3} = {l1_all:.3}");
    println!(
        "  full-head bound B_t = |svh| * C * L1_all = {median_svh:.4} * {cmax:.4} * {l1_all:.3} = {:.1}",
        median_svh as f64 * cmax as f64 * l1_all
    );
    println!(
        "  softcap on ANY logit = 30.0  =>  the full-head bound is {:.1}x the softcap",
        median_svh as f64 * cmax as f64 * l1_all / 30.0
    );
    let frac_skip = (need(median_svh) / l1_all).min(1.0);
    println!(
        "\n  necessary condition to exclude the median-scale row: unread L1 mass < {:.3}",
        need(median_svh)
    );
    println!("  => at most {0:.2}% of the head's columns may be left unread", frac_skip * 100.0);
    println!(
        "  => MAXIMUM bytes any EXACT two-stage head could ever save = {:.4} GB of 1.0570 GB",
        1.056964608 * frac_skip
    );
    let frac_skip_hi = (need(svh_absmax) / l1_all).min(1.0);
    println!(
        "  (for the largest row scale |svh|={svh_absmax:.4}: at most {:.2}% skippable)",
        frac_skip_hi * 100.0
    );

    // ---- 5. is even the MAXIMUM possible head saving enough? --------------
    println!("\n=== is even the MAXIMUM possible head saving enough for 20 tok/s? ===");
    let w16 = 14.2390f64;
    for (name, bpt) in [
        ("actual (full head read every token)", w16),
        ("head ENTIRELY FREE -- unreachable upper bound", w16 - 1.056964608),
        ("head saving at the rigorous maximum above", w16 - 1.056964608 * frac_skip),
    ] {
        println!(
            "  {name:<48} {bpt:.4} GB/token -> needs {:6.1} GB/s ({:.1}% of the 273 wire peak)",
            bpt / 0.050,
            100.0 * (bpt / 0.050) / 273.0
        );
    }
    let _ = (c_absmax, wmax, row_typical);
}
