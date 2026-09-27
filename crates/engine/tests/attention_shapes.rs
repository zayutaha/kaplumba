//! Attention parity at the real Gemma 4 shapes, which the small synthetic
//! fixture does not reach: 8:1 grouped-query, head_dim 256 and 512.
use half::f16;
use kaplumba_kernels::attention::{AttentionKernels, AttentionRequest};

fn vals(n: usize, seed: f32) -> Vec<u16> {
    (0..n).map(|i| f16::from_f32(((i as f32 + seed) * 0.037).sin()).to_bits()).collect()
}

fn reference(
    q: &[u16], k: &[u16], v: &[u16],
    qp: &[u32], kp: &[u32],
    qh: usize, kvh: usize, hd: usize, window: usize,
) -> Vec<f32> {
    let groups = qh / kvh;
    let queries = qp.len();
    let keys = kp.len();
    let mut out = vec![0.0f32; queries * qh * hd];
    for qi in 0..queries {
        for h in 0..qh {
            let kh = h / groups;
            let mut scores = Vec::new();
            for ki in 0..keys {
                if kp[ki] > qp[qi] { continue; }
                if window != 0 && (qp[qi] - kp[ki]) as usize >= window { continue; }
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += f16::from_bits(q[(qi * qh + h) * hd + d]).to_f32()
                        * f16::from_bits(k[(ki * kvh + kh) * hd + d]).to_f32();
                }
                scores.push((ki, s));
            }
            let max = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            let mut acc = vec![0.0f32; hd];
            let mut total = 0.0f32;
            for (ki, s) in &scores {
                let e = (*s - max).exp();
                total += e;
                for d in 0..hd {
                    acc[d] += e * f16::from_bits(v[(ki * kvh + kh) * hd + d]).to_f32();
                }
            }
            for d in 0..hd {
                out[(qi * qh + h) * hd + d] = acc[d] / total;
            }
        }
    }
    out
}

fn check(name: &str, qh: usize, kvh: usize, hd: usize, window: usize, tokens: usize) {
    let kernels = AttentionKernels::new().expect("kernels");
    let q = vals(tokens * qh * hd, 1.0);
    let k = vals(tokens * kvh * hd, 2.0);
    let v = vals(tokens * kvh * hd, 3.0);
    let qp: Vec<u32> = (0..tokens as u32).collect();
    let kp: Vec<u32> = (0..tokens as u32).collect();
    let got = kernels
        .online_attention(AttentionRequest {
            queries: &q, keys: &k, values: &v,
            query_positions: &qp, key_positions: &kp,
            q_heads: qh, kv_heads: kvh, head_dim: hd,
            scale: 1.0, window,
        })
        .expect("attention");
    let want = reference(&q, &k, &v, &qp, &kp, qh, kvh, hd, window);
    let mut worst = 0.0f32;
    for (i, w) in want.iter().enumerate() {
        worst = worst.max((f16::from_bits(got[i]).to_f32() - w).abs());
    }
    println!("{name:34} worst={worst:.6} {}", if worst < 2e-2 { "OK" } else { "MISMATCH" });
    assert!(worst < 2e-2, "{name} worst {worst}");
}

#[test]
fn attention_matches_reference_at_real_gemma4_shapes() {
    // full attention: 32 q heads, 4 kv heads (8:1), head_dim 512
    check("full q32 kv4 hd512", 32, 4, 512, 0, 6);
    // sliding attention: 32 q heads, 16 kv heads (2:1), head_dim 256
    check("sliding q32 kv16 hd256", 32, 16, 256, 1024, 6);
    // sliding with the window actually biting
    check("sliding q4 kv2 hd64 window3", 4, 2, 64, 3, 6);
}
