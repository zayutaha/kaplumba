//! Scratch: dump per-layer trellis byte_offset % 4 (gated slow test below).

#[cfg(test)]
mod tests {
    use crate::metal_ctx::MetalContext;
    use crate::metal_linear::{MetalLinear, linear_handles};
    use crate::testutil::artifact_dir;

    #[test]
    fn dump_trellis_alignment_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let Ok(c) = MetalContext::new() else {
            eprintln!("skip: no metal");
            return;
        };
        let model = crate::load::load_exl3(&c, &dir).expect("load artifact");
        let _ml = MetalLinear::new(&c).expect("metal linear");
        let src: &dyn crate::gemma4::TensorSource = &model;
        let mut bad: Vec<String> = Vec::new();
        for i in 0..60usize {
            for m in ["self_attn.q_proj", "mlp.gate_proj"] {
                let name = format!("model.language_model.layers.{i}.{m}");
                match linear_handles(src, &name) {
                    Ok(h) => {
                        let a = h.trellis().byte_offset % 4;
                        eprintln!("L{i:02} {m:20} off%4={a}");
                        if a != 0 {
                            bad.push(name);
                        }
                    }
                    Err(e) => eprintln!("L{i:02} {m:20} err: {e}"),
                }
            }
        }
        match linear_handles(src, "lm_head") {
            Ok(h) => {
                let a = h.trellis().byte_offset % 4;
                eprintln!("lm_head off%4={a}");
                if a != 0 {
                    bad.push("lm_head".into());
                }
            }
            Err(e) => eprintln!("lm_head err: {e}"),
        }
        // The loader repacks at 64-byte alignment; any 2-mod-4 offset here
        // means a tensor slipped through onto the scalar RGT path (~5x slower
        // prefill). See load_exl3's repack block.
        assert!(
            bad.is_empty(),
            "{} tensors not 4-byte aligned (sg_eligible would fail): {bad:?}",
            bad.len()
        );
    }
}
