use kaplumba_engine::metal::MetalContext;
use kaplumba_engine::safetensors;
use kaplumba_engine::weights::TextWeights;
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let root = std::path::Path::new(&dir);
    let index = safetensors::read_index(&root.join("model.safetensors.index.json")).unwrap();
    let sel = safetensors::load_text_selection(root, index).unwrap();
    let ctx = MetalContext::new().unwrap();
    let w = TextWeights::load_dedicated(root, &sel, ctx, u64::MAX).unwrap();
    for n in ["model.language_model.embed_tokens.weight", "lm_head.trellis",
              "model.language_model.layers.0.self_attn.q_proj.trellis"] {
        match w.tensor(n) {
            Some(t) => println!("{n}\n   spans={} converted={} start={} end={} bytes={}",
                t.spans.len(), t.converted_from_bf16, t.start, t.end, t.end-t.start),
            None => println!("{n}: absent"),
        }
    }
    println!("resident_bytes = {:.2} GiB", w.resident_bytes() as f64 / 1073741824.0);
}
