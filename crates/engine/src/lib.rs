pub mod bench;
pub mod chat;
pub mod config;
pub mod dflash;
pub mod decode_profile;
pub mod drafter;
pub mod drafter_model;
pub mod exl3;
pub mod generate;
pub mod gpu_cache;
pub mod gpu_lock;
pub mod kv_cache;
pub mod metal;
pub mod metrics;
pub mod model;
pub mod safetensors;
pub mod sampler;
pub mod tokenizer;
pub mod weights;

pub const ENGINE_NAME: &str = "kaplumba-engine";

#[cfg(test)]
mod tests {
    #[test]
    fn exposes_engine_name() {
        assert_eq!(super::ENGINE_NAME, "kaplumba-engine");
    }
}
