pub mod config;
pub mod exl3;
pub mod metal;
pub mod safetensors;
pub mod weights;

pub const ENGINE_NAME: &str = "kaplumba-engine";

#[cfg(test)]
mod tests {
    #[test]
    fn exposes_engine_name() {
        assert_eq!(super::ENGINE_NAME, "kaplumba-engine");
    }
}
