pub mod config;
pub mod safetensors;

pub const ENGINE_NAME: &str = "kaplumba-engine";

#[cfg(test)]
mod tests {
    #[test]
    fn exposes_engine_name() {
        assert_eq!(super::ENGINE_NAME, "kaplumba-engine");
    }
}
