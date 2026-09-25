pub mod exl3;
pub mod linear;

pub const KERNELS_NAME: &str = "kaplumba-kernels";

#[cfg(test)]
mod tests {
    #[test]
    fn exposes_kernels_name() {
        assert_eq!(super::KERNELS_NAME, "kaplumba-kernels");
    }
}
