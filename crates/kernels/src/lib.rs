pub mod attention;
pub mod affine;
pub mod elements;
pub mod encoder;
pub mod exl3;
pub mod linear;
pub mod pipeline_audit;
pub mod sweep;

/// Is the element kernel named `role` ablated out by `KAPLUMBA_ABLATE`?
///
/// `KAPLUMBA_ABLATE` is a comma-separated list of element-kernel role names.
/// Setting one makes the corresponding `encode_*` return `Ok(())` **without
/// encoding its dispatch**, so every downstream kernel reads stale scratch and
/// the generated text is garbage. It exists to answer exactly one question, by
/// production interleaved A/B rather than by the profiler: *what is the marginal
/// wall-clock cost of this kernel?*
///
/// The profiler cannot answer it. Every `profile.stamp` commits a command
/// buffer, so `own_ms` for a small kernel is dominated by launch overhead --
/// 546 boundaries per decode step -- and a sub-10 us kernel is unreadable in it.
/// A/B is the only instrument that resolves these kernels.
///
/// Roles: `input_transform`, `input_norm_qkv`, `post_attn_pre_ffn`,
/// `finalize_post_ffn`, `finalize_o_proj`, `kv_norm_rope`, `attention`,
/// `head_softcap`, `head_argmax`.
///
/// Measured marginal cost per dispatch (n=6 interleaved pairs, production):
///
/// | role | tok/s gain when ablated | ms/token | us/dispatch |
/// |---|---|---|---|
/// | `input_transform` (x180/token) | +0.031 | 0.12 | 0.7 |
/// | all five fused transform/finalize roles (x300/token) | +0.44 | 1.39 | 4.6 |
///
/// So a *dispatch* costs ~0.7 us and encoding fewer dispatches is not the win;
/// what costs is a dispatch that runs a **single threadgroup**, which cannot
/// reach more than one core's share of memory bandwidth. Never set this outside
/// a timing run.
pub fn ablated(role: &str) -> bool {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashSet::new()));
    let mut roles = cache.lock().unwrap_or_else(|error| error.into_inner());
    if roles.is_empty() {
        if let Ok(list) = std::env::var("KAPLUMBA_ABLATE") {
            roles.extend(
                list.split(',')
                    .map(|role| role.trim().to_string())
                    .filter(|role| !role.is_empty()),
            );
        }
        // An empty set is indistinguishable from "not yet populated", so an
        // empty `KAPLUMBA_ABLATE` re-reads the environment on every call. Only
        // someone who set the variable to nothing pays that, and they asked for
        // no ablation.
    }
    roles.contains(role)
}

pub const KERNELS_NAME: &str = "kaplumba-kernels";

#[cfg(test)]
mod tests {
    #[test]
    fn exposes_kernels_name() {
        assert_eq!(super::KERNELS_NAME, "kaplumba-kernels");
    }
}
