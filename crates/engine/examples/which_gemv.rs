//! Which decode kernel does the live path actually dispatch?
//!
//! `LinearKernels::encode_gemv` tries **ishizuki first, EXL3 second**, and the
//! switch between them is `packed_offset % 4` (plus `KAPLUMBA_OWN_GEMV`). That
//! ordering is easy to get wrong when reasoning about a measurement, and getting
//! it wrong is silent: the `_mcg` codebook specialisation -- the 8.1 -> 9.2 tok/s
//! step in `spec/decode-19tok` -- only pays if the EXL3 arm is the one running,
//! and a bandwidth number attributed to the wrong kernel is not wrong in any way
//! that announces itself.
//!
//! This prints the decision for every (rate, alignment) the engine can produce,
//! plus the name each choice resolves to. Run it before attributing any EXL3
//! number to a kernel.

use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::LinearKernels;

fn main() {
    let kernels = LinearKernels::shared().expect("kernels");
    println!(
        "KAPLUMBA_OWN_GEMV = {}",
        std::env::var_os("KAPLUMBA_OWN_GEMV").is_some()
    );
    println!(
        "KAPLUMBA_DECODE_NR0 = {}",
        std::env::var("KAPLUMBA_DECODE_NR0").unwrap_or_else(|_| "unset (=1)".into())
    );
    println!();
    println!(
        "{:<10}{:>14}{:>12}{:>12}{:>12}",
        "rate", "packed_offset", "ishizuki?", "exl3 table", "live decode"
    );
    for (bits, label) in [(3u8, "3 bpw"), (6, "6 bpw")] {
        for offset in [0u64, 2] {
            let rate = Exl3Rate::Integer(bits);
            let ish = kernels.ishizuki_gemv_kernel_name(rate, offset);
            let exl3 = kernels
                .exl3_gemv_kernel_name(rate, offset, Exl3Codebook::Mcg)
                .unwrap_or("none");
            let live = kernels
                .decode_gemv_kernel_name(rate, offset, Exl3Codebook::Mcg)
                .unwrap_or_else(|| "none".to_string());
            println!(
                "{label:<10}{offset:>14}{:>12}{:>12}{live:>12}",
                if ish.is_some() { "yes" } else { "no" },
                exl3
            );
        }
    }
    println!(
        "\n`ishizuki? = yes` means `encode_gemv` dispatches linear_gemv_ishizuki_kN and the \
         EXL3 arm never runs for that tensor. `live decode` is the kernel name the EXL3 arm \
         resolves to after KAPLUMBA_DECODE_NR0 is applied."
    );
}
