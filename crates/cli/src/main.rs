//! kaplumba-engine CLI (T1.0: metal smoke; generate lands in T1.4).

use kaplumba_engine::{clear_buffer_u32, MetalContext};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("metal-smoke") => metal_smoke(),
        Some("--help") | Some("-h") | None => {
            eprintln!(
                "usage: kaplumba-engine <command>\n\
                 \n\
                 commands:\n\
                 metal-smoke    GPU fill-buffer smoke test (T1.0)\n\
                 generate       (T1.4, not yet)\n"
            );
            std::process::exit(2);
        }
        Some(other) => {
            eprintln!("unknown command: {other} (try --help)");
            std::process::exit(2);
        }
    }
}

fn metal_smoke() {
    match MetalContext::new() {
        Ok(ctx) => {
            let n = 1 << 16;
            match clear_buffer_u32(&ctx, n, 0xA5A5_A5A5) {
                Ok(v) if v.iter().all(|&x| x == 0xA5A5_A5A5) => {
                    println!("metal-smoke: OK  elems={n} pattern=0xA5A5A5A5");
                }
                Ok(_) => {
                    eprintln!("metal-smoke: FAIL buffer not fully filled");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("metal-smoke: FAIL fill error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("metal-smoke: FAIL context: {e}");
            std::process::exit(1);
        }
    }
}
