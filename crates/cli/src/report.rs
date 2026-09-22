//! Host/memory metrics and the `kaplumba-bench/1` report (R7.4).
//!
//! The report mirrors `scripts/bench.py`'s schema so runs are diffable across
//! engines (R7.3/R7.4). Keys the naive path cannot measure yet are reported as
//! `null` with `mem_note` explaining which task lands them — never fabricated.

use kaplumba_engine::{GenStats, MetalContext};
use serde_json::{Value, json};

pub const SCHEMA: &str = "kaplumba-bench/1";

/// Required report keys — asserted present on every report (`debug_assert!` in
/// [`build_report`]) so the two engines stay diffable (R7.4). This is the Python
/// harness's key set plus `sampling`; `tokens`/`text` are further Rust-only extras.
pub const REPORT_KEYS: [&str; 17] = [
    "schema",
    "backend",
    "model",
    "temp",
    "sampling",
    "pp_sizes",
    "roofline",
    "tg_pct_of_roofline",
    "host",
    "versions",
    "load_s",
    "ttft",
    "prefill",
    "decode",
    "mem",
    "engine",
    "mtp",
];

pub fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

pub fn tg_tps_ceiling(weights_gb: f64, bandwidth_gbps: f64) -> f64 {
    bandwidth_gbps / weights_gb
}

// ---------------------------------------------------------------------------
// host / memory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HostInfo {
    pub platform: String,
    pub cpu: String,
    pub mem_bytes: u64,
    pub cpu_count: usize,
}

fn sysctl_raw(name: &str, len: &mut usize) -> Option<Vec<u8>> {
    let cname = std::ffi::CString::new(name).ok()?;
    unsafe {
        if libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        if *len == 0 {
            return None;
        }
        let mut buf = vec![0u8; *len];
        if libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        buf.truncate(*len);
        Some(buf)
    }
}

fn sysctl_string(name: &str) -> Option<String> {
    let mut len = 0usize;
    let mut buf = sysctl_raw(name, &mut len)?;
    while buf.last() == Some(&0) {
        buf.pop();
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

fn sysctl_u64(name: &str) -> Option<u64> {
    let mut len = 0usize;
    let buf = sysctl_raw(name, &mut len)?;
    if buf.len() < 8 {
        return None;
    }
    Some(u64::from_ne_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]))
}

fn uname_string() -> Option<String> {
    unsafe {
        let mut u: libc::utsname = std::mem::zeroed();
        if libc::uname(&mut u) != 0 {
            return None;
        }
        let cstr = |c: &[libc::c_char]| -> String {
            let bytes: Vec<u8> = c
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };
        Some(format!("{} {}", cstr(&u.sysname), cstr(&u.release)))
    }
}

pub fn host_info() -> HostInfo {
    HostInfo {
        platform: format!(
            "{}-{} {}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            uname_string().unwrap_or_default()
        )
        .trim_end()
        .to_string(),
        cpu: sysctl_string("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into()),
        mem_bytes: sysctl_u64("hw.memsize").unwrap_or(0),
        cpu_count: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
    }
}

/// Peak resident set size of this process (bytes). `getrusage` reports bytes on
/// macOS; 0 if unavailable.
pub fn peak_rss_bytes() -> u64 {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            ru.ru_maxrss.max(0) as u64
        } else {
            0
        }
    }
}

/// `(current MTLDevice allocation, recommended max working set)` in bytes.
pub fn metal_bytes(ctx: &MetalContext) -> (u64, u64) {
    let d = ctx.device();
    (
        d.current_allocated_size() as u64,
        d.recommended_max_working_set_size(),
    )
}

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

use crate::args::GenerateArgs;

pub struct RunOutcome<'a> {
    pub load_s: f64,
    pub wall_ms: f64,
    pub stats: &'a GenStats,
    pub text: &'a str,
    pub backend: &'a str,
    pub n_layers: usize,
    pub weights_bytes: u64,
    pub peak_rss_bytes: u64,
    pub metal_alloc_bytes: u64,
    pub metal_working_set_bytes: u64,
}

pub fn build_report(args: &GenerateArgs, run: &RunOutcome<'_>) -> Value {
    let report = build_report_inner(args, run);
    debug_assert!(
        {
            let obj = report.as_object().expect("report is an object");
            REPORT_KEYS.iter().all(|k| obj.contains_key(*k))
        },
        "report is missing a kaplumba-bench/1 key"
    );
    report
}

fn build_report_inner(args: &GenerateArgs, run: &RunOutcome<'_>) -> Value {
    let s = run.stats;
    let ceiling = tg_tps_ceiling(args.weights_gb, args.bandwidth_gbps);
    let tg = s.tg_tps();
    let host = host_info();
    let model = args.model.display().to_string();

    json!({
        "schema": SCHEMA,
        "backend": run.backend,
        "model": model,
        // f32 args are rounded so `0.95` does not surface as 0.949999988079071.
        "temp": round3(args.temp as f64),
        "sampling": {
            "temp": round3(args.temp as f64),
            "top_k": args.top_k,
            "top_p": round3(args.top_p as f64),
            "seed": args.seed,
            "greedy": args.temp <= 0.0 || args.top_k == 1,
        },
        "pp_sizes": [s.prompt_tokens],
        "roofline": {
            "weights_gb": args.weights_gb,
            "bandwidth_gbps": args.bandwidth_gbps,
            "tg_tps_ceiling": round3(ceiling),
        },
        "tg_pct_of_roofline": if ceiling > 0.0 { round3(100.0 * tg / ceiling) } else { 0.0 },
        "host": {
            "platform": host.platform,
            "cpu": host.cpu,
            "mem_bytes": host.mem_bytes,
            "cpu_count": host.cpu_count,
        },
        "versions": {
            "engine": env!("CARGO_PKG_VERSION"),
            "metal_binding": "metal 0.33.0",
        },
        "load_s": round3(run.load_s),
        "ttft": [{
            "turn": 1,
            "prompt_tokens": s.prompt_tokens,
            "prefill_ms": round3(s.prefill_s * 1000.0),
            "ttft_ms": round3(s.ttft_ms),
            "wall_ms": round3(run.wall_ms),
        }],
        "prefill": [{
            "prompt_tokens": s.prompt_tokens,
            "prefill_s": round3(s.prefill_s),
            "pp_tps": round3(s.pp_tps()),
            "gen_tokens": s.gen_tokens,
        }],
        "decode": {
            "prompt_tokens": s.prompt_tokens,
            "gen_tokens": s.gen_tokens,
            "decode_steps": s.decode_steps,
            "decode_s": round3(s.decode_s),
            "tg_tps": round3(tg),
            "finish_reason": s.finish_reason.as_str(),
        },
        "mem": {
            "weights_bytes": run.weights_bytes,
            "metal_alloc_bytes": run.metal_alloc_bytes,
            "metal_recommended_working_set_bytes": run.metal_working_set_bytes,
            "peak_rss_bytes": run.peak_rss_bytes,
            "phys_footprint_bytes": Value::Null,
            "lifetime_max_phys_footprint_bytes": Value::Null,
            "metal_peak_bytes": Value::Null,
            "mem_note": "weights_bytes/metal_alloc_bytes are the authoritative resident figures (UMA Metal buffers are not in getrusage maxrss, which under-reports); phys_footprint/metal_peak instrumentation is R5.5 (T2.2)",
        },
        "engine": {
            "name": run.backend,
            "layers": run.n_layers,
            "layers_run": args.layers.map(|n| n.min(run.n_layers)).unwrap_or(run.n_layers),
            "kv_cache": false,
            "naive_recompute": true,
            "computed_tokens": s.computed_tokens,
            "redone_fraction": round3(s.redone_fraction()),
            "eos_ids": args.eos,
        },
        "mtp": false,
        "tokens": s.tokens,
        "text": run.text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kaplumba_engine::{FinishReason, GenStats};

    fn fake_stats() -> GenStats {
        GenStats {
            prompt_tokens: 4,
            gen_tokens: 3,
            prefill_s: 2.0,
            decode_s: 6.0,
            decode_steps: 2,
            ttft_ms: 2000.0,
            finish_reason: FinishReason::Length,
            tokens: vec![10, 11, 12],
            computed_tokens: 4 + 5 + 6,
        }
    }

    fn fake_outcome(stats: &GenStats) -> RunOutcome<'_> {
        RunOutcome {
            load_s: 4.25,
            wall_ms: 8123.456,
            stats,
            text: "hello",
            backend: "rust-naive-cpu",
            n_layers: 60,
            weights_bytes: 16_028_000_000,
            peak_rss_bytes: 17_000_000_000,
            metal_alloc_bytes: 16_500_000_000,
            metal_working_set_bytes: 19_070_000_000,
        }
    }

    #[test]
    fn report_has_exactly_the_pinned_schema_keys() {
        let stats = fake_stats();
        let report = build_report(&GenerateArgs::default(), &fake_outcome(&stats));
        let obj = report.as_object().expect("report is an object");
        let mut keys: Vec<&str> = obj.keys().map(|s| s.as_str()).collect();
        keys.sort_unstable();
        // Harness keys + the two Rust-only extras; anything else is drift.
        let mut expected: Vec<&str> = REPORT_KEYS.to_vec();
        expected.push("text");
        expected.push("tokens");
        expected.sort_unstable();
        assert_eq!(keys, expected);
        assert_eq!(report["schema"], SCHEMA);
        assert!(keys.contains(&"host") && keys.contains(&"mem"));
    }

    #[test]
    fn report_values_are_consistent_with_stats() {
        let stats = fake_stats();
        let args = GenerateArgs::default();
        let report = build_report(&args, &fake_outcome(&stats));

        assert_eq!(report["load_s"], 4.25);
        assert_eq!(report["ttft"][0]["turn"], 1);
        assert_eq!(report["ttft"][0]["prompt_tokens"], 4);
        assert_eq!(report["ttft"][0]["prefill_ms"], 2000.0);
        assert_eq!(report["ttft"][0]["wall_ms"], 8123.456);
        assert_eq!(report["prefill"][0]["pp_tps"], 2.0); // 4 tokens / 2 s
        assert_eq!(report["decode"]["tg_tps"], 0.333); // 2 steps / 6 s
        assert_eq!(report["decode"]["finish_reason"], "length");
        assert_eq!(report["decode"]["gen_tokens"], 3);
        assert_eq!(report["engine"]["layers"], 60);
        assert_eq!(
            report["engine"]["layers_run"], 60,
            "no --layers → full depth"
        );
        // --layers N truncates the depth actually executed.
        let shallow = GenerateArgs {
            layers: Some(2),
            ..GenerateArgs::default()
        };
        let r3 = build_report(&shallow, &fake_outcome(&stats));
        assert_eq!(r3["engine"]["layers_run"], 2);
        assert_eq!(r3["engine"]["layers"], 60);
        assert_eq!(report["engine"]["redone_fraction"], 2.143); // 15 / (4+3)
        assert_eq!(report["tokens"], json!([10, 11, 12]));
        assert_eq!(report["text"], "hello");
        assert_eq!(report["mtp"], false);
        // Roofline is the DE2 number: 273 / 15.5 = 17.613.
        assert_eq!(report["roofline"]["tg_tps_ceiling"], 17.613);
        assert!(report["host"]["mem_bytes"].as_u64().unwrap_or(0) > 0);
        assert!(!report["host"]["platform"].as_str().unwrap().is_empty());
        // Unmeasured metrics are null, never invented.
        assert!(report["mem"]["phys_footprint_bytes"].is_null());
        assert!(report["mem"]["metal_peak_bytes"].is_null());
    }

    #[test]
    fn greedy_sampling_mode_is_flagged() {
        let stats = fake_stats();
        let args = GenerateArgs {
            temp: 0.0,
            ..GenerateArgs::default()
        };
        let report = build_report(&args, &fake_outcome(&stats));
        assert_eq!(report["sampling"]["greedy"], true);
        assert_eq!(report["temp"], 0.0);
        // f32 → f64 widening must not leak into the report.
        let exact = GenerateArgs {
            temp: 1.0,
            top_p: 0.95,
            ..GenerateArgs::default()
        };
        let r4 = build_report(&exact, &fake_outcome(&stats));
        assert_eq!(r4["temp"], 1.0);
        assert_eq!(
            r4["sampling"]["top_p"], 0.95,
            "got {}",
            r4["sampling"]["top_p"]
        );

        // No decode steps → zero rate, no NaN.
        let no_decode = GenStats {
            decode_s: 0.0,
            decode_steps: 0,
            ..fake_stats()
        };
        let r2 = build_report(&args, &fake_outcome(&no_decode));
        assert_eq!(r2["decode"]["tg_tps"], 0.0);
        assert_eq!(r2["tg_pct_of_roofline"], 0.0);
    }

    #[test]
    fn metrics_are_reported_from_the_host() {
        let rss = peak_rss_bytes();
        assert!(rss > 1_000_000, "peak_rss too small: {rss}");
        let h = host_info();
        assert!(h.cpu_count >= 1);
        assert!(
            h.mem_bytes >= 1,
            "sysctl hw.memsize failed: {}",
            h.mem_bytes
        );
        assert!(h.cpu.contains("Apple") || !h.cpu.is_empty());
        assert!(
            sysctl_string("this.does.not.exist").is_none(),
            "unknown sysctl must be None"
        );
        assert!(sysctl_u64("hw.memsize").unwrap_or(0) > 0);
    }

    #[test]
    fn round3_matches_harness_rounding() {
        // Same 3-decimal presentation as the Python harness. Exact halves are not
        // reachable from real measurements (and Python rounds halves to even
        // while Rust rounds away from zero) so this only pins ordinary values.
        assert_eq!(round3(1.23456), 1.235);
        assert_eq!(round3(0.0), 0.0);
        assert_eq!(round3(1.0005), 1.001);
        assert_eq!(round3(17.6132), 17.613);
        assert_eq!(round3(2.142857), 2.143);
    }
}
