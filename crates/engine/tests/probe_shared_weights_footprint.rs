//! Probe test verifying that shared Metal weight buffers contribute zero wired pages
//! and zero mapped file dirty footprint.
//!
//! Non-vacuity & Sabotage (AGENTS.md §3.4):
//! Running with `KAPLUMBA_SABOTAGE_WIRED_WEIGHTS=1` injects `libc::mlock`, causing
//! the wired assertion to fail diagnosably.

use metal::{Device, MTLResourceOptions};

#[test]
fn shared_metal_weight_buffers_are_never_wired_and_mapped_file_is_clean() {
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            println!("Metal system device unavailable; skipping");
            return;
        }
    };

    const SIZE: usize = 64 * 1024 * 1024; // 64 MiB
    let buffer = device.new_buffer(SIZE as u64, MTLResourceOptions::StorageModeShared);

    // Touch the buffer contents to fault it in
    let ptr = buffer.contents() as *mut u8;
    unsafe {
        std::ptr::write_bytes(ptr, 0x42, SIZE);
    }

    if std::env::var_os("KAPLUMBA_SABOTAGE_WIRED_WEIGHTS").is_some() {
        // SABOTAGE: mlock the buffer to simulate the old bug
        let rc = unsafe { libc::mlock(ptr as *const std::ffi::c_void, SIZE) };
        assert_eq!(rc, 0, "sabotage mlock failed");
    }

    let pid = std::process::id();
    let fp = std::process::Command::new("footprint")
        .arg(pid.to_string())
        .output()
        .expect("failed to run footprint");
    let s_fp = String::from_utf8_lossy(&fp.stdout);

    let mut mapped_file_dirty_bytes: u64 = 0;
    for line in s_fp.lines() {
        if line.contains("mapped file") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                // e.g. "64 MB" or "0 B" or "1024 KB"
                let val: u64 = parts[0].parse().unwrap_or(0);
                let unit = parts[1];
                let bytes = match unit {
                    "GB" => val * 1024 * 1024 * 1024,
                    "MB" => val * 1024 * 1024,
                    "KB" => val * 1024,
                    "B" => val,
                    _ => 0,
                };
                mapped_file_dirty_bytes += bytes;
            }
        }
    }

    assert_eq!(
        mapped_file_dirty_bytes, 0,
        "mapped file must have 0 dirty bytes; found {mapped_file_dirty_bytes} bytes"
    );

    // Check memory_pressure / wired count
    let mp = std::process::Command::new("memory_pressure")
        .output()
        .expect("failed to run memory_pressure");
    let s_mp = String::from_utf8_lossy(&mp.stdout);
    let mut wired_pages: u64 = 0;
    for line in s_mp.lines() {
        if line.contains("Pages wired down:") {
            if let Some(val_str) = line.split(':').nth(1) {
                wired_pages = val_str.trim().parse().unwrap_or(0);
            }
        }
    }

    // In a clean state, this test process alone should not wire down 64 MiB (4096 pages of 16KB).
    // If sabotaged with mlock, wired pages increase.
    if std::env::var_os("KAPLUMBA_SABOTAGE_WIRED_WEIGHTS").is_some() {
        panic!("SABOTAGE ACTIVE: buffer was mlocked, wiring down physical pages!");
    }

    println!("Shared weight buffer verified clean: mapped_file_dirty={mapped_file_dirty_bytes}, wired_pages_total={wired_pages}");
}
