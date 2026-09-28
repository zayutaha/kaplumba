//! Single-instance guard for the GPU.
//!
//! The model needs 11.5-14 GiB of unified memory. Two processes loading it at
//! once exceed physical memory and start evicting and swapping, which has
//! already been measured in this repo at 3.9 GiB of swap and a drop to ~1.4
//! tok/s (`crates/engine/src/weights.rs`). It also makes every benchmark
//! untrustworthy, because the two instances fight for bandwidth and page
//! residency.
//!
//! So loading is serialised with an advisory `flock(2)` on
//! `~/.kaplumba/gpu.lock`. A second process blocks *before* it touches the
//! weights, prints who it is waiting for, and only proceeds once the first
//! exits. The lock lives in the binary rather than in a wrapper script because a
//! script can be bypassed by running the binary directly, and that is exactly
//! what a parallel agent fleet will do.
//!
//! `flock` is released by the kernel when the file descriptor closes, which
//! includes normal exit, `abort`, and a `kill -9`. A crashed holder therefore
//! does not wedge the lock.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// How long to wait for a peer before giving up. Generous: a 8192-token prefill
/// plus a decode loop can legitimately hold the lock for minutes.
const MAX_WAIT: Duration = Duration::from_secs(60 * 30);

/// Held for the process lifetime. Dropping this would release the lock, so it
/// lives in a `OnceLock` that is never dropped.
static HELD: OnceLock<File> = OnceLock::new();

fn lock_path() -> PathBuf {
    let base = std::env::var_os("KAPLUMBA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".kaplumba")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("gpu.lock")
}

fn free_memory_bytes() -> Option<u64> {
    // Darwin reports free pages in vm_stat as "Pages free".
    let out = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Pages free:") {
            let pages: u64 = rest.trim().trim_end_matches('.').parse().ok()?;
            return Some(pages * 16_384);
        }
    }
    None
}

/// Block until this process owns the GPU, then return. Called immediately
/// before the weights are mapped.
///
/// The second and later processes in a fleet do not fail: they wait, print who
/// they are, and run afterwards, so a hundred agents queue up instead of
/// thrashing. That is the intended behaviour, not an error.
pub fn acquire(label: &str) -> std::io::Result<()> {
    // Threads inside one process serialise here; the lock itself is process-wide
    // so separate processes queue behind this too.
    static THREAD_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = THREAD_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    if HELD.get().is_some() {
        return Ok(());
    }
    let file = acquire_lock_file(label)?;
    // `set` can only fail if another thread won the race, which the mutex above
    // prevents; if it somehow happened the loser still leaves its fd open via
    // the returned `File`, so the lock stays valid either way.
    if HELD.set(file).is_err() {
        return Ok(());
    }
    Ok(())
}

fn acquire_lock_file(label: &str) -> std::io::Result<File> {
    {
        let path = lock_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        // A separate fd for the non-blocking probe so the message loop does not
        // disturb the blocking acquire below.
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;

        let fd = file.as_raw_fd();
        let started = Instant::now();
        let mut announced = false;

        loop {
            // SAFETY: `fd` is a valid, open descriptor for the whole loop; the
            // duration argument is a valid timespec pointer.
            let rc = unsafe {
                libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB)
            };
            if rc == 0 {
                break;
            }
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
                return Err(err);
            }
            if !announced {
                if let Some(free) = free_memory_bytes() {
                    eprintln!(
                        "kaplumba: GPU is busy, waiting for the lock ({}). \
                         Free memory now {:.1} GiB -- loading the model again now \
                         would push the machine into swap.",
                        path.display(),
                        free as f64 / (1024.0 * 1024.0 * 1024.0),
                    );
                } else {
                    eprintln!(
                        "kaplumba: GPU is busy, waiting for the lock ({}).",
                        path.display()
                    );
                }
                // Leave a breadcrumb in the lock file so a waiting agent can see
                // who holds it, without disturbing the lock itself.
                let _ = file.set_len(0);
                let _ = writeln!(file, "{label} (pid {})", std::process::id());
                announced = true;
            }
            if started.elapsed() > MAX_WAIT {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("waited {}s for the GPU lock at {}", MAX_WAIT.as_secs(), path.display()),
                ));
            }
            std::thread::sleep(Duration::from_millis(500));
        }

        // Record ownership for anyone waiting behind us.
        let _ = file.set_len(0);
        let _ = writeln!(file, "{label} (pid {})", std::process::id());
        let _ = file.flush();
        Ok(file)
    }
}

/// Whether this process currently holds the lock. Used by tests and by the
/// `--metrics` banner so a benchmark run records whether it was contended.
pub fn held() -> bool {
    HELD.get().is_some()
}
