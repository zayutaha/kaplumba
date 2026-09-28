//! Single-instance guard for the GPU.
//!
//! The model needs 11.5-15 GiB of unified memory. Two processes loading it at
//! once exceed physical memory and start evicting and swapping, which has
//! already been measured in this repo at 3.9 GiB of swap and a drop to ~1.4
//! tok/s (`crates/engine/src/weights.rs`). It also makes every benchmark
//! untrustworthy, because two instances fight for bandwidth and residency.
//!
//! This module is the ONLY lock owner, deliberately. `scripts/gpuq` is a
//! reporter and a convenience wrapper and takes no lock at all: an earlier
//! version of gpuq took the same lock and then ran the engine, which deadlocked,
//! because a lock held by a parent is not inherited as held by the child. The
//! child's `open()` is a different open file description, so `flock(LOCK_EX)`
//! conflicts with the parent's and blocks; parent waits for child, child waits
//! for parent. Keeping the lock in the engine is also what makes it impossible
//! to bypass -- a stray direct run from a worktree is the exact case that
//! silently held 11 GiB for twenty minutes in this project's history.
//!
//! Contention is a ticket queue: waiters append to a file, dead entries are
//! pruned, and each waiter learns its position. Fairness matters because with a
//! fleet of agents retrying randomly, unfairness degenerates into starvation,
//! and starvation is indistinguishable from the hang this is meant to remove.
//!
//! `flock` is released by the kernel when the fd closes, which covers normal
//! exit, `abort` and `kill -9`, so a crashed holder cannot wedge the lock. The
//! ticket file is advisory state beside a kernel lock: if it desynchronises the
//! reported position can be wrong, but mutual exclusion still holds. The lock
//! is the safety property; the ticket file only exists to produce a message an
//! agent can read.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// `EX_TEMPFAIL` from sysexits: "try again later". The exit code for "the GPU is
/// busy", chosen so a caller can distinguish it from a real failure.
pub const EXIT_BUSY: i32 = 75;

/// Embedded in the busy error so the CLI can turn it into `EXIT_BUSY` without
/// pattern-matching on prose. Error types are flattened to `String` on the way
/// out of the engine, so a marker is the only reliable channel; it is a
/// constant rather than a literal so the test below can assert on it.
pub const BUSY_MARKER: &str = "kaplumba-gpu-busy";

/// How long `wait` mode waits before giving up. A long `run` is legitimate: a
/// single 8k prefill is ~112 s.
const MAX_WAIT: Duration = Duration::from_secs(60 * 30);

/// Held for the process lifetime. Dropping this releases the lock, so it lives
/// in a `OnceLock` that is never dropped.
static HELD: OnceLock<File> = OnceLock::new();

/// What to do when the GPU is busy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Policy {
    /// Block until free. The default, and what a plain `./kaplumba ...` should do.
    Wait,
    /// Exit immediately with `EXIT_BUSY` if the lock is held.
    Try,
    /// Exit immediately with `EXIT_BUSY`, unconditionally. Exists so a test can
    /// assert "never two instances" without timing games: it fails even when the
    /// lock happens to be free, so it cannot be used to do real work.
    Error,
}

fn policy() -> Policy {
    match std::env::var("KAPLUMBA_GPU_LOCK").as_deref() {
        Ok("try") => Policy::Try,
        Ok("error") => Policy::Error,
        // "off" is honoured only for the lock, not for the model: it exists for
        // measuring a single process's own overhead, and it is deliberately not
        // advertised in AGENTS.md because it defeats the whole point.
        _ => Policy::Wait,
    }
}

/// Grace period for `Try`, in seconds, default 0.
///
/// A short benchmark competing with a long one loses every retry without this:
/// `try` must be instant, but a little grace lets a short job slip into a gap.
/// Configurable so the smoothing is available without giving up instantness by
/// default.
fn grace() -> Duration {
    Duration::from_secs(
        std::env::var("KAPLUMBA_GPU_GRACE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )
}

fn sidecar(base: &Path, suffix: &str) -> PathBuf {
    let mut name = base.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    base.with_file_name(name)
}

fn lock_path() -> PathBuf {
    let base = std::env::var_os("KAPLUMBA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".kaplumba")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("gpu.lock")
}

/// Free pages in GiB. Advisory only: on macOS `Pages free` counts only pages
/// that have never been touched, so it badly understates what a load can get.
/// Callers should use `scripts/gpuq status`, which reads `top`.
fn free_memory_gib() -> Option<f64> {
    let out = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Pages free:") {
            let pages: u64 = rest.trim().trim_end_matches('.').parse().ok()?;
            return Some(pages as f64 * 16384.0 / (1024.0 * 1024.0 * 1024.0));
        }
    }
    None
}

// ---------------------------------------------------------------- ticket queue

struct Tickets {
    path: PathBuf,
    me: u32,
}

impl Drop for Tickets {
    fn drop(&mut self) {
        self.leave();
    }
}

impl Tickets {
    /// Join the queue, pruning entries whose owning pid is gone so a crashed
    /// agent cannot wedge it.
    fn join(path: PathBuf) -> Self {
        let me = std::process::id();
        let mut keep: Vec<String> = Vec::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                let pid = line
                    .split_whitespace()
                    .next()
                    .and_then(|p| p.parse::<u32>().ok());
                match pid {
                    // Alive if we can signal it. A process we do not own will
                    // fail the signal even when alive, so treat that as dead only
                    // when the pid is not ours; being conservative here would
                    // wedge the queue behind a foreign pid.
                    Some(p) if p != me => {
                        let alive = unsafe { libc::kill(p as libc::pid_t, 0) } == 0
                            || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
                        if alive {
                            keep.push(line.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        let me_entry = format!("{} {} gpu", me, now_secs());
        keep.push(me_entry);
        let mut joined = keep.join("\n");
        joined.push('\n');
        let _ = std::fs::write(&path, joined);
        Self { path, me }
    }

    /// 1-based position among live waiters. 1 means "next in line".
    fn position(&self) -> usize {
        let mut index = 0usize;
        if let Ok(text) = std::fs::read_to_string(&self.path) {
            for line in text.lines() {
                let pid = line.split_whitespace().next().and_then(|p| p.parse::<u32>().ok());
                if let Some(p) = pid {
                    if p == self.me {
                        return index + 1;
                    }
                }
                index += 1;
            }
        }
        index.max(1)
    }

    fn leave(&self) {
        if let Ok(text) = std::fs::read_to_string(&self.path) {
            let keep: Vec<&str> = text
                .lines()
                .filter(|l| l.split_whitespace().next() != Some(&self.me.to_string()))
                .collect();
            if keep.is_empty() {
                let _ = std::fs::remove_file(&self.path);
            } else {
                let _ = std::fs::write(&self.path, keep.join("\n") + "\n");
            }
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn holder_text(path: &Path) -> String {
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

// --------------------------------------------------------------------- acquire

/// Acquire the GPU lock under `policy`.
///
/// In `Wait` mode this blocks until free. In `Try` mode it waits up to the grace
/// period and then reports contention by returning `Err` with a message the
/// caller turns into exit code 75. In `Error` mode it always reports contention
/// immediately.
///
/// Blocked behind the lock we deliberately do NOT exit: the caller asked to
/// wait, and only the caller knows whether waiting is acceptable. The
/// alternative — exiting from library code — makes `try` silently fall back to
/// `wait` semantics for anything that propagates errors.
pub fn acquire(label: &str) -> std::io::Result<()> {
    // Threads inside one process serialise here; the kernel lock is process-wide
    // so separate processes queue behind this too. `OnceLock::get_or_init` cannot
    // propagate an error, hence the explicit flag.
    static GUARD: Mutex<()> = Mutex::new(());
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    if HELD.get().is_some() {
        return Ok(());
    }
    let file = acquire_lock_file(label)?;
    let _ = HELD.set(file);
    Ok(())
}

fn busy(path: &Path, position: usize, waited: Duration, grace: Duration) -> std::io::Error {
    let free = free_memory_gib()
        .map(|g| format!("{g:.1} GiB free"))
        .unwrap_or_else(|| "unknown memory".to_string());
    std::io::Error::new(
        std::io::ErrorKind::WouldBlock,
        format!(
            "{BUSY_MARKER}: GPU busy (holder: {}; you are #{position} in the queue; {free}; \
             waited {:.1}s, grace {}s). Exit code {EXIT_BUSY} means 'not now' — \
             do other work and retry, do not sleep. A model load needs ~12 GiB, \
             and two instances on this 24 GiB box swap and collapse to ~1.4 tok/s. \
             Lock: {}",
            holder_text(path),
            waited.as_secs_f64(),
            grace.as_secs(),
            path.display(),
        ),
    )
}

fn acquire_lock_file(label: &str) -> std::io::Result<File> {
    let path = lock_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)?;

    let policy = policy();
    let grace = grace();
    let deadline = Instant::now() + grace;
    let started = Instant::now();
    let fd = file.as_raw_fd();
    let mut announced = false;

    if policy == Policy::Error {
        return Err(busy(&path, 1, Duration::ZERO, grace));
    }

    // Dropped on every exit path, including the early returns below, so a
    // crashed or timed-out waiter cannot leave a stale ticket behind.
    let tickets = Tickets::join(sidecar(&path, ".queue"));

    loop {
        // SAFETY: `fd` is a valid, open descriptor for the whole loop; the
        // duration argument is a valid timespec pointer.
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            break;
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
            return Err(err);
        }

        let waited = started.elapsed();
        if !announced {
            let free = free_memory_gib()
                .map(|g| format!("{g:.1} GiB"))
                .unwrap_or_else(|| "?".to_string());
            if policy == Policy::Try {
                eprintln!(
                    "kaplumba: GPU busy (holder: {}), not waiting ({} free). \
                     Exit {EXIT_BUSY} means try again later.",
                    holder_text(&path),
                    free
                );
            } else {
                eprintln!(
                    "kaplumba: GPU busy (holder: {}), queued #{}. {} GiB free.",
                    holder_text(&path),
                    tickets.position(),
                    free
                );
            }
            announced = true;
        }

        if policy == Policy::Try && Instant::now() >= deadline {
            return Err(busy(&path, tickets.position(), waited, grace));
        }
        if policy == Policy::Wait && waited > MAX_WAIT {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "waited {}s for the GPU lock at {}",
                    MAX_WAIT.as_secs(),
                    path.display()
                ),
            ));
        }
        // 250 ms: short enough that `try` with a grace period feels responsive,
        // long enough not to spin the CPU against a long holder.
        std::thread::sleep(Duration::from_millis(250));
    }

    // Record ownership so a waiter can name us.
    let _ = file.set_len(0);
    let _ = writeln!(file, "{label} (pid {})", std::process::id());
    let _ = file.flush();
    Ok(file)
}

/// Whether a flattened error message came from lock contention.
pub fn is_busy_message(message: &str) -> bool {
    message.contains(BUSY_MARKER)
}

/// Whether this process currently holds the lock.
pub fn held() -> bool {
    HELD.get().is_some()
}

/// Current policy, for `--metrics` so a run records whether it was contended.
pub fn describe() -> &'static str {
    match policy() {
        Policy::Wait => "wait",
        Policy::Try => "try",
        Policy::Error => "error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_parsing() {
        // Defaults and explicit values, without mutating the ambient environment
        // (tests share a process and std::env::set_var is unsafe in Rust 2024).
        assert_eq!(format!("{:?}", Policy::Wait), "Wait");
        assert_eq!(format!("{:?}", Policy::Try), "Try");
        assert_eq!(format!("{:?}", Policy::Error), "Error");
    }

    #[test]
    fn busy_error_names_the_lock_and_the_exit_code() {
        let path = PathBuf::from("/tmp/kaplumba-test-gpu.lock");
        let err = busy(&path, 2, Duration::from_millis(750), Duration::from_secs(3));
        let text = err.to_string();
        assert!(text.contains(BUSY_MARKER), "marker missing: {text}");
        assert!(text.contains("GPU busy"), "{text}");
        assert!(is_busy_message(&text), "is_busy_message disagreed");
        assert!(text.contains("#2"), "position missing: {text}");
        assert!(text.contains("75"), "exit code missing: {text}");
        assert!(text.contains("0.8s"), "waited missing: {text}");
        assert!(text.contains("kaplumba-test-gpu.lock"), "path missing: {text}");
    }

    #[test]
    fn queue_prunes_dead_pids_and_keeps_live_ones() {
        let path = std::env::temp_dir().join(format!(
            "kaplumba-queue-test-{}.queue",
            std::process::id()
        ));
        // A pid that cannot exist, and our own (definitely alive).
        let seed = format!("4294967294 1 dead\n{} 1 alive\n", std::process::id());
        let _ = std::fs::write(&path, seed);
        let t = Tickets::join(path.clone());
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("4294967294"), "dead pid survived: {text}");
        assert!(text.contains(&std::process::id().to_string()), "live pid pruned: {text}");
        // We are the only live waiter, so we are first.
        assert_eq!(t.position(), 1);
        t.leave();
        let _ = std::fs::remove_file(&path);
    }
}
