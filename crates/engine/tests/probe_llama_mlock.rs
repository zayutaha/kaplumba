//! Probe: [`LlamaMlock`] mirrors `llama.cpp`'s `llama_mlock` -- incremental
//! growth, granularity rounding, and a *soft* clamp on failure (`R2` of
//! `spec/llama-mlock-zero-swap`).
//!
//! No Metal and no model: this locks plain anonymous `mmap` pages, so it can
//! run on any box (and next to a model load, unlike the GPU probes).
//!
//! Non-vacuity & sabotage (AGENTS.md §3, testing.md §2 Sabotage 2):
//! changing `LlamaMlock::grow_to` to ignore the `mlock` return code must make
//! [`oversized_grow_fails_soft_without_panicking`] fail on
//! `failed() == true`. That sabotage run is recorded in
//! `spec/llama-mlock-zero-swap/implementation_notes.md`.

use kaplumba_engine::weights::LlamaMlock;

fn page_size() -> usize {
    // SAFETY: valid sysconf query, as in `LlamaMlock::lock_granularity`.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(value > 0, "sysconf(_SC_PAGESIZE) = {value}");
    value as usize
}

/// Private anonymous mapping, the cheapest region that can be wired.
fn map(len: usize) -> *mut u8 {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(ptr, libc::MAP_FAILED, "mmap({len}) failed");
    ptr.cast::<u8>()
}

fn unmap(ptr: *mut u8, len: usize) {
    let rc = unsafe { libc::munmap(ptr.cast(), len) };
    assert_eq!(rc, 0, "munmap failed");
}

/// Incremental growth with granularity rounding, exactly as
/// `llama_mlock::impl::grow_to` (`llama-mmap.cpp:803`):
///
/// * the target rounds **up** to `lock_granularity()`;
/// * only `[addr + size, target)` is locked, so `size` walks upward;
/// * a smaller or equal target is a no-op (no unlock, no shrink);
/// * every size is a multiple of the granularity.
#[test]
fn grow_to_is_incremental_and_page_aligned() {
    let page = LlamaMlock::lock_granularity();
    assert_eq!(
        page,
        page_size(),
        "lock_granularity must be sysconf(_SC_PAGESIZE), llama.cpp's POSIX rule"
    );

    let len = 4 * page;
    let base = map(len);
    let mut lock = LlamaMlock::init(base.cast());
    assert_eq!(lock.size(), 0, "init wires nothing");
    assert!(!lock.failed(), "a fresh lock has not failed");

    // Rounds up to exactly one page.
    lock.grow_to(1);
    assert_eq!(lock.size(), page, "grow_to(1) must round to one whole page");
    assert!(!lock.failed());

    // Rounds up to two pages, and locks only the second (incremental).
    lock.grow_to(page + 1);
    assert_eq!(lock.size(), 2 * page);
    assert!(!lock.failed());

    // Shrinking or repeating targets are no-ops: llama.cpp only ever grows.
    lock.grow_to(page);
    assert_eq!(lock.size(), 2 * page, "grow_to must never shrink the lock");
    lock.grow_to(2 * page);
    assert_eq!(lock.size(), 2 * page, "an equal target must not re-lock");
    assert!(!lock.failed());

    // One more incremental step.
    lock.grow_to(3 * page);
    assert_eq!(lock.size(), 3 * page, "incremental growth to the third page");
    assert!(!lock.failed());
    assert_eq!(lock.size() % page, 0, "locked size must stay page-aligned");

    // No Drop unlock: dropping the lock object must leave the pages wired
    // (the region belongs to the mapping, not the lock -- `munmap` below is
    // what releases the wire, exactly as `ShardReader`'s Drop does).
    drop(lock);
    let still = LlamaMlock::init(base.cast());
    assert_eq!(still.size(), 0, "a new lock starts at zero regardless");
    drop(still);

    unmap(base, len);
}

/// Soft clamp on `ENOMEM`: llama.cpp sets `failed_already`, logs once, and
/// keeps going (`llama-mmap.cpp:811-816`). The model load must survive this,
/// so nothing here may panic, and the lock must remember its last good size.
#[test]
fn oversized_grow_fails_soft_without_panicking() {
    let page = LlamaMlock::lock_granularity();
    let len = 2 * page;
    let base = map(len);
    let mut lock = LlamaMlock::init(base.cast());

    // A legitimate first lock succeeds...
    lock.grow_to(page);
    assert_eq!(lock.size(), page);
    assert!(!lock.failed());

    // ...then an absurd target (100 GiB, far past the mapping and past any
    // wire limit) must fail softly: no panic, `failed_already` latched.
    lock.grow_to(100 * 1024 * 1024 * 1024);
    assert!(
        lock.failed(),
        "a 100 GiB mlock must set failed_already; it returned success, which \
         means grow_to is not checking mlock's return code"
    );
    assert_eq!(
        lock.size(),
        page,
        "a failed grow must not advance size past what is actually locked"
    );

    // `failed_already` short-circuit: even a target that *would* fit is a
    // no-op now -- the llama.cpp behavior that keeps a failing box from
    // spamming one syscall per tensor.
    lock.grow_to(2 * page);
    assert_eq!(
        lock.size(),
        page,
        "after a failure every later grow_to must be a no-op"
    );
    assert!(lock.failed());

    drop(lock);
    unmap(base, len);
}

/// The failure path must leave the *previously locked* bytes locked: a soft
/// clamp is a stop-growing, not an undo.
#[test]
fn a_failed_grow_keeps_the_successful_prefix() {
    let page = LlamaMlock::lock_granularity();
    let len = page;
    let base = map(len);
    let mut lock = LlamaMlock::init(base.cast());
    lock.grow_to(page);
    assert!(!lock.failed());
    lock.grow_to(usize::MAX); // rounds (saturating) then must fail ENOMEM
    assert!(lock.failed(), "usize::MAX must not succeed and must not panic");
    assert_eq!(
        lock.size(),
        page,
        "the successful prefix stays locked while growth is refused"
    );
    drop(lock);
    unmap(base, len);
}
