//! Graceful shutdown without dependencies: Ctrl-C / SIGTERM (Unix) and Ctrl-C / Ctrl-Break / close (Windows)
//! set a flag that the server loop polls.
use std::sync::atomic::{AtomicBool, Ordering};

static REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn requested() -> bool { REQUESTED.load(Ordering::SeqCst) }
/// Ask for shutdown (what the signal handlers do; also used by tests).
pub fn request() { REQUESTED.store(true, Ordering::SeqCst); }
#[cfg(test)]
pub fn reset() { REQUESTED.store(false, Ordering::SeqCst); }

#[cfg(unix)]
mod sys {
    unsafe extern "C" { fn signal(signum: i32, handler: usize) -> usize; }
    extern "C" fn on_signal(_: i32) { super::request(); }
    pub fn install() -> bool {
        // SAFETY: `on_signal` only stores to an atomic, which is async-signal-safe.
        unsafe { signal(2, on_signal as extern "C" fn(i32) as usize) != usize::MAX && signal(15, on_signal as extern "C" fn(i32) as usize) != usize::MAX }
    }
}

#[cfg(windows)]
mod sys {
    unsafe extern "system" { fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32; }
    unsafe extern "system" fn on_ctrl(_: u32) -> i32 { super::request(); 1 }
    pub fn install() -> bool {
        // SAFETY: the handler only stores to an atomic.
        unsafe { SetConsoleCtrlHandler(Some(on_ctrl), 1) != 0 }
    }
}

#[cfg(not(any(unix, windows)))]
mod sys { pub fn install() -> bool { false } }

/// Install the handlers; `false` if the platform refused (Ctrl-C then kills the process as usual).
pub fn install() -> bool { sys::install() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_round_trip_and_install() {
        reset();
        assert!(!requested());
        assert!(install());
        request();
        assert!(requested());
        reset();
    }

    #[cfg(unix)]
    #[test]
    fn a_real_signal_sets_the_flag() {
        unsafe extern "C" { fn raise(sig: i32) -> i32; }
        // use SIGTERM: with our handler installed it must not kill the test process
        assert!(install());
        reset();
        // SAFETY: plain libc call; our handler is installed.
        assert_eq!(unsafe { raise(15) }, 0);
        for _ in 0..100 { if requested() { break; } std::thread::sleep(std::time::Duration::from_millis(10)); }
        assert!(requested());
        reset();
    }
}
