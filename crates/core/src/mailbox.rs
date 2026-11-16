//! A one-slot mailbox for values that are only worth reading once, the newest one: a health
//! check, a status report, a resource snapshot. A sampler thread that keeps producing while the
//! window is hidden (and so not draining its channel) would otherwise grow the queue without
//! limit; here a newer value replaces an older one that nobody has read, and the channel carries
//! at most one wake-up per unread slot.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Latest<T> {
    slot: Mutex<Option<T>>,
    /// A wake-up has been announced and the slot has not been taken since.
    announced: AtomicBool,
}

impl<T> Default for Latest<T> {
    fn default() -> Self {
        Self {
            slot: Mutex::new(None),
            announced: AtomicBool::new(false),
        }
    }
}

impl<T> Latest<T> {
    /// Stores `value`, replacing an unread one. Returns true when the caller must now send a
    /// wake-up to the reader (none is on its way yet); false when one is already pending.
    pub fn put(&self, value: T) -> bool {
        *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(value);
        !self.announced.swap(true, Ordering::SeqCst)
    }

    /// The newest value, if any. Call it when a wake-up arrives.
    pub fn take(&self) -> Option<T> {
        // Cleared first: a value stored right after this line announces itself again, so no
        // update is ever left unread with no wake-up on its way.
        self.announced.store(false, Ordering::SeqCst);
        self.slot.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn a_newer_value_replaces_an_unread_one_and_only_one_wake_up_is_asked_for() {
        let m = Latest::default();
        assert!(m.put(1));
        assert!(!m.put(2));
        assert!(!m.put(3));
        assert_eq!(m.take(), Some(3));
        assert_eq!(m.take(), None);
        assert!(m.put(4), "after a take the next put announces itself again");
    }

    #[test]
    fn under_concurrency_the_last_value_always_arrives_and_wake_ups_stay_few() {
        for _ in 0..50 {
            let m = Arc::new(Latest::default());
            let wakes = Arc::new(AtomicUsize::new(0));
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            let producer = {
                let (m, wakes, tx) = (m.clone(), wakes.clone(), tx.clone());
                std::thread::spawn(move || {
                    for i in 1..=2000u32 {
                        if m.put(i) {
                            wakes.fetch_add(1, Ordering::SeqCst);
                            tx.send(()).unwrap();
                        }
                    }
                })
            };
            drop(tx);
            let mut last = 0;
            let mut woken = 0;
            while let Ok(()) = rx.recv() {
                woken += 1;
                if let Some(v) = m.take() {
                    assert!(v > last, "values never go backwards");
                    last = v;
                }
            }
            producer.join().unwrap();
            if let Some(v) = m.take() {
                last = last.max(v);
            }
            assert_eq!(last, 2000, "the newest value was lost");
            assert_eq!(woken, wakes.load(Ordering::SeqCst));
            assert!(woken <= 2000);
        }
    }

    #[test]
    fn an_unread_queue_cannot_grow_without_limit() {
        let m = Latest::default();
        let sent = (0..100_000).filter(|i| m.put(*i)).count();
        assert_eq!(sent, 1);
    }
}
