//! A short read test of the drive that holds the models. Models that stream their experts from the
//! SSD read them in many separate blocks, so the speed of random reads is what sets the pace.
//! The test reads blocks at random places of the largest file it finds, one at a time (queue depth
//! one), each block once. What is measured is therefore the speed of single 2 MiB random reads
//! through the operating system, NOT the drive's best parallel speed, and a file the system has
//! recently read can come back from memory (no O_DIRECT or cache dropping is attempted: either
//! needs privileges or platform code), so the figure is an UPPER estimate for a cold read.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const BLOCK: usize = 2 * 1024 * 1024;
const MIN_FILE: u64 = 256 * 1024 * 1024;

/// How fast the drive is for this kind of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiskClass {
    Fast,
    Fair,
    Slow,
}

pub fn classify(mb_per_s: f64) -> DiskClass {
    if mb_per_s >= 2500.0 {
        DiskClass::Fast
    } else if mb_per_s >= 800.0 {
        DiskClass::Fair
    } else {
        DiskClass::Slow
    }
}

/// The largest regular file under `dir` (two levels deep), at least `MIN_FILE` bytes.
pub fn largest_file(dir: &Path) -> Option<(PathBuf, u64)> {
    fn walk(dir: &Path, depth: u8, best: &mut Option<(PathBuf, u64)>) {
        let Ok(rd) = fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            // `fs::metadata` follows symbolic links (a model folder may link its weights in)
            let Ok(md) = fs::metadata(&p) else { continue };
            if md.is_file() {
                if md.len() >= MIN_FILE && best.as_ref().is_none_or(|b| md.len() > b.1) {
                    *best = Some((p, md.len()));
                }
            } else if md.is_dir() && depth > 0 {
                walk(&p, depth - 1, best);
            }
        }
    }
    let mut best = None;
    walk(dir, 2, &mut best);
    best
}

/// Random 2 MiB reads, each block at most once, for at most `budget` time or 1 GiB; returns MB/s
/// (decimal). Errors when nothing could be read.
pub fn random_read_speed(dir: &Path, budget: Duration) -> Result<f64, String> {
    random_read(dir, budget).map(|(bytes, secs)| bytes as f64 / 1_000_000.0 / secs)
}

/// The same read, as (bytes read, seconds taken), so the figures can be checked.
fn random_read(dir: &Path, budget: Duration) -> Result<(u64, f64), String> {
    let (path, len) = largest_file(dir).ok_or("no large file to read in that folder")?;
    let mut f = File::open(&path).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; BLOCK];
    let blocks = (len / BLOCK as u64).max(1);
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(88172645463325252)
        | 1;
    // a block read twice would come back from memory and inflate the figure
    let mut done: HashSet<u64> = HashSet::new();
    let mut misses = 0u32;
    let start = Instant::now();
    let mut total: u64 = 0;
    while start.elapsed() < budget && total < 1024 * 1024 * 1024 && (done.len() as u64) < blocks {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let b = state % blocks;
        if !done.insert(b) {
            // many repeats mean most blocks are taken: stop instead of spinning
            misses += 1;
            if misses > 10_000 {
                break;
            }
            continue;
        }
        f.seek(SeekFrom::Start(b * BLOCK as u64))
            .map_err(|e| e.to_string())?;
        // read until the block is full: a single `read` may return less
        let mut got = 0;
        while got < BLOCK {
            match f.read(&mut buf[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.to_string()),
            }
        }
        total += got as u64;
    }
    if total == 0 {
        return Err("nothing could be read from the file (empty, shrunk or zero time)".into());
    }
    Ok((total, start.elapsed().as_secs_f64().max(1e-6)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert_eq!(classify(3500.0), DiskClass::Fast);
        assert_eq!(classify(1200.0), DiskClass::Fair);
        assert_eq!(classify(300.0), DiskClass::Slow);
    }

    #[test]
    fn reads_a_large_file_and_refuses_an_empty_folder() {
        let dir = std::env::temp_dir().join(format!("diskbench-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        assert!(random_read_speed(&dir, Duration::from_millis(50)).is_err());
        let f = File::create(dir.join("big.bin")).unwrap();
        f.set_len(MIN_FILE + 1024).unwrap();
        let v = random_read_speed(&dir, Duration::from_millis(200)).unwrap();
        assert!(v > 0.0);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Removes its folder when dropped, so a failed assertion leaves nothing behind.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(name: &str) -> Self {
            let d = std::env::temp_dir().join(format!("diskbench-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&d);
            fs::create_dir_all(&d).unwrap();
            Scratch(d)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn every_block_is_read_at_most_once_so_the_page_cache_is_not_re_read() {
        let d = Scratch::new("once");
        let f = File::create(d.0.join("big.bin")).unwrap();
        f.set_len(MIN_FILE).unwrap();
        let (bytes, secs) = random_read(&d.0, Duration::from_secs(30)).unwrap();
        assert_eq!(
            bytes, MIN_FILE,
            "128 blocks, each once, never more than the file"
        );
        assert!(secs > 0.0);
    }

    #[test]
    fn a_zero_budget_or_an_unreadable_file_is_an_error_not_zero_speed() {
        let d = Scratch::new("zero");
        let f = File::create(d.0.join("big.bin")).unwrap();
        f.set_len(MIN_FILE + 7).unwrap();
        assert!(random_read_speed(&d.0, Duration::ZERO).is_err());
    }

    #[test]
    fn a_file_smaller_than_the_minimum_is_not_used() {
        let d = Scratch::new("small");
        fs::write(d.0.join("small.bin"), vec![1u8; 4096]).unwrap();
        assert!(largest_file(&d.0).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_linked_weights_file_is_found() {
        let d = Scratch::new("link");
        let store = d.0.join("store");
        fs::create_dir_all(&store).unwrap();
        let real = store.join("w.bin");
        File::create(&real).unwrap().set_len(MIN_FILE + 1).unwrap();
        let models = d.0.join("models");
        fs::create_dir_all(&models).unwrap();
        std::os::unix::fs::symlink(&real, models.join("w.safetensors")).unwrap();
        let (p, len) = largest_file(&models).expect("the link must be followed");
        assert!(p.ends_with("w.safetensors") && len == MIN_FILE + 1);
        assert!(random_read_speed(&models, Duration::from_millis(100)).unwrap() > 0.0);
    }
}
