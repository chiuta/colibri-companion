//! A short read test of the drive that holds the models. Models that stream their experts from the
//! SSD read them in many separate blocks, so the speed of random reads is what sets the pace.
//! The test reads blocks at random places of the largest file it finds; a file the system has
//! recently read can come back from memory, so the figure is an upper estimate.

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
            let Ok(md) = e.metadata() else { continue };
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

/// Random 2 MiB reads for at most `budget` time or 1 GiB; returns MB/s.
pub fn random_read_speed(dir: &Path, budget: Duration) -> Result<f64, String> {
    let (path, len) = largest_file(dir).ok_or("no large file to read in that folder")?;
    let mut f = File::open(&path).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; BLOCK];
    let blocks = (len / BLOCK as u64).max(1);
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(88172645463325252)
        | 1;
    let start = Instant::now();
    let mut total: u64 = 0;
    while start.elapsed() < budget && total < 1024 * 1024 * 1024 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let off = (state % blocks) * BLOCK as u64;
        f.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        total += n as u64;
    }
    let secs = start.elapsed().as_secs_f64().max(1e-6);
    Ok(total as f64 / 1_000_000.0 / secs)
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
}
