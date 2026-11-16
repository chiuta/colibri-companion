//! A short read test of the drive that holds the models. Models that stream their experts from the
//! SSD read them in many separate blocks, so the speed of random reads is what sets the pace.
//! The test reads blocks at random places of the largest file it finds, one at a time (queue depth
//! one), each block once. What is measured is therefore the speed of single 2 MiB random reads
//! through the operating system, NOT the drive's best parallel speed.
//!
//! A file the system has recently read comes back from memory and would pass for a fast drive. On
//! Linux the test asks the system to drop the file's clean pages first (`posix_fadvise`, which
//! needs no privileges) and then asks, for every block, whether it was in memory before it was
//! read (`mincore`): the answer is reported as [`CacheState`], so a figure is never presented as
//! the drive's when it may be the page cache's. On other systems nothing is checked
//! ([`CacheState::Unknown`]) and the figure is an UPPER estimate for a cold read.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const BLOCK: usize = 2 * 1024 * 1024;
const MIN_FILE: u64 = 256 * 1024 * 1024;

/// Whether the blocks that were read came from the drive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CacheState {
    /// Almost none of the blocks (at most 5%) were in memory before they were read: the figure is
    /// the drive's.
    Cold,
    /// Part of the blocks were already in memory (this share, 0 to 1): the figure may be the page
    /// cache's, not the drive's.
    MaybeCached(f64),
    /// This system was not asked: the figure is an upper estimate.
    Unknown,
}

/// The result of one read test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadReport {
    pub mb_per_s: f64,
    pub bytes: u64,
    pub seconds: f64,
    pub cache: CacheState,
}

impl ReadReport {
    /// The class of the figure, never `Fast` when more than half of the blocks were in memory.
    pub fn class(&self) -> DiskClass {
        let c = classify(self.mb_per_s);
        match self.cache {
            CacheState::MaybeCached(share) if share > 0.5 && c == DiskClass::Fast => {
                DiskClass::Fair
            }
            _ => c,
        }
    }
}

/// How fast the drive is for this kind of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiskClass {
    Fast,
    Fair,
    Slow,
}

impl DiskClass {
    /// The slowest speed that still falls in this class (MB/s). Advice that depends on a class is
    /// judged at this floor, never at an optimistic figure inside the class.
    pub fn floor_mbs(self) -> f64 {
        match self {
            DiskClass::Fast => 2500.0,
            DiskClass::Fair => 800.0,
            DiskClass::Slow => 50.0,
        }
    }
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
/// (decimal). Errors when nothing could be read. See [`random_read_report`] for whether the
/// figure came from the drive.
pub fn random_read_speed(dir: &Path, budget: Duration) -> Result<f64, String> {
    random_read_report(dir, budget).map(|r| r.mb_per_s)
}

/// The same test, with the figures needed to judge it.
pub fn random_read_report(dir: &Path, budget: Duration) -> Result<ReadReport, String> {
    let (bytes, seconds, cache) = random_read(dir, budget)?;
    Ok(ReadReport {
        mb_per_s: bytes as f64 / 1_000_000.0 / seconds,
        bytes,
        seconds,
        cache,
    })
}

#[cfg(target_os = "linux")]
mod pagecache {
    //! The two Linux calls behind [`super::CacheState`]. Every `unsafe` block states why it is sound.
    use std::fs::File;
    use std::os::fd::AsRawFd;

    pub fn page_size() -> usize {
        // SAFETY: sysconf has no preconditions.
        let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(n)
            .ok()
            .filter(|n| *n >= 512)
            .unwrap_or(4096)
    }

    /// Ask the system to drop the clean pages it holds of this file, and not to read ahead
    /// (random reads must not be answered by a neighbour's read-ahead). Advisory: the caller
    /// checks the result with [`Map::resident`].
    pub fn drop_cache_and_readahead(f: &File) {
        let fd = f.as_raw_fd();
        // SAFETY: `fd` is a descriptor owned by `f`, which outlives the calls; offset 0 and length
        // 0 mean the whole file.
        unsafe {
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_RANDOM);
        }
    }

    /// A read-only view of a file that is never read through, only asked about with `mincore`.
    pub struct Map {
        ptr: *mut u8,
        len: usize,
    }

    impl Map {
        pub fn new(f: &File, len: u64) -> Option<Map> {
            let len = usize::try_from(len).ok().filter(|l| *l > 0)?;
            // SAFETY: the descriptor is valid for the call, `len` > 0, and the result is compared
            // with MAP_FAILED before use. The mapping is only ever passed to mincore/munmap.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    f.as_raw_fd(),
                    0,
                )
            };
            (p != libc::MAP_FAILED).then(|| Map {
                ptr: p.cast::<u8>(),
                len,
            })
        }

        /// (pages of `[offset, offset + len)` that are in memory, pages asked about). `None` when
        /// the range is not inside the mapping or not page aligned, or the system refuses.
        pub fn resident(&self, offset: usize, len: usize) -> Option<(usize, usize)> {
            let page = page_size();
            let end = offset.checked_add(len)?;
            if len == 0 || offset % page != 0 || end > self.len {
                return None;
            }
            let pages = len.div_ceil(page);
            let mut v = vec![0u8; pages];
            // SAFETY: the range lies inside the mapping (checked above), is page aligned at its
            // start, and `v` has one byte per page of the range as mincore requires.
            let r = unsafe {
                libc::mincore(
                    self.ptr.add(offset).cast::<libc::c_void>(),
                    len,
                    v.as_mut_ptr(),
                )
            };
            (r == 0).then(|| (v.iter().filter(|b| **b & 1 == 1).count(), pages))
        }
    }

    impl Drop for Map {
        fn drop(&mut self) {
            // SAFETY: `ptr`/`len` are exactly what mmap returned and nothing else unmaps them.
            unsafe {
                libc::munmap(self.ptr.cast::<libc::c_void>(), self.len);
            }
        }
    }
}

/// The same read, as (bytes read, seconds taken, where the blocks came from).
fn random_read(dir: &Path, budget: Duration) -> Result<(u64, f64, CacheState), String> {
    let (path, len) = largest_file(dir).ok_or("no large file to read in that folder")?;
    let mut f = File::open(&path).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; BLOCK];
    let blocks = (len / BLOCK as u64).max(1);
    #[cfg(target_os = "linux")]
    let map = {
        pagecache::drop_cache_and_readahead(&f);
        pagecache::Map::new(&f, len)
    };
    // (pages in memory before they were read, pages asked about)
    let mut resident = (0usize, 0usize);
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(88172645463325252)
        | 1;
    // a block read twice would come back from memory and inflate the figure
    let mut done: HashSet<u64> = HashSet::new();
    let mut misses = 0u32;
    let start = Instant::now();
    let mut waited = Duration::ZERO; // time spent asking the system about the cache: not read time
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
        #[cfg(target_os = "linux")]
        if let Some(m) = &map {
            let t = Instant::now();
            if let Some((r, n)) = m.resident((b as usize).saturating_mul(BLOCK), BLOCK) {
                resident.0 += r;
                resident.1 += n;
            }
            waited += t.elapsed();
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
    let secs = start
        .elapsed()
        .saturating_sub(waited)
        .as_secs_f64()
        .max(1e-6);
    let cache = if resident.1 == 0 {
        CacheState::Unknown
    } else {
        let share = resident.0 as f64 / resident.1 as f64;
        if share <= 0.05 {
            CacheState::Cold
        } else {
            CacheState::MaybeCached(share)
        }
    };
    Ok((total, secs, cache))
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
        let (bytes, secs, _) = random_read(&d.0, Duration::from_secs(30)).unwrap();
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

    #[test]
    fn a_figure_from_memory_is_never_called_fast() {
        let r = |mb, cache| ReadReport {
            mb_per_s: mb,
            bytes: 1,
            seconds: 1.0,
            cache,
        };
        assert_eq!(r(6000.0, CacheState::Cold).class(), DiskClass::Fast);
        assert_eq!(r(6000.0, CacheState::Unknown).class(), DiskClass::Fast);
        assert_eq!(
            r(6000.0, CacheState::MaybeCached(0.9)).class(),
            DiskClass::Fair
        );
        assert_eq!(
            r(6000.0, CacheState::MaybeCached(0.2)).class(),
            DiskClass::Fast,
            "a small share in memory does not change the class"
        );
        assert_eq!(
            r(300.0, CacheState::MaybeCached(0.9)).class(),
            DiskClass::Slow
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_detector_sees_a_file_the_system_has_just_read() {
        use std::io::Write;
        let d = Scratch::new("warm");
        let path = d.0.join("w.bin");
        let mut f = File::create(&path).unwrap();
        // real data, not a hole: written, flushed, then read back so the pages are cached
        let chunk = vec![0xA5u8; 1 << 20];
        for _ in 0..(MIN_FILE >> 20) {
            f.write_all(&chunk).unwrap();
        }
        f.sync_all().unwrap();
        drop(f);
        let all = fs::read(&path).unwrap();
        assert_eq!(all.len() as u64, MIN_FILE);
        let f = File::open(&path).unwrap();
        let map = pagecache::Map::new(&f, MIN_FILE).expect("a regular file can be mapped");
        let (hit, total) = map.resident(0, BLOCK).unwrap();
        assert!(
            hit * 10 >= total * 9,
            "a file that was just read must show as cached: {hit}/{total}"
        );
        // the range is checked: out of the mapping, not page aligned, or empty
        assert!(map.resident(MIN_FILE as usize, BLOCK).is_none());
        assert!(map.resident(1, 4096).is_none());
        assert!(map.resident(0, 0).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_real_test_reports_where_the_blocks_came_from() {
        use std::io::Write;
        let d = Scratch::new("report");
        let mut f = File::create(d.0.join("r.bin")).unwrap();
        let chunk = vec![0x5Au8; 1 << 20];
        for _ in 0..(MIN_FILE >> 20) {
            f.write_all(&chunk).unwrap();
        }
        f.sync_all().unwrap();
        drop(f);
        let r = random_read_report(&d.0, Duration::from_secs(30)).unwrap();
        assert!(r.mb_per_s.is_finite() && r.mb_per_s > 0.0 && r.bytes == MIN_FILE);
        assert!(
            !matches!(r.cache, CacheState::Unknown),
            "Linux can be asked: {r:?}"
        );
    }

    #[test]
    fn the_class_floors_classify_into_their_own_class() {
        for c in [DiskClass::Fast, DiskClass::Fair, DiskClass::Slow] {
            assert_eq!(classify(c.floor_mbs()), c);
        }
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
