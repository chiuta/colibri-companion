//! Getting colibri onto a computer that does not have it, without needing git: Python
//! (which colibri needs anyway) downloads the project's ZIP from GitHub and unpacks it.
//! colibri is Apache-2.0; its files, including the LICENSE, are placed unchanged.

use crate::locate::PythonCmd;
use crate::proc::hide_window;
use std::io;
use std::path::Path;
use std::process::{Child, Stdio};

/// Largest download accepted (the real archive is a few megabytes) and, in the same unit, the
/// factor by which unpacking may grow it. A wrong or hostile address cannot fill the disk.
pub const MAX_ZIP_BYTES: u64 = 1 << 30;

/// Run as `python -I -c SCRIPT URL DEST MAX_BYTES`. It prints `P <done> <total>` lines while
/// downloading, then `OK` once `DEST` holds the colibri folder; failures go to stderr.
const SCRIPT: &str = r#"
import os, shutil, sys, tempfile, time, urllib.request, zipfile
url, dest, limit = sys.argv[1], os.path.abspath(sys.argv[2]), int(sys.argv[3])
parent = os.path.dirname(dest)
os.makedirs(parent, exist_ok=True)
class Https(urllib.request.HTTPRedirectHandler):
    # an archive that is later run as code never arrives over a plain-http hop
    def redirect_request(self, r, fp, code, msg, headers, newurl):
        if url.startswith("https://") and not newurl.lower().startswith("https://"):
            raise RuntimeError("redirected away from https")
        return urllib.request.HTTPRedirectHandler.redirect_request(self, r, fp, code, msg, headers, newurl)
opener = urllib.request.build_opener(Https)
started = time.monotonic()
work = tempfile.mkdtemp(prefix=".colibri-fetch-", dir=parent)
try:
    archive = os.path.join(work, "colibri.zip")
    req = urllib.request.Request(url, headers={"User-Agent": "colibri-companion"})
    try:
        with opener.open(req, timeout=60) as r, open(archive, "wb") as out:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(262144)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if done > limit:
                    raise RuntimeError("the download is larger than expected")
                if time.monotonic() - started > 1800:
                    raise RuntimeError("the download is taking too long")
                print("P", done, total, flush=True)
    except Exception as first:
        # Some Python installs (python.org on macOS) have no certificates: curl uses the system's.
        curl = shutil.which("curl")
        if not curl or url.startswith("file:") or "larger than expected" in str(first) or "too long" in str(first):
            raise
        import subprocess
        print("P", 0, 0, flush=True)
        r = subprocess.run([curl, "-fsSL", "--retry", "2", "--proto", "=https", "--proto-redir", "=https",
                            "--max-time", "1800", "--max-filesize", str(limit), "-o", archive, "--", url])
        if r.returncode != 0:
            raise RuntimeError("%s (curl also failed: %d)" % (first, r.returncode))
        # --max-filesize only counts when the server announces a size: check what arrived
        if os.path.getsize(archive) > limit:
            raise RuntimeError("the download is larger than expected")
    print("U", flush=True)
    with zipfile.ZipFile(archive) as z:
        # sizes first: testing the archive unpacks every member in memory
        infos = z.infolist()
        if len(infos) > 200000 or sum(i.file_size for i in infos) > limit * 4:
            raise RuntimeError("the archive is larger than expected")
        bad = z.testzip()
        if bad:
            raise RuntimeError("the download is damaged (%s)" % bad)
        z.extractall(os.path.join(work, "x"))
    top = os.path.join(work, "x")
    entries = [e for e in os.listdir(top)]
    if len(entries) == 1 and os.path.isdir(os.path.join(top, entries[0])):
        top = os.path.join(top, entries[0])
    if not os.path.isfile(os.path.join(top, "c", "coli")):
        raise RuntimeError("the archive does not look like colibri")
    if os.path.exists(dest):
        raise RuntimeError("%s already exists" % dest)
    shutil.move(top, dest)
    print("OK", flush=True)
finally:
    shutil.rmtree(work, ignore_errors=True)
"#;

/// What the downloader reported on one line of its output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FetchLine {
    Progress { done: u64, total: u64 },
    Unpacking,
    Done,
}

pub fn parse_line(line: &str) -> Option<FetchLine> {
    let line = line.trim();
    if line == "OK" {
        return Some(FetchLine::Done);
    }
    if line == "U" {
        return Some(FetchLine::Unpacking);
    }
    let mut it = line.strip_prefix("P ")?.split_whitespace();
    Some(FetchLine::Progress {
        done: it.next()?.parse().ok()?,
        total: it.next()?.parse().ok()?,
    })
}

/// Starts the download; read `stdout` for [`FetchLine`]s and `stderr` for the reason of a failure.
pub fn spawn_zip_download(python: &PythonCmd, url: &str, dest: &Path) -> io::Result<Child> {
    spawn_zip_download_capped(python, url, dest, MAX_ZIP_BYTES)
}

/// [`spawn_zip_download`] with an explicit limit on the size of the download.
pub fn spawn_zip_download_capped(
    python: &PythonCmd,
    url: &str,
    dest: &Path,
    max_bytes: u64,
) -> io::Result<Child> {
    // Code that will be run later: only over https (a local `file:` copy only in this crate's own tests).
    let lower = url.to_ascii_lowercase();
    let local_test = cfg!(test) && lower.starts_with("file:");
    if !(lower.starts_with("https://") || local_test)
        || url.chars().any(|c| c.is_control() || c.is_whitespace())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the archive address must be a plain https address",
        ));
    }
    let mut c = python.command();
    c.args(["-I", "-c", SCRIPT, url])
        .arg(dest)
        .arg(max_bytes.to_string());
    hide_window(&mut c);
    let child = c
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    crate::net::record_egress(url, "GET", "download", child.is_ok());
    child
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(windows))]
    use std::io::{BufRead, BufReader, Read};

    #[test]
    fn parses_progress_lines() {
        assert_eq!(
            parse_line("P 10 200\n"),
            Some(FetchLine::Progress {
                done: 10,
                total: 200
            })
        );
        assert_eq!(parse_line("U"), Some(FetchLine::Unpacking));
        assert_eq!(parse_line(" OK "), Some(FetchLine::Done));
        assert_eq!(parse_line("P x y"), None);
        assert_eq!(parse_line("hello"), None);
    }

    /// The whole path, with a file:// URL: download, check, unpack, rename.
    #[cfg(not(windows))]
    #[test]
    fn downloads_and_unpacks_a_zip() {
        let Some(python) = crate::locate::find_python() else {
            eprintln!("python >= 3.10 not available: skipping");
            return;
        };
        let base = std::env::temp_dir().join(format!("cc-fetch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let zip = base.join("src.zip");
        let mut mk = python.command();
        mk.args([
            "-c",
            "import sys, zipfile\nz = zipfile.ZipFile(sys.argv[1], 'w')\n\
             z.writestr('colibri-main/c/coli', '#!/usr/bin/env python3')\n\
             z.writestr('colibri-main/c/setup_flow.py', '')\n\
             z.writestr('colibri-main/LICENSE', 'Apache')\nz.close()",
        ])
        .arg(&zip);
        assert!(mk.status().unwrap().success());
        let url = format!("file://{}", zip.display());
        let dest = base.join("out").join("colibri");
        let mut child = spawn_zip_download(&python, &url, &dest).unwrap();
        let mut lines = Vec::new();
        for l in BufReader::new(child.stdout.take().unwrap()).lines() {
            lines.extend(parse_line(&l.unwrap()));
        }
        let mut err = String::new();
        child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut err)
            .unwrap();
        assert!(child.wait().unwrap().success(), "{err}");
        assert_eq!(lines.last(), Some(&FetchLine::Done));
        assert!(crate::locate::is_colibri_root(&dest));
        assert_eq!(
            std::fs::read_to_string(dest.join("LICENSE")).unwrap(),
            "Apache"
        );
        // nothing is left behind next to it, and a second run does not overwrite
        let leftovers: Vec<_> = std::fs::read_dir(dest.parent().unwrap())
            .unwrap()
            .flatten()
            .collect();
        assert_eq!(leftovers.len(), 1);
        let mut again = spawn_zip_download(&python, &url, &dest).unwrap();
        assert!(!again.wait().unwrap().success());
        assert!(crate::locate::is_colibri_root(&dest));
    }

    /// A download or archive bigger than the limit is stopped, and nothing is left behind.
    #[cfg(not(windows))]
    #[test]
    fn an_oversized_archive_is_refused() {
        use std::io::Read;
        let Some(python) = crate::locate::find_python() else {
            return;
        };
        let base = std::env::temp_dir().join(format!("cc-fetch-big-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let zip = base.join("big.zip");
        let mut mk = python.command();
        mk.args([
            "-c",
            "import sys, zipfile\nz = zipfile.ZipFile(sys.argv[1], 'w', zipfile.ZIP_DEFLATED)\n\
             z.writestr('colibri-main/c/coli', 'x' * 100000)\nz.close()",
        ])
        .arg(&zip);
        assert!(mk.status().unwrap().success());
        let url = format!("file://{}", zip.display());
        let dest = base.join("out").join("colibri");
        // the archive is small but unpacks past 4x the limit
        let mut child = spawn_zip_download_capped(&python, &url, &dest, 1000).unwrap();
        let mut err = String::new();
        child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut err)
            .unwrap();
        assert!(!child.wait().unwrap().success());
        assert!(err.contains("larger than expected"), "{err}");
        assert!(!dest.exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn only_plain_https_addresses_are_downloaded() {
        let python = PythonCmd {
            program: "python3".into(),
            pre_args: vec![],
            version: (3, 12),
        };
        let dest = std::env::temp_dir().join("cc-never-made");
        for bad in [
            "http://github.com/x.zip",
            "ftp://x/y.zip",
            "https://x/y z.zip",
            "https://x/\n.zip",
            "",
            "-o/etc/x",
        ] {
            let e = spawn_zip_download(&python, bad, &dest).unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::InvalidInput, "{bad:?}");
        }
        assert!(!dest.exists());
    }
}
