//! Knowing when colibri (and this app) have something new, so a new model family or a speed-up
//! upstream is not missed. The check is one small request to the GitHub API, made through
//! Python (which colibri needs anyway), only when the person has left update checks on.

use crate::locate::PythonCmd;
use crate::proc::{hide_window, run_capture};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const COLIBRI_COMMIT_API: &str = "https://api.github.com/repos/JustVugg/colibri/commits/main";
pub const COMPANION_RELEASES_API: &str =
    "https://api.github.com/repos/chiuta/colibri-companion/releases?per_page=10";
pub const COMPANION_RELEASES_PAGE: &str = "https://github.com/chiuta/colibri-companion/releases";
const MARKER: &str = ".colibri-companion-fetched";

const GET_SCRIPT: &str = r#"
import sys, urllib.request
req = urllib.request.Request(sys.argv[1], headers={"User-Agent": "colibri-companion", "Accept": "application/vnd.github+json"})
with urllib.request.urlopen(req, timeout=15) as r:
    sys.stdout.write(r.read().decode("utf-8", "replace"))
"#;

/// One GET, through Python. Returns the body.
pub fn http_get(python: &PythonCmd, url: &str) -> Result<String, String> {
    let mut c = python.command();
    c.args(["-I", "-c", GET_SCRIPT, url]);
    hide_window(&mut c);
    let out = run_capture(c, Duration::from_secs(25)).map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        Err(err
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("request failed")
            .to_string())
    }
}

// ------------------------------------------------------------------ colibri

#[derive(Clone, Debug, PartialEq)]
pub struct Upstream {
    pub sha: String,
    /// `YYYY-MM-DDTHH:MM:SSZ`
    pub date: String,
    /// First line of the commit message.
    pub title: String,
}

pub fn parse_commit(json: &str) -> Option<Upstream> {
    #[derive(Deserialize)]
    struct Person {
        date: Option<String>,
    }
    #[derive(Deserialize)]
    struct Inner {
        message: Option<String>,
        committer: Option<Person>,
    }
    #[derive(Deserialize)]
    struct Outer {
        sha: String,
        commit: Inner,
    }
    let o: Outer = serde_json::from_str(json).ok()?;
    Some(Upstream {
        sha: o.sha,
        date: o.commit.committer.and_then(|c| c.date).unwrap_or_default(),
        title: o
            .commit
            .message
            .unwrap_or_default()
            .lines()
            .next()
            .unwrap_or("")
            .to_string(),
    })
}

/// The commit a git checkout is at, read from `.git` without running git.
pub fn git_head(root: &Path) -> Option<String> {
    let head = fs::read_to_string(root.join(".git").join("HEAD")).ok()?;
    let head = head.trim();
    match head.strip_prefix("ref: ") {
        None => Some(head.to_string()),
        Some(r) => {
            if let Ok(s) = fs::read_to_string(root.join(".git").join(r)) {
                return Some(s.trim().to_string());
            }
            let packed = fs::read_to_string(root.join(".git").join("packed-refs")).ok()?;
            packed
                .lines()
                .find_map(|l| l.strip_suffix(r).map(|s| s.trim().to_string()))
        }
    }
}

/// Remember when a ZIP copy was fetched (a ZIP has no git history to ask).
pub fn write_fetched_marker(root: &Path) {
    let _ = fs::write(root.join(MARKER), iso_now());
}

fn fetched_marker(root: &Path) -> Option<String> {
    let t = fs::read_to_string(root.join(MARKER)).ok()?;
    let t = t.trim();
    (t.len() == 20).then(|| t.to_string())
}

/// Is `upstream` newer than the copy of colibri in `root`? `None` when that cannot be known.
pub fn colibri_is_outdated(root: &Path, upstream: &Upstream) -> Option<bool> {
    if let Some(sha) = git_head(root) {
        return Some(sha != upstream.sha);
    }
    let fetched = fetched_marker(root)?;
    (!upstream.date.is_empty()).then(|| upstream.date.as_str() > fetched.as_str())
}

pub fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    iso_from_secs(secs)
}

/// UTC civil time from Unix seconds (Howard Hinnant's algorithm), without a date library.
pub fn iso_from_secs(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

// ------------------------------------------------------------------ this app

#[derive(Clone, Debug, PartialEq)]
pub struct Release {
    pub tag: String,
    pub url: String,
    pub prerelease: bool,
}

/// The newest published (not draft) release in the list the API returns.
pub fn parse_latest_release(json: &str) -> Option<Release> {
    #[derive(Deserialize)]
    struct R {
        tag_name: String,
        html_url: String,
        #[serde(default)]
        draft: bool,
        #[serde(default)]
        prerelease: bool,
    }
    let list: Vec<R> = serde_json::from_str(json).ok()?;
    list.into_iter().find(|r| !r.draft).map(|r| Release {
        tag: r.tag_name,
        url: r.html_url,
        prerelease: r.prerelease,
    })
}

/// `1.2.3-alpha.4` -> ((1,2,3), pre-release number or MAX for a final release).
fn version_key(v: &str) -> Option<((u64, u64, u64), u64)> {
    let v = v.trim().trim_start_matches('v');
    let (core, pre) = match v.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (v, None),
    };
    let mut it = core.split('.').map(|n| n.parse::<u64>());
    let key = (it.next()?.ok()?, it.next()?.ok()?, it.next()?.ok()?);
    let pre = match pre {
        None => u64::MAX,
        Some(p) => p
            .rsplit('.')
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(0),
    };
    Some((key, pre))
}

/// Is `latest` a newer version than `current`? A bare `0.1.0` build counts as the first
/// alpha of 0.1.0 would: this app ships as 0.1.0-alpha.N until it has a final release.
pub fn is_newer(latest: &str, current: &str) -> bool {
    match (version_key(latest), version_key(current)) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_commit_api() {
        let u = parse_commit(
            r#"{"sha":"abc123","commit":{"message":"Add a model\n\nlong text","committer":{"date":"2026-11-14T09:30:00Z"}}}"#,
        )
        .unwrap();
        assert_eq!(u.sha, "abc123");
        assert_eq!(u.title, "Add a model");
        assert_eq!(u.date, "2026-11-14T09:30:00Z");
        assert!(parse_commit("{}").is_none());
    }

    #[test]
    fn decides_whether_colibri_is_outdated() {
        let base = std::env::temp_dir().join(format!("cc-upd-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let up = Upstream {
            sha: "new".into(),
            date: "2026-11-14T09:30:00Z".into(),
            title: String::new(),
        };
        // a git checkout compares commits
        fs::create_dir_all(base.join("g/.git/refs/heads")).unwrap();
        fs::write(base.join("g/.git/HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(base.join("g/.git/refs/heads/main"), "old\n").unwrap();
        assert_eq!(colibri_is_outdated(&base.join("g"), &up), Some(true));
        fs::write(base.join("g/.git/refs/heads/main"), "new\n").unwrap();
        assert_eq!(colibri_is_outdated(&base.join("g"), &up), Some(false));
        // a ZIP copy compares the fetch time with the upstream commit time
        fs::create_dir_all(base.join("z")).unwrap();
        assert_eq!(colibri_is_outdated(&base.join("z"), &up), None);
        fs::write(base.join("z").join(MARKER), "2026-11-10T00:00:00Z").unwrap();
        assert_eq!(colibri_is_outdated(&base.join("z"), &up), Some(true));
        fs::write(base.join("z").join(MARKER), "2026-11-15T00:00:00Z").unwrap();
        assert_eq!(colibri_is_outdated(&base.join("z"), &up), Some(false));
        // packed refs
        fs::create_dir_all(base.join("p/.git")).unwrap();
        fs::write(base.join("p/.git/HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(
            base.join("p/.git/packed-refs"),
            "# pack\nfeed refs/heads/main\n",
        )
        .unwrap();
        assert_eq!(git_head(&base.join("p")).as_deref(), Some("feed"));
    }

    #[test]
    fn dates_are_written_in_utc() {
        assert_eq!(iso_from_secs(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_from_secs(1_794_787_200), "2026-11-16T00:00:00Z");
        assert_eq!(iso_now().len(), 20);
    }

    #[test]
    fn compares_versions() {
        assert!(is_newer("v0.1.0-alpha.2", "0.1.0-alpha.1"));
        assert!(is_newer("v0.1.0", "0.1.0-alpha.9"));
        assert!(is_newer("v0.2.0-alpha.1", "0.1.0"));
        assert!(!is_newer("v0.1.0-alpha.1", "0.1.0-alpha.1"));
        assert!(!is_newer("v0.1.0-alpha.1", "0.1.0"));
        assert!(!is_newer("nonsense", "0.1.0"));
    }

    #[test]
    fn picks_the_newest_published_release() {
        let json = r#"[{"tag_name":"v0.2.0","html_url":"u2","draft":true,"prerelease":true},
                       {"tag_name":"v0.1.0-alpha.1","html_url":"u1","draft":false,"prerelease":true}]"#;
        let r = parse_latest_release(json).unwrap();
        assert_eq!((r.tag.as_str(), r.prerelease), ("v0.1.0-alpha.1", true));
        assert!(parse_latest_release("[]").is_none());
    }
}
