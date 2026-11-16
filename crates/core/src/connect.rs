//! Reaching the internet and online platforms (GitHub, GitLab, Gitea/Codeberg, plain web pages).
//!
//! The rest of the program only ever talks to the colibri server on this computer, with its own
//! tiny `http://` client. Online access is separate and **off until the person turns it on**.
//! It goes through the Python the program already uses for downloads (`urllib`, with the
//! system's certificate store), so no new library is added. Tokens travel in environment
//! variables, never on a command line where other programs could read them.

use crate::locate::PythonCmd;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::Path;

pub const MAX_BODY: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Github,
    Gitlab,
    /// Gitea, Forgejo, Codeberg.
    Gitea,
    /// Any address that answers to a bearer token (or none).
    Generic,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Github, Kind::Gitlab, Kind::Gitea, Kind::Generic];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Github => "GitHub",
            Kind::Gitlab => "GitLab",
            Kind::Gitea => "Gitea / Forgejo / Codeberg",
            Kind::Generic => "Web / API",
        }
    }

    pub fn default_base(self) -> &'static str {
        match self {
            Kind::Github => "https://api.github.com",
            Kind::Gitlab => "https://gitlab.com/api/v4",
            Kind::Gitea => "https://codeberg.org/api/v1",
            Kind::Generic => "https://",
        }
    }

    /// Where the person makes a token, for the help text.
    pub fn token_page(self) -> &'static str {
        match self {
            Kind::Github => "https://github.com/settings/personal-access-tokens",
            Kind::Gitlab => "https://gitlab.com/-/user_settings/personal_access_tokens",
            Kind::Gitea => "https://codeberg.org/user/settings/applications",
            Kind::Generic => "",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub method: &'static str,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectError {
    BadUrl(&'static str),
    NeedsPython,
    Http(u16, String),
    Network(String),
    TooBig,
    Parse(String),
}

/// Only https, or http to a machine on the local network (a self-hosted server), never an
/// address with a password in it.
pub fn check_url(url: &str) -> Result<(), ConnectError> {
    let u = url.trim();
    let (secure, rest) = if let Some(r) = u.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = u.strip_prefix("http://") {
        (false, r)
    } else {
        return Err(ConnectError::BadUrl("the address must start with https://"));
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.is_empty() || authority.contains(char::is_whitespace) {
        return Err(ConnectError::BadUrl("the address has no host"));
    }
    if authority.contains('@') {
        return Err(ConnectError::BadUrl(
            "an address with a user name or password in it is refused",
        ));
    }
    if !secure && !crate::net::is_local_base(u) && !is_private_lan(u) {
        return Err(ConnectError::BadUrl(
            "plain http is only allowed for servers on this computer or network",
        ));
    }
    Ok(())
}

/// A server on the home or office network (10.x, 172.16–31.x, 192.168.x), where plain http is
/// as safe as it gets for a self-hosted tool.
fn is_private_lan(url: &str) -> bool {
    crate::http::parse_authority(url).is_ok_and(|(host, _)| {
        host.parse::<std::net::Ipv4Addr>()
            .is_ok_and(|ip| ip.is_private())
    })
}

pub fn host_of(url: &str) -> String {
    let rest = url.split("://").nth(1).unwrap_or(url);
    rest.split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .rsplit('@')
        .next()
        .unwrap_or("")
        .to_string()
}

pub fn percent_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Like `percent_encode`, but keeps `/` so a path stays a path. `.` and `..` steps are encoded
/// too, so a typed path cannot climb out of the repository.
fn encode_path(s: &str) -> String {
    s.split('/')
        .map(|p| {
            if p == "." || p == ".." {
                p.replace('.', "%2E")
            } else {
                percent_encode(p)
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

const AGENT: &str = "colibri-companion";

fn auth_headers(kind: Kind, token: &str) -> Vec<(String, String)> {
    let mut h = vec![("User-Agent".to_string(), AGENT.to_string())];
    let t = token.trim();
    match kind {
        Kind::Github => {
            h.push(("Accept".into(), "application/vnd.github+json".into()));
            h.push(("X-GitHub-Api-Version".into(), "2022-11-28".into()));
            if !t.is_empty() {
                h.push(("Authorization".into(), format!("Bearer {t}")));
            }
        }
        Kind::Gitlab => {
            if !t.is_empty() {
                h.push(("PRIVATE-TOKEN".into(), t.to_string()));
            }
        }
        Kind::Gitea => {
            if !t.is_empty() {
                h.push(("Authorization".into(), format!("token {t}")));
            }
        }
        Kind::Generic => {
            if !t.is_empty() {
                h.push(("Authorization".into(), format!("Bearer {t}")));
            }
        }
    }
    h
}

/// What can be asked of a code platform. `repo` is `owner/name` (GitLab: `group/project`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op<'a> {
    Whoami,
    Repos,
    Tree { repo: &'a str, path: &'a str },
    File { repo: &'a str, path: &'a str },
    Issues { repo: &'a str },
}

pub fn build(kind: Kind, base: &str, token: &str, op: &Op) -> Request {
    let base = base.trim().trim_end_matches('/');
    // `owner/name` goes into the address: each part is encoded so a typed name cannot add a
    // query, a fragment or a `..` step.
    let enc_repo = |r: &str| encode_path(r.trim_matches('/'));
    let mut headers = auth_headers(kind, token);
    let url = match (kind, op) {
        (_, Op::Whoami) => format!("{base}/user"),
        (Kind::Github | Kind::Gitea, Op::Repos) => {
            format!("{base}/user/repos?per_page=100&sort=updated")
        }
        (Kind::Gitlab, Op::Repos) => format!(
            "{base}/projects?membership=true&per_page=100&order_by=last_activity_at&simple=true"
        ),
        (Kind::Github | Kind::Gitea, Op::Tree { repo, path }) => {
            // No trailing slash for the top folder: some gateways refuse it.
            let p = encode_path(path.trim_matches('/'));
            if p.is_empty() {
                format!("{base}/repos/{}/contents", enc_repo(repo))
            } else {
                format!("{base}/repos/{}/contents/{p}", enc_repo(repo))
            }
        }
        (Kind::Gitlab, Op::Tree { repo, path }) => format!(
            "{base}/projects/{}/repository/tree?path={}&per_page=100",
            percent_encode(repo),
            percent_encode(path)
        ),
        (Kind::Github, Op::File { repo, path }) => {
            headers.retain(|(k, _)| k != "Accept");
            headers.push(("Accept".into(), "application/vnd.github.raw+json".into()));
            format!(
                "{base}/repos/{}/contents/{}",
                enc_repo(repo),
                encode_path(path)
            )
        }
        (Kind::Gitea, Op::File { repo, path }) => {
            format!("{base}/repos/{}/raw/{}", enc_repo(repo), encode_path(path))
        }
        (Kind::Gitlab, Op::File { repo, path }) => format!(
            "{base}/projects/{}/repository/files/{}/raw?ref=HEAD",
            percent_encode(repo),
            percent_encode(path)
        ),
        (Kind::Github | Kind::Gitea, Op::Issues { repo }) => {
            format!(
                "{base}/repos/{}/issues?state=open&per_page=30",
                enc_repo(repo)
            )
        }
        (Kind::Gitlab, Op::Issues { repo }) => format!(
            "{base}/projects/{}/issues?state=opened&per_page=30",
            percent_encode(repo)
        ),
        (Kind::Generic, _) => base.to_string(),
    };
    Request {
        method: "GET",
        url,
        headers,
        body: None,
    }
}

/// A plain page or file, with no token unless one is given.
pub fn page(url: &str, token: &str) -> Request {
    let mut headers = auth_headers(Kind::Generic, token);
    headers.push((
        "Accept".into(),
        "text/html,text/plain,application/json,*/*;q=0.5".into(),
    ));
    Request {
        method: "GET",
        url: url.trim().to_string(),
        headers,
        body: None,
    }
}

/// A message to a webhook (Slack, Discord, n8n, Home Assistant, anything that takes JSON).
pub fn webhook(url: &str, json_body: &str) -> Request {
    Request {
        method: "POST",
        url: url.trim().to_string(),
        headers: vec![
            ("User-Agent".into(), AGENT.into()),
            ("Content-Type".into(), "application/json".into()),
        ],
        body: Some(json_body.to_string()),
    }
}

const SCRIPT: &str = r#"
import os, sys, json, ssl, urllib.request, urllib.error, urllib.parse
req = json.loads(sys.stdin.buffer.read().decode("utf-8"))
limit = int(os.environ["CC_LIMIT"])
import ipaddress
def inner(url):
    h = urllib.parse.urlsplit(url).hostname or ""
    if h.lower() == "localhost" or h.lower().endswith(".localhost"):
        return True
    try:
        a = ipaddress.ip_address(h)
    except ValueError:
        return False
    return a.is_loopback or a.is_private or a.is_link_local or a.is_unspecified or a.is_reserved
class Same(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, r, fp, code, msg, headers, newurl):
        n = urllib.request.HTTPRedirectHandler.redirect_request(self, r, fp, code, msg, headers, newurl)
        if n is None:
            return None
        if urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(r.full_url).netloc:
            keep = {"user-agent", "accept"}
            for k in list(n.headers):
                if k.lower() not in keep:
                    del n.headers[k]
            for k in list(n.unredirected_hdrs):
                del n.unredirected_hdrs[k]
        if not newurl.startswith("https://") and not r.full_url.startswith("http://"):
            return None
        # A public page must not send the program to this computer or this network.
        if inner(newurl) and not inner(r.full_url):
            return None
        return n
data = req["body"].encode("utf-8") if req.get("body") else None
r = urllib.request.Request(req["url"], data=data, method=req["method"])
for k, v in req["headers"]:
    r.add_header(k, v)
op = urllib.request.build_opener(Same, urllib.request.HTTPSHandler(context=ssl.create_default_context()))
try:
    with op.open(r, timeout=30) as resp:
        body = resp.read(limit + 1)
        sys.stdout.buffer.write(body)
except urllib.error.HTTPError as e:
    sys.stderr.write("HTTP %d %s" % (e.code, e.reason))
    sys.exit(2)
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(3)
"#;

/// A header with a line break in it would split the request in two.
fn clean_headers(req: &Request) -> Result<(), ConnectError> {
    for (k, v) in &req.headers {
        if k.chars().chain(v.chars()).any(char::is_control) {
            return Err(ConnectError::BadUrl(
                "a header (the token?) contains a line break or control character",
            ));
        }
    }
    Ok(())
}

/// Takes secrets out of an error message before it is shown.
fn redact(mut text: String, req: &Request) -> String {
    for (k, v) in &req.headers {
        let kl = k.to_ascii_lowercase();
        if matches!(kl.as_str(), "authorization" | "private-token") {
            let secret = v.rsplit(' ').next().unwrap_or(v);
            if secret.len() >= 4 {
                text = text.replace(secret, "…");
            }
        }
    }
    text
}

fn request_json(req: &Request) -> String {
    serde_json::json!({
        "method": req.method,
        "url": req.url,
        "headers": req.headers,
        "body": req.body,
    })
    .to_string()
}

/// Sends the request and returns the body. Needs the person to have switched online access on:
/// callers check that; this only validates the address.
pub fn send(python: Option<&PythonCmd>, req: &Request) -> Result<Vec<u8>, ConnectError> {
    check_url(&req.url)?;
    clean_headers(req)?;
    let python = python.ok_or(ConnectError::NeedsPython)?;
    let mut cmd = python.command();
    cmd.arg("-I")
        .arg("-c")
        .arg(SCRIPT)
        .env("CC_LIMIT", MAX_BODY.to_string());
    // On standard input, not in the environment: no size limit, and never visible to other programs.
    let out = crate::proc::run_capture_input(
        cmd,
        Some(request_json(req).into_bytes()),
        std::time::Duration::from_secs(120),
    )
    .map_err(|e| ConnectError::Network(redact(e.to_string(), req)))?;
    if out.status.success() {
        if out.stdout.len() > MAX_BODY {
            return Err(ConnectError::TooBig);
        }
        return Ok(out.stdout);
    }
    let err = redact(String::from_utf8_lossy(&out.stderr).trim().to_string(), req);
    if let Some(rest) = err.strip_prefix("HTTP ") {
        let code = rest
            .split_whitespace()
            .next()
            .and_then(|c| c.parse().ok())
            .unwrap_or(0);
        return Err(ConnectError::Http(code, rest.to_string()));
    }
    Err(ConnectError::Network(err))
}

pub fn explain(e: &ConnectError, romanian: bool) -> String {
    let (ro, en): (String, String) = match e {
        ConnectError::BadUrl(m) => (
            format!("Adresă refuzată: {m}"),
            format!("Address refused: {m}"),
        ),
        ConnectError::NeedsPython => (
            "Îmi trebuie Python ca să ajung pe internet.".into(),
            "I need Python to reach the internet.".into(),
        ),
        ConnectError::Http(401 | 403, _) => (
            "Acces refuzat: tokenul lipsește, a expirat sau nu are dreptul.".into(),
            "Access refused: the token is missing, expired or not allowed.".into(),
        ),
        ConnectError::Http(404, _) => (
            "Nu am găsit ce am cerut (adresă sau nume greșit, sau depozit privat).".into(),
            "Not found (wrong address or name, or a private repository).".into(),
        ),
        ConnectError::Http(429, _) => (
            "Prea multe cereri: așteaptă puțin și încearcă din nou.".into(),
            "Too many requests: wait a little and try again.".into(),
        ),
        ConnectError::Http(_, m) => (
            format!("Serverul a răspuns: {m}"),
            format!("The server answered: {m}"),
        ),
        ConnectError::Network(m) => (
            format!("Nu am ajuns la server: {m}"),
            format!("Could not reach the server: {m}"),
        ),
        ConnectError::TooBig => (
            "Răspunsul e prea mare (peste 8 MB).".into(),
            "The answer is too big (over 8 MB).".into(),
        ),
        ConnectError::Parse(m) => (
            format!("Răspuns neașteptat: {m}"),
            format!("Unexpected answer: {m}"),
        ),
    };
    if romanian { ro } else { en }
}

// ------------------------------------------------------------------ reading the answers

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repo {
    /// What goes back into later requests (`owner/name`, or the GitLab path).
    pub id: String,
    pub description: String,
    pub private: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Issue {
    pub number: u64,
    pub title: String,
    pub author: String,
    pub body: String,
}

fn json(bytes: &[u8]) -> Result<Value, ConnectError> {
    serde_json::from_slice(bytes).map_err(|e| ConnectError::Parse(e.to_string()))
}

fn s(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

pub fn parse_repos(kind: Kind, bytes: &[u8]) -> Result<Vec<Repo>, ConnectError> {
    let v = json(bytes)?;
    let list = v
        .as_array()
        .ok_or_else(|| ConnectError::Parse("not a list".into()))?;
    Ok(list
        .iter()
        .filter_map(|r| {
            let id = if kind == Kind::Gitlab {
                s(r, "path_with_namespace")
            } else {
                s(r, "full_name")
            };
            if id.is_empty() {
                return None;
            }
            let private = if kind == Kind::Gitlab {
                s(r, "visibility") == "private"
            } else {
                r.get("private").and_then(Value::as_bool).unwrap_or(false)
            };
            Some(Repo {
                id,
                description: s(r, "description"),
                private,
            })
        })
        .collect())
}

pub fn parse_tree(kind: Kind, bytes: &[u8]) -> Result<Vec<Entry>, ConnectError> {
    let v = json(bytes)?;
    let list = v
        .as_array()
        .ok_or_else(|| ConnectError::Parse("this is a file, not a folder".into()))?;
    let mut out: Vec<Entry> = list
        .iter()
        .map(|e| {
            let t = s(e, "type");
            Entry {
                name: s(e, "name"),
                path: s(e, "path"),
                is_dir: t == "dir" || t == "tree",
                size: e.get("size").and_then(Value::as_u64),
            }
        })
        .filter(|e| !e.name.is_empty())
        .collect();
    let _ = kind;
    out.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
    Ok(out)
}

pub fn parse_issues(kind: Kind, bytes: &[u8]) -> Result<Vec<Issue>, ConnectError> {
    let v = json(bytes)?;
    let list = v
        .as_array()
        .ok_or_else(|| ConnectError::Parse("not a list".into()))?;
    Ok(list
        .iter()
        // GitHub lists pull requests among issues; they carry a `pull_request` key.
        .filter(|i| i.get("pull_request").is_none())
        .map(|i| {
            let number = if kind == Kind::Gitlab {
                i.get("iid")
            } else {
                i.get("number")
            }
            .and_then(Value::as_u64)
            .unwrap_or(0);
            let author = if kind == Kind::Gitlab {
                i.pointer("/author/name")
            } else {
                i.pointer("/user/login")
            }
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
            let body = if kind == Kind::Gitlab {
                s(i, "description")
            } else {
                s(i, "body")
            };
            Issue {
                number,
                title: s(i, "title"),
                author,
                body,
            }
        })
        .collect())
}

pub fn issues_text(repo: &str, issues: &[Issue]) -> String {
    let mut out = format!("Open issues of {repo}\n\n");
    for i in issues {
        out.push_str(&format!("#{} {} (by {})\n", i.number, i.title, i.author));
        let b = i.body.trim();
        if !b.is_empty() {
            let cut: String = b.chars().take(600).collect();
            out.push_str(&cut);
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

/// The readable text of a downloaded page: HTML is stripped, JSON and text are kept.
pub fn page_text(bytes: &[u8]) -> Result<String, ConnectError> {
    if bytes.contains(&0) {
        return Err(ConnectError::Parse("this is not a text page".into()));
    }
    let raw = String::from_utf8_lossy(bytes).into_owned();
    let head: String = raw.chars().take(400).collect::<String>().to_lowercase();
    if head.contains("<html") || head.contains("<!doctype html") || head.contains("<body") {
        Ok(crate::attach::html_to_text(&raw))
    } else {
        Ok(raw)
    }
}

// ------------------------------------------------------------------ saved connections

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Service {
    pub name: String,
    pub kind: Kind,
    pub base: String,
    /// Only filled when the person chose to keep it on this computer.
    #[serde(default)]
    pub token: String,
    #[serde(default)]
    pub remember: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Connections {
    /// Online access is off until the person turns it on.
    #[serde(default)]
    pub internet: bool,
    #[serde(default)]
    pub services: Vec<Service>,
}

impl Connections {
    pub fn load(path: &Path) -> Connections {
        fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// Writes the file; a token goes to disk only for a service marked `remember`.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(d) = path.parent() {
            fs::create_dir_all(d)?;
        }
        let mut copy = self.clone();
        for sv in &mut copy.services {
            if !sv.remember {
                sv.token.clear();
            }
        }
        let tmp = path.with_extension("tmp");
        let mut f = fs::File::create(&tmp)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            f.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        f.write_all(&serde_json::to_vec_pretty(&copy).unwrap_or_default())?;
        drop(f);
        fs::rename(tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_are_checked() {
        assert!(check_url("https://github.com/x").is_ok());
        assert!(check_url("http://192.168.1.5:3000/api").is_ok());
        assert!(check_url("http://localhost:8000").is_ok());
        assert!(check_url("http://example.com").is_err());
        assert!(check_url("ftp://example.com").is_err());
        assert!(check_url("https://user:pass@example.com").is_err());
        assert!(check_url("https://").is_err());
        assert!(check_url("https://exa mple.com").is_err());
        assert_eq!(host_of("https://api.github.com/user?x=1"), "api.github.com");
    }

    #[test]
    fn github_and_gitlab_requests_are_shaped_right() {
        let g = build(
            Kind::Github,
            "https://api.github.com/",
            "tok",
            &Op::File {
                repo: "a/b",
                path: "src/my file.rs",
            },
        );
        assert_eq!(
            g.url,
            "https://api.github.com/repos/a/b/contents/src/my%20file.rs"
        );
        assert!(
            g.headers
                .iter()
                .any(|(k, v)| k == "Authorization" && v == "Bearer tok")
        );
        assert!(
            g.headers
                .iter()
                .any(|(k, v)| k == "Accept" && v.contains("raw"))
        );
        let l = build(
            Kind::Gitlab,
            "https://gitlab.com/api/v4",
            "tok",
            &Op::File {
                repo: "grp/proj",
                path: "dir/a.md",
            },
        );
        assert_eq!(
            l.url,
            "https://gitlab.com/api/v4/projects/grp%2Fproj/repository/files/dir%2Fa.md/raw?ref=HEAD"
        );
        assert!(l.headers.iter().any(|(k, _)| k == "PRIVATE-TOKEN"));
        let t = build(
            Kind::Gitea,
            "https://codeberg.org/api/v1",
            "",
            &Op::Tree {
                repo: "o/r",
                path: "",
            },
        );
        assert!(t.url.ends_with("/repos/o/r/contents"));
        assert!(
            !t.headers.iter().any(|(k, _)| k == "Authorization"),
            "no token, no header"
        );
        assert!(
            build(Kind::Github, "https://api.github.com", "t", &Op::Repos)
                .url
                .contains("per_page=100")
        );
    }

    #[test]
    fn answers_are_parsed() {
        let repos =
            br#"[{"full_name":"a/b","description":"d","private":true},{"description":"no name"}]"#;
        let r = parse_repos(Kind::Github, repos).unwrap();
        assert_eq!(r.len(), 1);
        assert!(r[0].private && r[0].id == "a/b");
        let gl = br#"[{"path_with_namespace":"g/p","description":null,"visibility":"private"}]"#;
        assert!(parse_repos(Kind::Gitlab, gl).unwrap()[0].private);
        let tree = br#"[{"name":"z.rs","path":"z.rs","type":"file","size":10},{"name":"src","path":"src","type":"dir"},{"name":"a","path":"a","type":"tree"}]"#;
        let t = parse_tree(Kind::Github, tree).unwrap();
        assert_eq!(
            t.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
            ["a", "src", "z.rs"]
        );
        assert!(parse_tree(Kind::Github, br#"{"type":"file"}"#).is_err());
        let issues = br#"[{"number":1,"title":"Bug","user":{"login":"u"},"body":"x"},{"number":2,"title":"PR","pull_request":{},"user":{"login":"u"}}]"#;
        let i = parse_issues(Kind::Github, issues).unwrap();
        assert_eq!(i.len(), 1);
        assert!(issues_text("a/b", &i).contains("#1 Bug (by u)"));
        assert!(parse_repos(Kind::Github, b"nope").is_err());
    }

    #[test]
    fn pages_are_turned_into_text() {
        let html = b"<html><body><h1>Title</h1><p>Hello</p><script>x()</script></body></html>";
        let t = page_text(html).unwrap();
        assert!(t.contains("Title") && t.contains("Hello") && !t.contains("x()"));
        assert_eq!(page_text(br#"{"a":1}"#).unwrap(), r#"{"a":1}"#);
        assert!(page_text(&[0, 1, 2]).is_err());
    }

    #[test]
    fn tokens_are_written_only_when_the_person_chose_to() {
        let dir = std::env::temp_dir().join(format!("cc-conn-{}", std::process::id()));
        let path = dir.join("connections.json");
        let c = Connections {
            internet: true,
            services: vec![
                Service {
                    name: "a".into(),
                    kind: Kind::Github,
                    base: "b".into(),
                    token: "SECRET1".into(),
                    remember: false,
                },
                Service {
                    name: "c".into(),
                    kind: Kind::Gitlab,
                    base: "d".into(),
                    token: "SECRET2".into(),
                    remember: true,
                },
            ],
        };
        c.save(&path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains("SECRET1") && text.contains("SECRET2"));
        let back = Connections::load(&path);
        assert!(back.internet && back.services.len() == 2 && back.services[0].token.is_empty());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_to_send_a_bad_address_before_touching_the_network() {
        let r = page("http://example.com", "");
        assert!(matches!(send(None, &r), Err(ConnectError::BadUrl(_))));
        let ok = page("https://example.com", "");
        assert_eq!(send(None, &ok), Err(ConnectError::NeedsPython));
        assert!(explain(&ConnectError::Http(401, "x".into()), true).contains("token"));
    }

    #[test]
    fn typed_names_cannot_climb_or_add_a_query() {
        let r = build(
            Kind::Github,
            "https://api.github.com",
            "",
            &Op::File {
                repo: "a/b?x=1#y",
                path: "../../etc/passwd",
            },
        );
        assert!(!r.url.contains('?') && !r.url.contains('#'), "{}", r.url);
        assert!(!r.url.contains("/../"), "{}", r.url);
        assert!(r.url.contains("%2E%2E"));
    }

    #[test]
    fn a_header_with_a_line_break_is_refused_and_secrets_are_hidden() {
        let r = build(
            Kind::Github,
            "https://api.github.com",
            "tok\nX: 1",
            &Op::Whoami,
        );
        assert!(clean_headers(&r).is_err());
        let ok = build(
            Kind::Github,
            "https://api.github.com",
            "ghp_secret123",
            &Op::Whoami,
        );
        assert!(clean_headers(&ok).is_ok());
        let shown = redact("Invalid header value 'Bearer ghp_secret123'".into(), &ok);
        assert!(!shown.contains("ghp_secret123"), "{shown}");
        let gl = build(
            Kind::Gitlab,
            "https://gitlab.com/api/v4",
            "glpat-abcdef",
            &Op::Whoami,
        );
        assert!(!redact("x glpat-abcdef y".into(), &gl).contains("glpat"));
    }
}
