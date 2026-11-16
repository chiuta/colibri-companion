//! Settings and saved conversations, as plain JSON files the user can read,
//! back up or delete by hand. Nothing leaves the computer.

use crate::chat::Thinking;
use crate::paths;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Write through a temporary file so a crash never leaves half a file behind. The temporary
/// name is unique (a neighbouring `x.tmp` is never touched, two writers never share one), the
/// data is flushed to disk before the rename, and on Unix the file is private to its owner:
/// settings hold the API key and conversations hold what the person wrote.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let stem = path.file_name().unwrap_or_default().to_os_string();
    // A temporary file left by a crashed run (the same process id can come back) must never block
    // saving for good: take the next free name.
    let mut tries = 0;
    let (tmp, mut f) = loop {
        let mut name = stem.clone();
        name.push(format!(
            ".{}-{}.tmp",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let tmp = path.with_file_name(name);
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        match opts.open(&tmp) {
            Ok(f) => break (tmp, f),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists && tries < 1000 => tries += 1,
            Err(e) => return Err(e),
        }
    };
    let result = (|| {
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

// ------------------------------------------------------------------ settings

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lang {
    #[default]
    Ro,
    En,
    Fr,
    It,
    Es,
    Pt,
    De,
    Hu,
}

impl Lang {
    pub const ALL: [Lang; 8] = [
        Lang::En,
        Lang::Ro,
        Lang::Fr,
        Lang::It,
        Lang::Es,
        Lang::Pt,
        Lang::De,
        Lang::Hu,
    ];

    /// Each language in its own name, as people look for it.
    pub fn native_name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Ro => "Română",
            Lang::Fr => "Français",
            Lang::It => "Italiano",
            Lang::Es => "Español",
            Lang::Pt => "Português",
            Lang::De => "Deutsch",
            Lang::Hu => "Magyar",
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ro => "ro",
            Lang::Fr => "fr",
            Lang::It => "it",
            Lang::Es => "es",
            Lang::Pt => "pt",
            Lang::De => "de",
            Lang::Hu => "hu",
        }
    }

    /// `ro`, `fr_FR.UTF-8`, `pt-BR`… → the language, when it is one of the eight.
    pub fn from_code(code: &str) -> Option<Lang> {
        let c = code.trim().to_lowercase();
        Lang::ALL.into_iter().find(|l| {
            c == l.code()
                || c.starts_with(&format!("{}_", l.code()))
                || c.starts_with(&format!("{}-", l.code()))
                || c.starts_with(&format!("{}.", l.code()))
        })
    }

    /// First launch: follow the system language when it is one of the eight, else English.
    pub fn from_env() -> Self {
        ["LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"]
            .iter()
            .filter_map(|k| std::env::var(k).ok())
            .flat_map(|v| v.split(':').map(str::to_owned).collect::<Vec<_>>())
            .find_map(|v| Lang::from_code(&v))
            .unwrap_or(Lang::En)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Theme {
    #[default]
    Light,
    Dark,
    /// Maximum contrast, for low vision.
    HighContrast,
}

/// How the model should answer, in words a person can choose between.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Style {
    Precise,
    #[default]
    Balanced,
    Creative,
}

impl Style {
    pub fn temperature(self) -> f32 {
        match self {
            Style::Precise => 0.2,
            Style::Balanced => 0.7,
            Style::Creative => 1.0,
        }
    }
    pub fn top_p(self) -> f32 {
        match self {
            Style::Precise => 0.8,
            Style::Balanced => 0.95,
            Style::Creative => 0.98,
        }
    }
}

/// Which engine build runs the model. `Auto` lets colibri use the best one this computer
/// has: CUDA for an NVIDIA card, else Vulkan for another graphics chip, else the processor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Vulkan,
    Cuda,
}

impl Backend {
    pub const ALL: [Backend; 4] = [Backend::Auto, Backend::Cuda, Backend::Vulkan, Backend::Cpu];
    /// The value for `coli setup --backend`; nothing for `Auto`, which is colibri's own choice.
    pub fn as_arg(self) -> Option<&'static str> {
        match self {
            Backend::Auto => None,
            Backend::Cpu => Some("cpu"),
            Backend::Vulkan => Some("vulkan"),
            Backend::Cuda => Some("cuda"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub lang: Lang,
    pub theme: Theme,
    /// 1.0 = normal. The whole interface scales, for comfortable reading.
    pub font_scale: f32,
    /// Options for people with disabilities.
    pub access: crate::access::Access,
    /// Give the model pinned notes and fragments of earlier conversations.
    pub memory_on: bool,
    pub colibri_root: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
    /// Where the OpenAI-compatible server is. Default: colibri's own.
    pub server_url: String,
    pub api_key: String,
    pub style: Style,
    pub thinking: Thinking,
    pub show_thinking: bool,
    pub system_prompt: String,
    pub max_tokens: Option<u32>,
    pub start_server_automatically: bool,
    pub accepted_licenses: Vec<String>,
    pub backend: Backend,
    /// Ask GitHub (once a day) whether colibri or this app has a newer version.
    pub check_updates: bool,
    /// Apple Silicon: run on the Mac's GPU through colibri's Metal backend.
    pub apple_gpu: bool,
    /// AMD Ryzen AI NPU through colibri's experimental XDNA path (Windows). Off unless asked for.
    pub use_npu: bool,
    /// colibri's `balanced` policy: it moves the experts it keeps in memory toward the ones the
    /// conversation actually uses, without changing any answer. Off by default (colibri's default).
    pub adaptive_cache: bool,
    /// Unix seconds of the last check.
    pub last_update_check: u64,
    /// The document collection the chat answers from, when the person turned it on.
    pub docs_active: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            lang: Lang::from_env(),
            theme: Theme::Light,
            font_scale: 1.1,
            access: Default::default(),
            memory_on: true,
            docs_active: None,
            colibri_root: None,
            models_dir: None,
            server_url: "http://127.0.0.1:8000".into(),
            api_key: String::new(),
            style: Style::Balanced,
            thinking: Thinking::Auto,
            show_thinking: true,
            system_prompt: String::new(),
            max_tokens: None,
            start_server_automatically: true,
            accepted_licenses: Vec::new(),
            backend: Backend::Auto,
            check_updates: true,
            apple_gpu: true,
            use_npu: false,
            adaptive_cache: false,
            last_update_check: 0,
        }
    }
}

/// Copies a settings file that could not be read whole to `settings.json.bad` (then `.bad1`, `.bad2`…
/// up to `.bad9`, then the last is replaced), so the next save never destroys the only copy.
fn keep_aside(path: &Path) {
    let base = path.as_os_str().to_os_string();
    for n in 0..10 {
        let mut name = base.clone();
        name.push(if n == 0 {
            ".bad".to_string()
        } else {
            format!(".bad{n}")
        });
        let target = PathBuf::from(name);
        if !target.exists() || n == 9 {
            let _ = fs::copy(path, target);
            return;
        }
    }
}

impl Settings {
    /// Reads the settings. A file that cannot be read at all is kept beside the original as
    /// `settings.json.bad` (the next save would otherwise overwrite it) and the defaults are used.
    /// A file with one wrong value keeps every other value: only the unreadable setting resets.
    pub fn load_from(path: &Path) -> Self {
        let Ok(text) = fs::read_to_string(path) else {
            return Self::default();
        };
        let aside = || keep_aside(path);
        let value: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                aside();
                return Self::default();
            }
        };
        let mut s = match serde_json::from_value::<Settings>(value.clone()) {
            Ok(s) => s,
            Err(_) => {
                // something in the file is wrong: keep the original, and every value that is fine
                aside();
                let mut good = serde_json::Map::new();
                if let Some(obj) = value.as_object() {
                    for (k, v) in obj {
                        let mut one = serde_json::Map::new();
                        one.insert(k.clone(), v.clone());
                        if serde_json::from_value::<Settings>(one.into()).is_ok() {
                            good.insert(k.clone(), v.clone());
                        }
                    }
                }
                serde_json::from_value(good.into()).unwrap_or_default()
            }
        };
        s.sanitize();
        s
    }

    /// Brings values a hand-edited file could make harmful back into a usable range.
    pub fn sanitize(&mut self) {
        if !self.font_scale.is_finite() {
            self.font_scale = 1.1;
        }
        self.font_scale = self.font_scale.clamp(0.6, 3.0);
        if self.server_url.trim().is_empty() {
            self.server_url = "http://127.0.0.1:8000".into();
        }
    }
    pub fn load() -> Self {
        Self::load_from(&paths::settings_file())
    }
    pub fn save_to(&self, path: &Path) -> io::Result<()> {
        write_atomic(path, serde_json::to_string_pretty(self)?.as_bytes())
    }
    pub fn save(&self) -> io::Result<()> {
        self.save_to(&paths::settings_file())
    }
}

// ------------------------------------------------------------------ conversations

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
    pub reasoning: String,
    pub tokens_per_second: Option<f64>,
    pub completion_tokens: Option<u64>,
    /// Set when the user pressed Stop before the answer was finished.
    pub stopped: bool,
    /// Pictures sent with this message, as `data:` URLs.
    pub images: Vec<String>,
    /// What the person sees in their own bubble when the sent text includes attached files.
    pub display: String,
    /// Document passages this answer was written from, numbered like [1], [2] in the text.
    pub sources: Vec<String>,
    /// What this answer cost (tokens, speed, machine load); absent on older conversations.
    pub stats: Option<crate::meter::AnswerStats>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created: u64,
    pub updated: u64,
    pub messages: Vec<StoredMessage>,
    /// Set when the person renamed it: the title is then never rewritten from the first message.
    pub title_custom: bool,
    /// Running summary of the oldest messages, for conversations too long for the model's window.
    #[serde(default)]
    pub summary: String,
    /// How many messages the summary covers.
    #[serde(default)]
    pub summary_upto: usize,
}

impl Conversation {
    pub fn new() -> Self {
        let now = now_secs();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        Self {
            id: format!("c{now}-{nanos:09}"),
            title: String::new(),
            created: now,
            updated: now,
            messages: Vec::new(),
            title_custom: false,
            summary: String::new(),
            summary_upto: 0,
        }
    }

    /// The title is the opening of the first question.
    pub fn retitle(&mut self) {
        if self.title_custom {
            return;
        }
        if let Some(first) = self.messages.iter().find(|m| m.role == "user") {
            let one_line = first
                .content
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let mut t: String = one_line.chars().take(48).collect();
            if one_line.chars().count() > 48 {
                t.push('…');
            }
            self.title = t;
        }
    }

    /// A new conversation that keeps the messages up to and including `upto`, so the person can
    /// try another direction from that point. The original is left untouched.
    pub fn branch_at(&self, upto: usize, suffix: &str) -> Conversation {
        let mut c = Conversation::new();
        let end = (upto + 1).min(self.messages.len());
        c.messages = self.messages[..end].to_vec();
        // A summary only stays valid while it covers messages that are kept.
        if self.summary_upto <= end {
            c.summary = self.summary.clone();
            c.summary_upto = self.summary_upto;
        }
        c.retitle();
        if !self.title.is_empty() {
            c.title = format!(
                "{} {suffix}",
                self.title.chars().take(70).collect::<String>()
            );
            c.title_custom = true;
        }
        c
    }

    /// A name the person chose. An empty name goes back to the automatic title.
    pub fn rename(&mut self, title: &str) {
        let t = title.split_whitespace().collect::<Vec<_>>().join(" ");
        self.title_custom = !t.is_empty();
        if t.is_empty() {
            self.retitle();
        } else {
            self.title = t.chars().take(80).collect();
        }
    }

    /// `title-2026-11-14.md`: safe on every file system, and readable.
    pub fn export_file_name(&self) -> String {
        let slug: String = self
            .title
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .split('-')
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join("-")
            .chars()
            .take(40)
            .collect();
        let slug = slug.trim_end_matches('-').to_string();
        let slug = if slug.is_empty() {
            "conversation".to_string()
        } else {
            slug
        };
        format!("{slug}-{}.md", date_label(self.updated.max(self.created)))
    }

    /// Writes the Markdown into `dir` under a name that does not exist yet; returns the path.
    pub fn export_to(&self, dir: &Path, markdown: &str) -> io::Result<PathBuf> {
        fs::create_dir_all(dir)?;
        let name = self.export_file_name();
        let (stem, ext) = name.rsplit_once('.').unwrap_or((name.as_str(), "md"));
        for n in 1..1000 {
            let file = if n == 1 {
                format!("{stem}.{ext}")
            } else {
                format!("{stem}-{n}.{ext}")
            };
            let path = dir.join(file);
            if let Ok(mut f) = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                io::Write::write_all(&mut f, markdown.as_bytes())?;
                return Ok(path);
            }
        }
        Err(io::Error::other("too many files with this name"))
    }

    /// Whether the title or any message contains `needle` (case-insensitive).
    pub fn matches(&self, needle: &str) -> bool {
        let n = needle.to_lowercase();
        contains_lowercase(&self.title, &n)
            || self
                .messages
                .iter()
                .any(|m| contains_lowercase(&m.content, &n))
    }

    pub fn to_markdown(&self, user: &str, assistant: &str) -> String {
        let mut out = format!(
            "# {}\n\n",
            if self.title.is_empty() {
                "Conversation"
            } else {
                &self.title
            }
        );
        for m in &self.messages {
            let who = if m.role == "user" { user } else { assistant };
            out.push_str(&format!("**{who}**\n\n"));
            if !m.reasoning.is_empty() {
                out.push_str("<details><summary>reasoning</summary>\n\n");
                out.push_str(m.reasoning.trim());
                out.push_str("\n\n</details>\n\n");
            }
            out.push_str(m.content.trim());
            out.push_str("\n\n");
        }
        out
    }
}

/// Letters, digits, `-` and `_`, not too long: an id that is its own file name everywhere.
fn is_plain_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 120
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// A name that is one ordinary file name on every system.
fn is_plain_file_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 120
        && !n.starts_with('.')
        && !n.ends_with(['.', ' '])
        && !n
            .chars()
            .any(|c| c.is_control() || "/\\:*?\"<>|".contains(c))
}

/// Windows refuses file names such as `CON.json` or `nul.txt` (a device name with any
/// extension, in any case). Such a stem gets a leading underscore, on every system, so the
/// same conversation folder works when it is copied to a Windows machine.
pub fn windows_safe_stem(stem: &str) -> String {
    let up = stem.to_ascii_uppercase();
    let base = up.split('.').next().unwrap_or("");
    let reserved = matches!(base, "CON" | "PRN" | "AUX" | "NUL")
        || (base.len() == 4
            && (base.starts_with("COM") || base.starts_with("LPT"))
            && matches!(base.as_bytes()[3], b'1'..=b'9'));
    if reserved {
        format!("_{stem}")
    } else {
        stem.to_string()
    }
}

/// The id of the conversation stored in this file: its name without `.json`.
/// `hay.to_lowercase().contains(needle_lower)` without building the lowercase copy when both are
/// plain ASCII (the common case): for ASCII the two are the same thing. Other text takes the
/// general path, so a letter that lowercases to several characters still matches as before.
fn contains_lowercase(hay: &str, needle_lower: &str) -> bool {
    if needle_lower.is_empty() {
        return true;
    }
    if needle_lower.is_ascii() && hay.is_ascii() {
        return find_ascii_ci(hay.as_bytes(), needle_lower.as_bytes());
    }
    hay.to_lowercase().contains(needle_lower)
}

/// Whether `hay` holds `needle` ignoring ASCII case. Looks for the first letter in both cases and
/// only then compares the rest, which is several times faster than comparing at every position.
fn find_ascii_ci(hay: &[u8], needle: &[u8]) -> bool {
    let Some((&first, rest)) = needle.split_first() else {
        return true;
    };
    let (lo, up) = (first.to_ascii_lowercase(), first.to_ascii_uppercase());
    let Some(last_start) = hay.len().checked_sub(needle.len()) else {
        return false;
    };
    (0..=last_start).any(|i| {
        (hay[i] == lo || hay[i] == up) && hay[i + 1..i + needle.len()].eq_ignore_ascii_case(rest)
    })
}

fn id_of_file(path: &Path) -> Option<String> {
    path.file_stem()?.to_str().map(str::to_string)
}

pub struct ConversationStore {
    dir: PathBuf,
}

impl ConversationStore {
    pub fn new() -> Self {
        Self {
            dir: paths::conversations_dir(),
        }
    }
    pub fn at(dir: PathBuf) -> Self {
        Self { dir }
    }
    /// The file of a conversation. The file name is the one source of truth for the id: a plain id
    /// (letters, digits, `-`, `_`) is the file's stem as it is; the listings take the id from the
    /// stem, never from the content, so what is shown can always be loaded and deleted. An id
    /// with other characters (it cannot come from this app) is mapped to a plain name that still
    /// differs for different ids, and a file already there under that exact name is used as it is.
    fn file(&self, id: &str) -> PathBuf {
        if is_plain_id(id) {
            return self.dir.join(format!("{}.json", windows_safe_stem(id)));
        }
        let literal = self.dir.join(format!("{id}.json"));
        if is_plain_file_name(id) && literal.is_file() {
            return literal;
        }
        let safe: String = id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .take(80)
            .collect();
        // a short hash of the whole id: "a/b" and "ab" must not share a file
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in id.bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        let stem = if safe.is_empty() {
            format!("_{h:016x}")
        } else {
            format!("{safe}-{h:016x}")
        };
        self.dir.join(format!("{}.json", windows_safe_stem(&stem)))
    }
    pub fn save(&self, c: &Conversation) -> io::Result<()> {
        write_atomic(
            &self.file(&c.id),
            serde_json::to_string_pretty(c)?.as_bytes(),
        )
    }
    pub fn delete(&self, id: &str) -> io::Result<()> {
        match fs::remove_file(self.file(id)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }
    /// Ids of the conversations whose title or text contains `needle`, newest first.
    pub fn search(&self, needle: &str) -> Vec<String> {
        let needle = needle.trim();
        if needle.is_empty() {
            return Vec::new();
        }
        let lower = needle.to_lowercase();
        // A file that is plain ASCII and does not hold the needle anywhere in its raw text cannot
        // hold it in a title or message either, as long as the needle has no character JSON writes
        // as an escape: the text is stored as it is otherwise. Most files are skipped unparsed.
        let skippable = lower.is_ascii()
            && !lower
                .bytes()
                .any(|b| b == b'"' || b == b'\\' || b < 0x20 || b == 0x7f);
        let mut hits: Vec<(u64, String)> = fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| Some((id_of_file(&e.path())?, fs::read(e.path()).ok()?)))
            .filter(|(_, bytes)| {
                !(skippable && bytes.is_ascii() && !find_ascii_ci(bytes, lower.as_bytes()))
            })
            .filter_map(|(id, bytes)| {
                let c: Conversation = serde_json::from_slice(&bytes).ok()?;
                c.matches(needle).then_some((c.updated, id))
            })
            .collect();
        hits.sort_by_key(|(updated, _)| std::cmp::Reverse(*updated));
        hits.into_iter().map(|(_, id)| id).collect()
    }

    /// One conversation, in full.
    pub fn load(&self, id: &str) -> Option<Conversation> {
        let mut c: Conversation = fs::read_to_string(self.file(id))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())?;
        c.id = id.to_string(); // the name decides, not what a hand-edited file claims
        Some(c)
    }

    /// The sidebar list, newest first: ids, titles and dates only. The messages are skipped
    /// while reading (they are not even allocated), so a long history opens fast and uses little memory.
    pub fn summaries(&self) -> Vec<Conversation> {
        #[derive(Deserialize)]
        #[serde(default)]
        struct Meta {
            id: String,
            title: String,
            created: u64,
            updated: u64,
            title_custom: bool,
            messages: serde::de::IgnoredAny,
        }
        impl Default for Meta {
            fn default() -> Self {
                Meta {
                    id: String::new(),
                    title: String::new(),
                    created: 0,
                    updated: 0,
                    title_custom: false,
                    messages: serde::de::IgnoredAny,
                }
            }
        }
        let mut all: Vec<Conversation> = fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| Some((id_of_file(&e.path())?, fs::read(e.path()).ok()?)))
            .filter_map(|(id, bytes)| {
                // from a slice, not from a reader: a reader is parsed byte by byte, about three
                // times slower, and the skipped messages are most of the file
                let m = serde_json::from_slice::<Meta>(&bytes).ok()?;
                Some((id, m))
            })
            .map(|(id, m)| Conversation {
                id,
                title: m.title,
                created: m.created,
                updated: m.updated,
                messages: Vec::new(),
                title_custom: m.title_custom,
                summary: String::new(),
                summary_upto: 0,
            })
            .collect();
        all.sort_by_key(|c| std::cmp::Reverse(c.updated));
        all
    }

    /// Newest first, with all messages. Unreadable files are skipped, never deleted.
    pub fn list(&self) -> Vec<Conversation> {
        let mut all: Vec<Conversation> = fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| Some((id_of_file(&e.path())?, fs::read_to_string(e.path()).ok()?)))
            .filter_map(|(id, t)| {
                let mut c: Conversation = serde_json::from_str(&t).ok()?;
                c.id = id;
                Some(c)
            })
            .collect();
        all.sort_by_key(|c| std::cmp::Reverse(c.updated));
        all
    }
}

impl Default for ConversationStore {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------------------------------------------ dates

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `2026-11-14` (UTC; good enough to tell conversations apart).
pub fn date_label(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("cc-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn dates() {
        assert_eq!(date_label(0), "1970-01-01");
        assert_eq!(date_label(1_794_000_000), "2026-11-06");
        assert_eq!(date_label(951_782_400), "2000-02-29");
    }

    #[test]
    fn conversation_round_trip_and_order() {
        let dir = temp("conv");
        let store = ConversationStore::at(dir.clone());
        let mut a = Conversation::new();
        a.messages.push(StoredMessage {
            role: "user".into(),
            content: "Bună!   Ce faci astăzi, într-o zi atât de frumoasă și lungă pentru titlu?"
                .into(),
            ..Default::default()
        });
        a.retitle();
        a.updated = 100;
        assert!(a.title.ends_with('…') && a.title.chars().count() <= 49);
        let mut b = Conversation::new();
        b.id = "other".into();
        b.updated = 200;
        store.save(&a).unwrap();
        store.save(&b).unwrap();
        let list = store.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "other");
        assert_eq!(list[1], a);
        store.delete("other").unwrap();
        store.delete("other").unwrap(); // deleting twice is fine
        assert_eq!(store.list().len(), 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn ids_cannot_escape_the_folder() {
        let store = ConversationStore::at(PathBuf::from("/tmp/x"));
        for id in ["../../etc/passwd", "..", "a/../b", "/abs/olute", "C:\\x\\y"] {
            let f = store.file(id);
            assert_eq!(f.parent(), Some(Path::new("/tmp/x")), "{id}");
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            assert!(
                !name.starts_with('.') && !name.contains(['/', '\\', ':']),
                "{name}"
            );
        }
    }

    #[test]
    fn settings_survive_missing_and_unknown_fields() {
        let dir = temp("settings");
        let path = dir.join("settings.json");
        assert_eq!(
            Settings::load_from(&path).server_url,
            "http://127.0.0.1:8000"
        );
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, r#"{"font_scale":1.5,"from_the_future":true}"#).unwrap();
        let s = Settings::load_from(&path);
        assert_eq!(s.font_scale, 1.5);
        let mut s2 = s.clone();
        s2.style = Style::Creative;
        s2.save_to(&path).unwrap();
        assert_eq!(Settings::load_from(&path).style, Style::Creative);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn markdown_export() {
        let mut c = Conversation::new();
        c.title = "T".into();
        c.messages.push(StoredMessage {
            role: "user".into(),
            content: "q".into(),
            ..Default::default()
        });
        c.messages.push(StoredMessage {
            role: "assistant".into(),
            content: "a".into(),
            reasoning: "r".into(),
            ..Default::default()
        });
        let md = c.to_markdown("Eu", "Colibri");
        assert!(md.contains("**Eu**") && md.contains("**Colibri**") && md.contains("<details>"));
    }
}

#[cfg(test)]
mod lang_tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("cc-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn codes_round_trip() {
        for l in Lang::ALL {
            assert_eq!(Lang::from_code(l.code()), Some(l));
        }
        assert_eq!(Lang::from_code("pt_BR.UTF-8"), Some(Lang::Pt));
        assert_eq!(Lang::from_code("hu-HU"), Some(Lang::Hu));
        assert_eq!(Lang::from_code("ja_JP"), None);
        assert_eq!(Lang::from_code("C"), None);
    }

    #[test]
    fn summaries_skip_the_messages_and_load_returns_them() {
        let dir = temp("summaries");
        let store = ConversationStore::at(dir.clone());
        let mut c = Conversation::new();
        c.title = "Salut".into();
        c.messages.push(StoredMessage {
            role: "user".into(),
            content: "x".repeat(10_000),
            ..Default::default()
        });
        store.save(&c).unwrap();
        let list = store.summaries();
        assert_eq!(list.len(), 1);
        assert_eq!(
            (list[0].id.as_str(), list[0].title.as_str()),
            (c.id.as_str(), "Salut")
        );
        assert!(list[0].messages.is_empty());
        assert_eq!(store.load(&c.id).unwrap().messages.len(), 1);
        assert!(store.load("missing").is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_search_and_export_name() {
        let dir = temp("search");
        let store = ConversationStore::at(dir.clone());
        let mut a = Conversation::new();
        a.messages.push(StoredMessage {
            role: "user".into(),
            content: "Cum fac o supă de legume?".into(),
            ..Default::default()
        });
        a.retitle();
        a.updated = 1_000;
        let mut b = Conversation::new();
        b.id = format!("{}x", b.id);
        b.messages.push(StoredMessage {
            role: "user".into(),
            content: "Explică-mi Rust".into(),
            ..Default::default()
        });
        b.rename("  Învăț   Rust  ");
        b.retitle(); // must not undo the name
        assert_eq!(b.title, "Învăț Rust");
        store.save(&a).unwrap();
        store.save(&b).unwrap();
        assert_eq!(store.search("SUPĂ"), vec![a.id.clone()]);
        assert_eq!(store.search("rust"), vec![b.id.clone()]);
        assert!(store.search("  ").is_empty());
        b.rename("");
        assert_eq!(b.title, "Explică-mi Rust");
        assert!(
            b.export_file_name().starts_with("Explic-mi-Rust-")
                || b.export_file_name().starts_with("Explică-mi-Rust-")
        );
        assert!(b.export_file_name().ends_with(".md"));
        let mut long = Conversation::new();
        long.title = "Help me write a polite email asking for a deadline extension".into();
        assert!(!long.export_file_name().contains("--"));
        assert_eq!(
            Conversation::new().export_file_name().split('-').next(),
            Some("conversation")
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn exports_never_overwrite() {
        let dir = temp("export");
        let mut c = Conversation::new();
        c.title = "Salut lume".into();
        let first = c.export_to(&dir, "one").unwrap();
        let second = c.export_to(&dir, "two").unwrap();
        assert_ne!(first, second);
        assert_eq!(fs::read_to_string(&first).unwrap(), "one");
        assert_eq!(fs::read_to_string(&second).unwrap(), "two");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_branch_keeps_the_messages_up_to_the_chosen_one() {
        let mut c = Conversation::new();
        for (i, role) in ["user", "assistant", "user", "assistant"]
            .iter()
            .enumerate()
        {
            c.messages.push(StoredMessage {
                role: role.to_string(),
                content: format!("m{i}"),
                ..Default::default()
            });
        }
        c.retitle();
        let b = c.branch_at(1, "(2)");
        assert_eq!(b.messages.len(), 2);
        assert_eq!(b.messages[1].content, "m1");
        assert_ne!(b.id, c.id);
        assert!(b.title.ends_with("(2)"));
        assert_eq!(c.messages.len(), 4);
        assert_eq!(c.branch_at(99, "x").messages.len(), 4);
    }

    #[test]
    fn a_neighbouring_tmp_file_survives_and_nothing_is_left_behind() {
        let dir = temp("atomic");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("settings.tmp"), b"mine").unwrap();
        write_atomic(&dir.join("settings.json"), b"{}").unwrap();
        assert_eq!(fs::read(dir.join("settings.tmp")).unwrap(), b"mine");
        assert_eq!(fs::read(dir.join("settings.json")).unwrap(), b"{}");
        let names: Vec<_> = fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(names.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn written_files_are_private_to_their_owner() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp("mode");
        write_atomic(&dir.join("s.json"), b"{}").unwrap();
        let mode = fs::metadata(dir.join("s.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o077, 0, "{mode:o}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_wrong_setting_does_not_reset_the_others() {
        let dir = temp("lenient");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        fs::write(
            &path,
            r#"{"font_scale":"big","theme":"Dark","api_key":"k","unknown":1}"#,
        )
        .unwrap();
        let s = Settings::load_from(&path);
        assert_eq!(s.theme, Theme::Dark);
        assert_eq!(s.api_key, "k");
        assert_eq!(s.font_scale, Settings::default().font_scale);
        // absurd values from a hand edit are brought back into range
        fs::write(&path, r#"{"font_scale":0.0,"server_url":" "}"#).unwrap();
        let s = Settings::load_from(&path);
        assert!(s.font_scale >= 0.6);
        assert_eq!(s.server_url, "http://127.0.0.1:8000");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_settings_file_is_kept_aside_not_lost() {
        let dir = temp("corrupt");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        fs::write(&path, b"{ \"api_key\": \"secret\" ").unwrap();
        let s = Settings::load_from(&path);
        assert_eq!(s, Settings::load_from(&dir.join("missing.json")));
        assert!(
            fs::read_to_string(dir.join("settings.json.bad"))
                .unwrap()
                .contains("secret")
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_id_with_no_usable_characters_still_gets_a_name() {
        let store = ConversationStore::at(PathBuf::from("/tmp/x"));
        let f = store.file("///");
        assert_eq!(f.parent(), Some(Path::new("/tmp/x")));
        assert_ne!(f.file_name().unwrap(), ".json");
        assert_ne!(f, store.file("////"), "different ids, different files");
    }

    #[test]
    fn a_stale_temporary_file_from_a_crashed_run_never_blocks_saving() {
        use std::sync::atomic::Ordering;
        let dir = temp("stale-tmp");
        fs::create_dir_all(&dir).unwrap();
        let cur = SEQ.load(Ordering::Relaxed);
        for n in cur..cur + 60 {
            fs::write(
                dir.join(format!("s.json.{}-{n}.tmp", std::process::id())),
                b"old",
            )
            .unwrap();
        }
        write_atomic(&dir.join("s.json"), b"new").unwrap();
        assert_eq!(fs::read(dir.join("s.json")).unwrap(), b"new");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_file_name_decides_the_conversation_id() {
        let dir = temp("names");
        fs::create_dir_all(&dir).unwrap();
        let store = ConversationStore::at(dir.clone());
        // a file whose content claims another id (copied, renamed or edited by hand)
        let mut c = Conversation::new();
        c.id = "claimed-elsewhere".into();
        fs::write(dir.join("c1-1.json"), serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(store.list()[0].id, "c1-1");
        assert_eq!(store.summaries()[0].id, "c1-1");
        assert_eq!(store.load("c1-1").unwrap().id, "c1-1");
        assert!(store.load("claimed-elsewhere").is_none());
        store.delete("c1-1").unwrap();
        assert!(!dir.join("c1-1.json").exists());
        assert!(store.list().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn different_ids_never_share_a_file_and_device_names_are_avoided() {
        let dir = temp("collide");
        fs::create_dir_all(&dir).unwrap();
        let store = ConversationStore::at(dir.clone());
        let ids = [
            "ab", "a/b", "a\\b", "A B", "CON", "nul", "com1", "lpt9", "../x", "é", "",
        ];
        for id in ids {
            let mut c = Conversation::new();
            c.id = id.to_string();
            c.title = format!("t:{id}");
            store.save(&c).unwrap();
        }
        let files: Vec<_> = fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(files.len(), ids.len(), "every id has its own file");
        for id in ids {
            assert_eq!(store.load(id).unwrap().title, format!("t:{id}"), "{id:?}");
        }
        for f in files {
            let name = f.file_name().to_string_lossy().to_string();
            assert!(!name.contains(['/', '\\']) && name.len() < 140, "{name}");
            let stem = name.trim_end_matches(".json").to_ascii_uppercase();
            assert!(
                !["CON", "NUL", "COM1", "LPT9", "PRN", "AUX"].contains(&stem.as_str()),
                "{name}"
            );
        }
        for id in ids {
            store.delete(id).unwrap();
        }
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_settings_file_that_is_not_an_object_or_loses_a_value_is_kept_aside_every_time() {
        let dir = temp("aside");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        fs::write(&path, b"[1,2,3]").unwrap();
        let _ = Settings::load_from(&path);
        assert!(dir.join("settings.json.bad").exists());
        fs::write(&path, br#"{"font_scale":"big","api_key":"k2"}"#).unwrap();
        let s = Settings::load_from(&path);
        assert_eq!(s.api_key, "k2");
        // the first copy is not overwritten by the second
        assert_eq!(fs::read(dir.join("settings.json.bad")).unwrap(), b"[1,2,3]");
        assert!(dir.join("settings.json.bad1").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// Timing of the sidebar operations on a big history (`cargo test --release -p companion-core
    /// perf_ -- --ignored --nocapture`). Prints only; asserts no speed.
    #[test]
    #[ignore]
    fn perf_search_and_listing_on_a_big_history() {
        let dir = std::env::temp_dir().join(format!("cc-perf-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let store = ConversationStore::at(dir.clone());
        let sentence = "A sentence of ordinary text that wraps. ".repeat(15);
        for i in 0..500 {
            let mut c = Conversation::new();
            c.id = format!("c{i}");
            c.title = format!("Title {i}");
            c.updated = i;
            for j in 0..40 {
                c.messages.push(StoredMessage {
                    role: if j % 2 == 0 { "user" } else { "assistant" }.into(),
                    content: format!("{sentence} message {j}"),
                    ..Default::default()
                });
            }
            store.save(&c).unwrap();
        }
        let time = |label: &str, f: &dyn Fn() -> usize| {
            let _ = f();
            let t = std::time::Instant::now();
            let n = (0..5).map(|_| f()).sum::<usize>();
            println!(
                "perf_store {label:<28} {:>8.1} ms/call (result {})",
                t.elapsed().as_secs_f64() * 1000.0 / 5.0,
                n / 5
            );
        };
        let old = |needle: &str| {
            let n = needle.to_lowercase();
            store
                .list()
                .into_iter()
                .filter(|c| {
                    c.title.to_lowercase().contains(&n)
                        || c.messages
                            .iter()
                            .any(|m| m.content.to_lowercase().contains(&n))
                })
                .count()
        };
        let old_summaries = || {
            #[derive(Deserialize)]
            #[serde(default)]
            struct Meta {
                title: String,
                updated: u64,
                messages: serde::de::IgnoredAny,
            }
            impl Default for Meta {
                fn default() -> Self {
                    Meta {
                        title: String::new(),
                        updated: 0,
                        messages: serde::de::IgnoredAny,
                    }
                }
            }
            fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .filter_map(|e| fs::File::open(e.path()).ok())
                .filter_map(|f| serde_json::from_reader::<_, Meta>(io::BufReader::new(f)).ok())
                .inspect(|m| {
                    std::hint::black_box(m.title.len() + m.updated as usize);
                })
                .count()
        };
        time("summaries OLD (from_reader)", &old_summaries);
        time("search OLD (list+lowercase)", &|| old("zzzz-not-there"));
        time("search (no hit)", &|| store.search("zzzz-not-there").len());
        time("search (hit in title)", &|| store.search("title 4").len());
        time("search (non-ASCII needle)", &|| store.search("żółć").len());
        time("summaries (sidebar list)", &|| store.summaries().len());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_ascii_fast_search_agrees_with_the_general_one() {
        let cases = [
            ("Hello World", "hello"),
            ("Hello World", "o w"),
            ("Hello", "xyz"),
            ("", "a"),
            ("abc", ""),
            ("Straße", "straße"),
            ("İstanbul", "i̇stanbul"),
            ("K\u{212A}elvin", "kk"),
            ("ŻÓŁĆ text", "żółć"),
            ("ascii only", "ascii"),
        ];
        for (hay, needle) in cases {
            assert_eq!(
                contains_lowercase(hay, needle),
                hay.to_lowercase().contains(needle),
                "{hay:?} / {needle:?}"
            );
        }
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alphabet: Vec<char> = "abAB ,.éÉİK\u{212A}ß1".chars().collect();
        for _ in 0..3000 {
            let make = |len: usize, n: &mut dyn FnMut() -> u64| -> String {
                (0..len)
                    .map(|_| alphabet[(n() % alphabet.len() as u64) as usize])
                    .collect()
            };
            let hay = make((next() % 12) as usize, &mut next);
            let needle = make((next() % 3) as usize, &mut next).to_lowercase();
            assert_eq!(
                contains_lowercase(&hay, &needle),
                hay.to_lowercase().contains(&needle),
                "{hay:?} / {needle:?}"
            );
        }
    }

    #[test]
    fn summaries_read_the_same_titles_and_dates_as_before() {
        let dir = std::env::temp_dir().join(format!("cc-sum-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let store = ConversationStore::at(dir.clone());
        for (i, title) in ["b", "a"].iter().enumerate() {
            let mut c = Conversation::new();
            c.id = format!("c{i}");
            c.title = (*title).into();
            c.updated = 10 + i as u64;
            c.title_custom = i == 1;
            c.messages.push(StoredMessage {
                content: "text with \"quotes\" and ünïcode".into(),
                ..Default::default()
            });
            store.save(&c).unwrap();
        }
        fs::write(dir.join("broken.json"), b"{not json").unwrap();
        let s = store.summaries();
        assert_eq!(s.len(), 2);
        assert_eq!(
            (s[0].id.as_str(), s[0].title.as_str(), s[0].updated),
            ("c1", "a", 11)
        );
        assert!(s[0].title_custom && s[0].messages.is_empty());
        assert_eq!((s[1].id.as_str(), s[1].title.as_str()), ("c0", "b"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_finds_what_the_full_parse_finds_and_in_the_same_order() {
        let dir = std::env::temp_dir().join(format!("cc-srch-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let store = ConversationStore::at(dir.clone());
        let texts = [
            "plain words here",
            "He said \"Quoted\" loudly",
            "line one\nline two",
            "back\\slash and tab\there",
            "Żółć and ünïcode",
            "needle in the NEEDLE case",
        ];
        for (i, t) in texts.iter().enumerate() {
            let mut c = Conversation::new();
            c.id = format!("c{i}");
            c.title = format!("T{i}");
            c.updated = 100 - i as u64;
            c.messages.push(StoredMessage {
                content: (*t).into(),
                ..Default::default()
            });
            store.save(&c).unwrap();
        }
        for needle in [
            "plain",
            "QUOTED",
            "\"quoted\"",
            "line two",
            "line one\nline two",
            "back\\slash",
            "tab\there",
            "żółć",
            "ÜNÏCODE",
            "needle",
            "role",
            "t3",
            "nothing-like-this",
            "c0",
        ] {
            let slow: Vec<String> = store
                .list()
                .into_iter()
                .filter(|c| c.matches(needle))
                .map(|c| c.id)
                .collect();
            assert_eq!(store.search(needle), slow, "needle {needle:?}");
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
