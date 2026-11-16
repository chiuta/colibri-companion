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

/// Write through a temporary file so a crash never leaves half a file behind.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
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
    /// Unix seconds of the last check.
    pub last_update_check: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            lang: Lang::from_env(),
            theme: Theme::Light,
            font_scale: 1.1,
            access: Default::default(),
            memory_on: true,
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
            last_update_check: 0,
        }
    }
}

impl Settings {
    pub fn load_from(path: &Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
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
        self.title.to_lowercase().contains(&n)
            || self
                .messages
                .iter()
                .any(|m| m.content.to_lowercase().contains(&n))
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
    fn file(&self, id: &str) -> PathBuf {
        let safe: String = id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        self.dir.join(format!("{safe}.json"))
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
        self.list()
            .into_iter()
            .filter(|c| c.matches(needle))
            .map(|c| c.id)
            .collect()
    }

    /// One conversation, in full.
    pub fn load(&self, id: &str) -> Option<Conversation> {
        fs::read_to_string(self.file(id))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
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
            .filter_map(|e| fs::File::open(e.path()).ok())
            .filter_map(|f| serde_json::from_reader::<_, Meta>(io::BufReader::new(f)).ok())
            .filter(|m| !m.id.is_empty())
            .map(|m| Conversation {
                id: m.id,
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
            .filter_map(|e| fs::read_to_string(e.path()).ok())
            .filter_map(|t| serde_json::from_str(&t).ok())
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
        assert_eq!(
            store.file("../../etc/passwd"),
            PathBuf::from("/tmp/x/etcpasswd.json")
        );
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
}
