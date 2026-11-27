//! The whole chat history as one portable file, and back.
//!
//! The format is plain JSON, written down in `docs/CHAT-HISTORY.md`, so other programs (the
//! CartaColibri library app, a script, a spreadsheet import) can read and write it:
//!
//! ```json
//! { "format": "colibri-chat-history", "version": 1, "app": "colibri-companion",
//!   "exported": 1790000000,
//!   "conversations": [ { "id": "c1-0", "title": "…", "created": 1, "updated": 2,
//!       "messages": [ { "role": "user", "content": "…", "reasoning": "", "sources": [] } ] } ] }
//! ```
//!
//! Importing never trusts the file: sizes are capped, roles are checked, pictures must be
//! `data:image/` URLs, ids are cleaned, and a conversation already here is only replaced by a
//! newer copy of itself.

use crate::store::{Conversation, ConversationStore, StoredMessage, now_secs};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "colibri-chat-history";
pub const VERSION: u64 = 1;
/// A history file larger than this is refused (it is read into memory).
pub const MAX_FILE_BYTES: u64 = 128 << 20;
const MAX_CONVERSATIONS: usize = 20_000;
const MAX_MESSAGES: usize = 100_000;
const MAX_TEXT: usize = 32 << 20;
const MAX_IMAGE: usize = 16 << 20;
const MAX_SOURCES: usize = 200;

#[derive(Debug, PartialEq)]
pub enum HistoryError {
    TooBig,
    NotHistory,
    NewerVersion(u64),
    Read(String),
}

impl std::fmt::Display for HistoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooBig => write!(f, "file too big"),
            Self::NotHistory => write!(f, "not a chat history file"),
            Self::NewerVersion(v) => write!(f, "written by a newer version ({v})"),
            Self::Read(e) => write!(f, "{e}"),
        }
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct ImportReport {
    pub added: usize,
    pub updated: usize,
    /// Already here, same or newer.
    pub skipped: usize,
    /// Not a conversation (no messages, bad shape).
    pub rejected: usize,
}

/// Every saved conversation, oldest first, as the portable file. `images` keeps the pictures
/// (large); without them the file stays small and readable.
pub fn export_json(convs: &[Conversation], images: bool) -> String {
    let mut list: Vec<Value> = Vec::with_capacity(convs.len());
    for c in convs {
        let msgs: Vec<Value> = c
            .messages
            .iter()
            .map(|m| {
                let mut o = serde_json::Map::new();
                o.insert("role".into(), m.role.clone().into());
                o.insert("content".into(), m.content.clone().into());
                if !m.reasoning.is_empty() {
                    o.insert("reasoning".into(), m.reasoning.clone().into());
                }
                if !m.display.is_empty() {
                    o.insert("display".into(), m.display.clone().into());
                }
                if !m.sources.is_empty() {
                    o.insert("sources".into(), m.sources.clone().into());
                }
                if m.stopped {
                    o.insert("stopped".into(), true.into());
                }
                if images && !m.images.is_empty() {
                    o.insert("images".into(), m.images.clone().into());
                }
                Value::Object(o)
            })
            .collect();
        list.push(serde_json::json!({
            "id": c.id, "title": c.title, "created": c.created, "updated": c.updated,
            "messages": msgs,
        }));
    }
    list.sort_by_key(|v| v["created"].as_u64().unwrap_or(0));
    let doc = serde_json::json!({
        "format": FORMAT, "version": VERSION, "app": "colibri-companion",
        "exported": now_secs(), "conversations": list,
    });
    serde_json::to_string_pretty(&doc).unwrap_or_default()
}

/// Cheap check on the first bytes of a file: is it worth reading as a history?
pub fn looks_like_history(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    if f.metadata()
        .map(|m| m.len() > MAX_FILE_BYTES)
        .unwrap_or(true)
    {
        return false;
    }
    let mut head = vec![0u8; 4096];
    let n = f.read(&mut head).unwrap_or(0);
    String::from_utf8_lossy(&head[..n]).contains(FORMAT)
}

fn clean_text(v: &Value) -> String {
    let s = v.as_str().unwrap_or("");
    if s.len() > MAX_TEXT {
        let mut end = MAX_TEXT;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        return s[..end].to_string();
    }
    s.to_string()
}

fn clean_message(v: &Value) -> Option<StoredMessage> {
    let role = v.get("role")?.as_str()?;
    if !["user", "assistant", "system"].contains(&role) {
        return None;
    }
    let mut m = StoredMessage {
        role: role.to_string(),
        content: clean_text(v.get("content").unwrap_or(&Value::Null)),
        reasoning: clean_text(v.get("reasoning").unwrap_or(&Value::Null)),
        display: clean_text(v.get("display").unwrap_or(&Value::Null)),
        stopped: v.get("stopped").and_then(Value::as_bool).unwrap_or(false),
        ..Default::default()
    };
    if let Some(a) = v.get("sources").and_then(Value::as_array) {
        m.sources = a
            .iter()
            .take(MAX_SOURCES)
            .filter_map(|s| s.as_str().map(|s| s.chars().take(20_000).collect()))
            .collect();
    }
    if let Some(a) = v.get("images").and_then(Value::as_array) {
        m.images = a
            .iter()
            .filter_map(Value::as_str)
            .filter(|s| s.starts_with("data:image/") && s.len() <= MAX_IMAGE)
            .take(16)
            .map(str::to_string)
            .collect();
    }
    Some(m)
}

fn clean_id(v: Option<&Value>) -> String {
    let raw = v.and_then(Value::as_str).unwrap_or("");
    let id: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(80)
        .collect();
    if id.is_empty() {
        Conversation::new().id
    } else {
        id
    }
}

/// Reads the portable file. Conversations that cannot be read are counted, not fatal.
pub fn parse(text: &str) -> Result<(Vec<Conversation>, usize), HistoryError> {
    if text.len() as u64 > MAX_FILE_BYTES {
        return Err(HistoryError::TooBig);
    }
    let v: Value = serde_json::from_str(text).map_err(|e| HistoryError::Read(e.to_string()))?;
    if v.get("format").and_then(Value::as_str) != Some(FORMAT) {
        return Err(HistoryError::NotHistory);
    }
    let version = v.get("version").and_then(Value::as_u64).unwrap_or(0);
    if version > VERSION {
        return Err(HistoryError::NewerVersion(version));
    }
    let Some(list) = v.get("conversations").and_then(Value::as_array) else {
        return Err(HistoryError::NotHistory);
    };
    let now = now_secs();
    let (mut out, mut rejected) = (Vec::new(), 0usize);
    for (i, c) in list.iter().enumerate() {
        if i >= MAX_CONVERSATIONS {
            rejected += list.len() - i;
            break;
        }
        let Some(msgs) = c.get("messages").and_then(Value::as_array) else {
            rejected += 1;
            continue;
        };
        let messages: Vec<StoredMessage> = msgs
            .iter()
            .take(MAX_MESSAGES)
            .filter_map(clean_message)
            .collect();
        if messages.is_empty() {
            rejected += 1;
            continue;
        }
        let created = c
            .get("created")
            .and_then(Value::as_u64)
            .unwrap_or(now)
            .min(now + 86_400);
        let updated = c
            .get("updated")
            .and_then(Value::as_u64)
            .unwrap_or(created)
            .clamp(created, now + 86_400);
        let mut conv = Conversation {
            id: clean_id(c.get("id")),
            title: clean_text(c.get("title").unwrap_or(&Value::Null))
                .chars()
                .take(300)
                .collect(),
            created,
            updated,
            messages,
            title_custom: true,
            ..Default::default()
        };
        if conv.title.trim().is_empty() {
            conv.title_custom = false;
            conv.retitle();
        }
        out.push(conv);
    }
    Ok((out, rejected))
}

/// Imports into the store: new ids are added, a known id is replaced only by a newer copy.
pub fn import_into(store: &ConversationStore, text: &str) -> Result<ImportReport, HistoryError> {
    let (convs, rejected) = parse(text)?;
    let mut r = ImportReport {
        rejected,
        ..Default::default()
    };
    for c in convs {
        match store.load(&c.id) {
            None => {
                if store.save(&c).is_ok() {
                    r.added += 1;
                } else {
                    r.rejected += 1;
                }
            }
            Some(old) if old.updated < c.updated => {
                if store.save(&c).is_ok() {
                    r.updated += 1;
                } else {
                    r.rejected += 1;
                }
            }
            Some(_) => r.skipped += 1,
        }
    }
    Ok(r)
}

pub fn import_file(store: &ConversationStore, path: &Path) -> Result<ImportReport, HistoryError> {
    let meta = std::fs::metadata(path).map_err(|e| HistoryError::Read(e.to_string()))?;
    if meta.len() > MAX_FILE_BYTES {
        return Err(HistoryError::TooBig);
    }
    let text = std::fs::read_to_string(path).map_err(|e| HistoryError::Read(e.to_string()))?;
    import_into(store, &text)
}

/// Writes `colibri-chat-history-<date>.json` into `dir` under a name not used yet.
pub fn export_file(dir: &Path, convs: &[Conversation], images: bool) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let stem = format!(
        "colibri-chat-history-{}",
        crate::store::date_label(now_secs())
    );
    for n in 1..1000 {
        let name = if n == 1 {
            format!("{stem}.json")
        } else {
            format!("{stem}-{n}.json")
        };
        let path = dir.join(name);
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            std::io::Write::write_all(&mut f, export_json(convs, images).as_bytes())?;
            return Ok(path);
        }
    }
    Err(std::io::Error::other("too many files with this name"))
}

/// One Markdown file per conversation in a new folder inside `dir`; returns the folder.
pub fn export_markdown_folder(
    dir: &Path,
    convs: &[Conversation],
    user: &str,
    assistant: &str,
) -> std::io::Result<PathBuf> {
    let base = format!("colibri-chats-{}", crate::store::date_label(now_secs()));
    let mut folder = dir.join(&base);
    let mut n = 1;
    while folder.exists() {
        n += 1;
        folder = dir.join(format!("{base}-{n}"));
        if n > 1000 {
            return Err(std::io::Error::other("too many folders with this name"));
        }
    }
    std::fs::create_dir_all(&folder)?;
    for c in convs {
        c.export_to(&folder, &c.to_markdown(user, assistant))?;
    }
    Ok(folder)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(tag: &str) -> ConversationStore {
        let d = std::env::temp_dir().join(format!("hist-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        ConversationStore::at(d)
    }

    fn conv(id: &str, updated: u64, text: &str) -> Conversation {
        Conversation {
            id: id.into(),
            title: format!("T {id}"),
            created: 100,
            updated,
            messages: vec![
                StoredMessage {
                    role: "user".into(),
                    content: text.into(),
                    ..Default::default()
                },
                StoredMessage {
                    role: "assistant".into(),
                    content: "răspuns ăâîșț 🙂".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn round_trip_keeps_text_and_merges_by_age() {
        let a = store("a");
        let b = store("b");
        a.save(&conv("one", 200, "salut")).unwrap();
        a.save(&conv("two", 300, "hei")).unwrap();
        let json = export_json(&a.list(), false);
        let r = import_into(&b, &json).unwrap();
        assert_eq!((r.added, r.updated, r.skipped, r.rejected), (2, 0, 0, 0));
        assert_eq!(
            b.load("one").unwrap().messages[1].content,
            "răspuns ăâîșț 🙂"
        );
        // again: nothing changes
        let r = import_into(&b, &json).unwrap();
        assert_eq!((r.added, r.updated, r.skipped), (0, 0, 2));
        // a newer copy replaces, an older one does not
        let newer = export_json(&[conv("one", 900, "mai nou")], false);
        assert_eq!(import_into(&b, &newer).unwrap().updated, 1);
        assert_eq!(b.load("one").unwrap().messages[0].content, "mai nou");
        let older = export_json(&[conv("one", 150, "vechi")], false);
        assert_eq!(import_into(&b, &older).unwrap().skipped, 1);
        assert_eq!(b.load("one").unwrap().messages[0].content, "mai nou");
    }

    #[test]
    fn images_only_when_asked_and_only_real_pictures() {
        let mut c = conv("p", 200, "x");
        c.messages[0].images = vec!["data:image/png;base64,AAAA".into()];
        assert!(!export_json(std::slice::from_ref(&c), false).contains("data:image"));
        assert!(export_json(std::slice::from_ref(&c), true).contains("data:image"));
        let hostile = r#"{"format":"colibri-chat-history","version":1,"conversations":[
          {"id":"h","messages":[{"role":"user","content":"a","images":["http://evil/x.png","file:///etc/passwd","data:image/png;base64,AA"]}]}]}"#;
        let (v, _) = parse(hostile).unwrap();
        assert_eq!(v[0].messages[0].images, ["data:image/png;base64,AA"]);
    }

    #[test]
    fn hostile_files_are_refused_or_cleaned() {
        assert_eq!(parse("{}").unwrap_err(), HistoryError::NotHistory);
        assert_eq!(parse("[]").unwrap_err(), HistoryError::NotHistory);
        assert!(matches!(parse("not json"), Err(HistoryError::Read(_))));
        assert_eq!(
            parse(r#"{"format":"colibri-chat-history","version":99,"conversations":[]}"#)
                .unwrap_err(),
            HistoryError::NewerVersion(99)
        );
        let bad = r#"{"format":"colibri-chat-history","version":1,"conversations":[
          1, null, "x", {"messages":"no"}, {"id":"e","messages":[]},
          {"id":"../../etc/passwd","title":"t","created":"x","messages":[
             {"role":"root","content":"no"},{"role":"user","content":"ok"},{"content":"no role"}]},
          {"id":"","created":18446744073709551615,"updated":0,"messages":[{"role":"assistant","content":"z"}]}]}"#;
        let (v, rejected) = parse(bad).unwrap();
        assert_eq!(rejected, 5);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].id, "etcpasswd");
        assert_eq!(v[0].messages.len(), 1);
        assert!(v[1].created <= now_secs() + 86_400 && v[1].updated >= v[1].created);
        assert!(!v[1].id.is_empty());
        // nothing escapes the folder
        let s = store("evil");
        import_into(&s, bad).unwrap();
        assert!(s.load("etcpasswd").is_some());
    }

    #[test]
    fn too_many_conversations_are_counted_not_stored() {
        let one = r#"{"id":"c","messages":[{"role":"user","content":"x"}]}"#;
        let list = vec![one; MAX_CONVERSATIONS + 5].join(",");
        let text =
            format!(r#"{{"format":"colibri-chat-history","version":1,"conversations":[{list}]}}"#);
        let (v, rejected) = parse(&text).unwrap();
        assert_eq!(v.len(), MAX_CONVERSATIONS);
        assert_eq!(rejected, 5);
    }

    #[test]
    fn exports_to_files_and_does_not_overwrite() {
        let d = std::env::temp_dir().join(format!("hist-files-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let c = [conv("a", 200, "salut"), conv("b", 300, "hei")];
        let p1 = export_file(&d, &c, false).unwrap();
        let p2 = export_file(&d, &c, false).unwrap();
        assert_ne!(p1, p2);
        assert!(looks_like_history(&p1));
        let s = store("filesimport");
        assert_eq!(import_file(&s, &p1).unwrap().added, 2);
        let f = export_markdown_folder(&d, &c, "Eu", "Colibri").unwrap();
        let f2 = export_markdown_folder(&d, &c, "Eu", "Colibri").unwrap();
        assert_ne!(f, f2);
        assert_eq!(std::fs::read_dir(&f).unwrap().count(), 2);
        let other = d.join("other.json");
        std::fs::write(&other, "{\"a\":1}").unwrap();
        assert!(!looks_like_history(&other));
        assert!(!looks_like_history(&d.join("missing.json")));
    }
}
