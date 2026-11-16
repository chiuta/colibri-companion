//! Dropping a text file into the chat: read it safely and turn it into a fenced block
//! that goes into the message, so the model can see its content.

use std::fs;
use std::path::Path;

/// Larger files would swallow the model's context for little reason.
pub const MAX_BYTES: u64 = 256 * 1024;

/// The text of a file, or `None` when it is too large, unreadable, or not text.
pub fn read_text_file(path: &Path) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > MAX_BYTES {
        return None;
    }
    let bytes = fs::read(path).ok()?;
    if bytes.contains(&0) {
        return None;
    }
    let text = String::from_utf8(bytes).ok()?;
    Some(
        text.strip_prefix('\u{feff}')
            .map(str::to_owned)
            .unwrap_or(text),
    )
}

/// `**name**` followed by the text in a code fence long enough not to be closed by the text itself.
pub fn fenced(name: &str, text: &str) -> String {
    let mut fence = String::from("```");
    while text.contains(&fence) {
        fence.push('`');
    }
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .filter(|e| e.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or("");
    format!(
        "**{name}**\n{fence}{ext}\n{}\n{fence}\n",
        text.trim_end_matches('\n')
    )
}

// ------------------------------------------------------------------ every kind of file

use crate::locate::PythonCmd;
use crate::proc::{hide_window, run_capture};
use std::process::Command;
use std::time::Duration;

/// Text sent to the model is cut here (about 50k tokens), with a note, rather than refused.
pub const MAX_CHARS: usize = 3_000_000;
/// Pictures larger than this are refused: they would only slow every later message down.
pub const MAX_IMAGE_BYTES: u64 = 10 * 1024 * 1024;
/// Office and PDF files are unpacked by a helper; bigger than this is not tried.
pub const MAX_DOC_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Image,
    Pdf,
    Office,
    Html,
    /// Anything else: read as text when it is text.
    Text,
}

pub fn kind_of(path: &Path) -> Kind {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" => Kind::Image,
        "pdf" => Kind::Pdf,
        "docx" | "pptx" | "xlsx" | "odt" | "ods" | "odp" => Kind::Office,
        "html" | "htm" | "xhtml" => Kind::Html,
        _ => Kind::Text,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttachError {
    Unreadable,
    TooLarge,
    NotText,
    /// The helper that unpacks this kind of file is missing; says what to install.
    NeedsTool(&'static str),
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extracted {
    pub text: String,
    pub truncated: bool,
}

/// Cut on a character boundary, and say so.
pub fn limit(text: String) -> Extracted {
    if text.chars().count() <= MAX_CHARS {
        return Extracted {
            text,
            truncated: false,
        };
    }
    let cut: String = text.chars().take(MAX_CHARS).collect();
    Extracted {
        text: format!("{cut}\n[… cut here: the rest of the file is not included …]"),
        truncated: true,
    }
}

/// A picture as `(mime type, bytes)`, checked by its first bytes and not by its name.
pub fn read_image(path: &Path) -> Result<(&'static str, Vec<u8>), AttachError> {
    let meta = fs::metadata(path).map_err(|_| AttachError::Unreadable)?;
    if !meta.is_file() {
        return Err(AttachError::Unreadable);
    }
    if meta.len() > MAX_IMAGE_BYTES {
        return Err(AttachError::TooLarge);
    }
    let bytes = fs::read(path).map_err(|_| AttachError::Unreadable)?;
    let mime = image_mime(&bytes).ok_or(AttachError::NotText)?;
    Ok((mime, bytes))
}

pub fn image_mime(b: &[u8]) -> Option<&'static str> {
    if b.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("image/png")
    } else if b.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if b.starts_with(b"GIF8") {
        Some("image/gif")
    } else if b.len() > 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        Some("image/webp")
    } else if b.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

/// Standard base64, for sending a picture as a `data:` URL.
pub fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

pub fn data_url(mime: &str, bytes: &[u8]) -> String {
    format!("data:{mime};base64,{}", base64(bytes))
}

/// Visible text of an HTML page: scripts, styles and tags dropped, a few entities decoded.
pub fn html_to_text(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    // Where the last closing tag of each kind is: an unclosed `<script>` repeated thousands of
    // times must not make every one of them search to the end of the page.
    let last_close: Vec<Option<usize>> = ["script", "style", "head"]
        .iter()
        .map(|t| lower.rfind(&format!("</{t}")))
        .collect();
    let mut out = String::with_capacity(html.len() / 2);
    let (mut i, bytes) = (0, html.as_bytes());
    while i < bytes.len() {
        if bytes[i] == b'<' {
            let rest = &lower[i..];
            let skip_to = ["script", "style", "head"]
                .iter()
                .enumerate()
                .find_map(|(n, t)| {
                    rest.strip_prefix('<')
                        .filter(|r| {
                            r.starts_with(t) && r[t.len()..].starts_with(['>', ' ', '\t', '\n'])
                        })
                        .filter(|_| last_close[n].is_some_and(|l| l >= i))
                        .and_then(|_| lower[i..].find(&format!("</{t}")).map(|e| i + e))
                });
            if let Some(end) = skip_to {
                i = lower[end..].find('>').map_or(bytes.len(), |g| end + g + 1);
                continue;
            }
            let block = [
                "<br", "<p", "</p", "<div", "</div", "<li", "<tr", "<h1", "<h2", "<h3",
            ]
            .iter()
            .any(|t| rest.starts_with(t));
            i = html[i..].find('>').map_or(bytes.len(), |g| i + g + 1);
            if block && !out.ends_with('\n') {
                out.push('\n');
            }
            continue;
        }
        let ch = html[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&");
    let mut lines: Vec<&str> = decoded.lines().map(str::trim).collect();
    lines.dedup_by(|a, b| a.is_empty() && b.is_empty());
    lines.join("\n").trim().to_string()
}

const OFFICE_SCRIPT: &str = r#"
import re, sys, zipfile, html
path = sys.argv[1]
z = zipfile.ZipFile(path)
names = z.namelist()
LIMIT = 100 * 1024 * 1024
TOTAL = [0]
def rd(n):
    info = z.getinfo(n)
    TOTAL[0] += info.file_size
    if info.file_size > LIMIT or TOTAL[0] > 3 * LIMIT:
        sys.exit("a part inside this file is far bigger than it should be (refused)")
    return z.read(n)
def text_of(xml):
    xml = re.sub(r"</(w:p|a:p|text:p|text:h|row|table:table-row)>", "\n", xml)
    xml = re.sub(r"<w:tab/>|</(a:tc|table:table-cell)>", "\t", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(xml)
out = []
if "word/document.xml" in names:
    out.append(text_of(rd("word/document.xml").decode("utf-8", "replace")))
elif any(n.startswith("ppt/slides/slide") for n in names):
    slides = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[0]))
    for i, n in enumerate(slides, 1):
        out.append("--- slide %d ---\n%s" % (i, text_of(rd(n).decode("utf-8", "replace"))))
elif "xl/workbook.xml" in names:
    shared = []
    if "xl/sharedStrings.xml" in names:
        for si in re.findall(r"<si>(.*?)</si>", rd("xl/sharedStrings.xml").decode("utf-8", "replace"), re.S):
            shared.append(html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S))))
    sheets = sorted(n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
    for n in sheets:
        out.append("--- %s ---" % n.split("/")[-1])
        for row in re.findall(r"<row[^>]*>(.*?)</row>", rd(n).decode("utf-8", "replace"), re.S):
            cells = []
            for attrs, body in re.findall(r"<c([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
                v = re.search(r"<v>(.*?)</v>", body or "", re.S)
                val = v.group(1) if v else ""
                if 't="s"' in attrs and val.isdigit() and int(val) < len(shared):
                    val = shared[int(val)]
                elif 't="inlineStr"' in attrs:
                    val = html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", body or "", re.S)))
                cells.append(val)
            out.append("\t".join(cells))
elif "content.xml" in names:
    out.append(text_of(rd("content.xml").decode("utf-8", "replace")))
else:
    sys.exit("not a document this reader knows")
sys.stdout.buffer.write("\n".join(out).encode("utf-8"))
"#;

const PDF_SCRIPT: &str = r#"
import sys
try:
    from pypdf import PdfReader
except ImportError:
    sys.exit(3)
r = PdfReader(sys.argv[1])
sys.stdout.buffer.write("\n\n".join((p.extract_text() or "") for p in r.pages).encode("utf-8"))
"#;

/// A relative file name that starts with `-` would be read as an option by the helper programs.
fn dash_safe(path: &Path) -> std::path::PathBuf {
    if path.is_relative() && path.to_string_lossy().starts_with('-') {
        Path::new(".").join(path)
    } else {
        path.to_path_buf()
    }
}

fn pdf_text(python: Option<&PythonCmd>, path: &Path) -> Result<String, AttachError> {
    // poppler first: fast and the most faithful to the layout
    let mut c = Command::new("pdftotext");
    c.args(["-layout", "-q"]).arg(dash_safe(path)).arg("-");
    hide_window(&mut c);
    if let Ok(out) = run_capture(c, Duration::from_secs(60)) {
        if out.status.success() {
            return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
        }
    }
    let py = python.ok_or(AttachError::NeedsTool("pdftotext or pypdf"))?;
    let mut c = py.command();
    c.args(["-I", "-c", PDF_SCRIPT]).arg(dash_safe(path));
    hide_window(&mut c);
    let out =
        run_capture(c, Duration::from_secs(90)).map_err(|e| AttachError::Failed(e.to_string()))?;
    match out.status.code() {
        Some(0) => Ok(String::from_utf8_lossy(&out.stdout).into_owned()),
        Some(3) => Err(AttachError::NeedsTool("pdftotext or pypdf")),
        _ => Err(AttachError::Failed(
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("")
                .to_string(),
        )),
    }
}

fn office_text(python: &PythonCmd, path: &Path) -> Result<String, AttachError> {
    let mut c = python.command();
    c.args(["-I", "-c", OFFICE_SCRIPT]).arg(dash_safe(path));
    hide_window(&mut c);
    let out =
        run_capture(c, Duration::from_secs(60)).map_err(|e| AttachError::Failed(e.to_string()))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(AttachError::Failed(
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("")
                .to_string(),
        ))
    }
}

/// The text of any supported file (not pictures). Long text is cut, with a note.
pub fn extract_text(python: Option<&PythonCmd>, path: &Path) -> Result<Extracted, AttachError> {
    let meta = fs::metadata(path).map_err(|_| AttachError::Unreadable)?;
    if !meta.is_file() {
        return Err(AttachError::Unreadable);
    }
    let kind = kind_of(path);
    let big = match kind {
        Kind::Pdf | Kind::Office => MAX_DOC_BYTES,
        _ => 64 * 1024 * 1024,
    };
    if meta.len() > big {
        return Err(AttachError::TooLarge);
    }
    let text = match kind {
        Kind::Image => return Err(AttachError::NotText),
        Kind::Pdf => pdf_text(python, path)?,
        Kind::Office => office_text(python.ok_or(AttachError::NeedsTool("Python"))?, path)?,
        Kind::Html => {
            let bytes = fs::read(path).map_err(|_| AttachError::Unreadable)?;
            html_to_text(&String::from_utf8_lossy(&bytes))
        }
        Kind::Text => {
            let bytes = fs::read(path).map_err(|_| AttachError::Unreadable)?;
            if bytes.contains(&0) {
                return Err(AttachError::NotText);
            }
            let text = String::from_utf8(bytes).map_err(|_| AttachError::NotText)?;
            text.strip_prefix('\u{feff}')
                .map(str::to_owned)
                .unwrap_or(text)
        }
    };
    if text.trim().is_empty() {
        return Err(AttachError::Failed("no text found in the file".into()));
    }
    Ok(limit(text))
}

// ------------------------------------------------------------------ folders and big files

/// Folders that only hold generated or foreign files.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".cache",
    ".idea",
    ".vscode",
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Collected {
    pub files: Vec<std::path::PathBuf>,
    /// Hidden files and folders and well-known junk folders that were left out.
    pub skipped: usize,
    /// Stopped at the limit: there were more files.
    pub too_many: bool,
}

/// The files behind a list of dropped paths: files as they are, folders walked recursively
/// (sorted, hidden and junk skipped, symbolic links never followed), at most `max` files.
pub fn collect(roots: &[std::path::PathBuf], max: usize) -> Collected {
    let mut out = Collected::default();
    fn walk(dir: &Path, max: usize, out: &mut Collected, depth: usize) {
        let Ok(rd) = fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            if out.too_many {
                return;
            }
            let name = e.file_name().to_string_lossy().into_owned();
            let Ok(ft) = e.file_type() else {
                continue;
            };
            if ft.is_symlink() || name.starts_with('.') {
                out.skipped += 1;
                continue;
            }
            let path = e.path();
            if ft.is_dir() {
                if SKIP_DIRS.contains(&name.as_str()) || depth >= 12 {
                    out.skipped += 1;
                } else {
                    walk(&path, max, out, depth + 1);
                }
            } else if ft.is_file() {
                if out.files.len() >= max {
                    out.too_many = true;
                    return;
                }
                out.files.push(path);
            }
        }
    }
    for r in roots {
        if out.too_many {
            break;
        }
        match fs::symlink_metadata(r) {
            Ok(m) if m.is_dir() => walk(r, max, &mut out, 0),
            Ok(m) if m.is_file() => {
                if out.files.len() >= max {
                    out.too_many = true;
                } else {
                    out.files.push(r.clone());
                }
            }
            _ => {}
        }
    }
    out
}

/// Cuts `text` into pieces of at most `max_chars`, at line breaks where possible, so a big
/// file can be sent over several messages.
pub fn split_chunks(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(1000);
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_chars = 0usize;
    for line in text.split_inclusive('\n') {
        let n = line.chars().count();
        if n > max_chars {
            // one enormous line: hard cut
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
                cur_chars = 0;
            }
            let chars: Vec<char> = line.chars().collect();
            for part in chars.chunks(max_chars) {
                out.push(part.iter().collect());
            }
            continue;
        }
        if cur_chars + n > max_chars && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            cur_chars = 0;
        }
        cur.push_str(line);
        cur_chars += n;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// What to send for a question plus attached documents: one message when everything fits in
/// `budget` characters, otherwise the documents in numbered parts (each asking only for "OK")
/// followed by the question.
pub fn plan_parts(question: &str, docs: &[(String, String)], budget: usize) -> Vec<String> {
    let body: String = docs
        .iter()
        .map(|(n, t)| fenced(n, t))
        .collect::<Vec<_>>()
        .join("\n");
    let q = question.trim();
    let whole = if body.is_empty() {
        q.to_string()
    } else if q.is_empty() {
        body.clone()
    } else {
        format!("{body}\n{q}")
    };
    if whole.chars().count() <= budget || body.is_empty() {
        return vec![whole];
    }
    let room = budget.saturating_sub(300).max(2000);
    let chunks = split_chunks(&body, room);
    let n = chunks.len();
    let mut out: Vec<String> = chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "[Part {}/{n} of the attached files. Do not answer yet: reply only with OK.]\n{c}",
                i + 1
            )
        })
        .collect();
    out.push(if q.is_empty() {
        format!("That was all {n} parts of the files. Summarise them.")
    } else {
        format!("That was all {n} parts of the files. Now answer: {q}")
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_text_and_refuses_the_rest() {
        let dir = std::env::temp_dir().join(format!("cc-attach-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "\u{feff}salut\n").unwrap();
        fs::write(dir.join("b.bin"), [1u8, 0, 2]).unwrap();
        fs::write(dir.join("c.txt"), vec![b'x'; MAX_BYTES as usize + 1]).unwrap();
        fs::write(dir.join("d.txt"), [0xffu8, 0xfe, 0x41]).unwrap();
        assert_eq!(
            read_text_file(&dir.join("a.txt")).as_deref(),
            Some("salut\n")
        );
        assert!(read_text_file(&dir.join("b.bin")).is_none());
        assert!(read_text_file(&dir.join("c.txt")).is_none());
        assert!(read_text_file(&dir.join("d.txt")).is_none());
        assert!(read_text_file(&dir).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_fence_is_never_closed_by_the_content() {
        let out = fenced("notes.md", "a\n```rust\nfn x() {}\n```\n");
        assert!(out.starts_with("**notes.md**\n````md\n"));
        assert!(out.ends_with("\n````\n"));
        assert_eq!(fenced("x", "hi"), "**x**\n```\nhi\n```\n");
    }

    #[test]
    fn kinds_and_image_signatures() {
        assert_eq!(kind_of(Path::new("A.PNG")), Kind::Image);
        assert_eq!(kind_of(Path::new("r.pdf")), Kind::Pdf);
        assert_eq!(kind_of(Path::new("t.docx")), Kind::Office);
        assert_eq!(kind_of(Path::new("p.html")), Kind::Html);
        assert_eq!(kind_of(Path::new("main.rs")), Kind::Text);
        assert_eq!(image_mime(&[0x89, b'P', b'N', b'G', 1]), Some("image/png"));
        assert_eq!(image_mime(&[0xff, 0xd8, 0xff, 0xe0]), Some("image/jpeg"));
        assert_eq!(image_mime(b"RIFF\0\0\0\0WEBPVP8 "), Some("image/webp"));
        assert_eq!(image_mime(b"not an image"), None);
    }

    #[test]
    fn base64_matches_the_standard_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(data_url("image/png", b"foo"), "data:image/png;base64,Zm9v");
    }

    #[test]
    fn html_becomes_readable_text() {
        let t = html_to_text(
            "<html><head><title>x</title><style>p{}</style></head><body><h1>Titlu</h1>\
             <script>alert(1)</script><p>Una &amp; alta</p><div>doi<br>trei</div></body></html>",
        );
        assert_eq!(t, "Titlu\nUna & alta\ndoi\ntrei");
        assert!(!t.contains("alert") && !t.contains("p{}"));
    }

    #[test]
    fn long_text_is_cut_not_refused() {
        let e = limit("é".repeat(MAX_CHARS + 10));
        assert!(e.truncated && e.text.contains("cut here"));
        assert!(!limit("short".into()).truncated);
    }

    #[test]
    fn a_picture_is_recognised_by_content_and_size_checked() {
        let dir = std::env::temp_dir().join(format!("cc-img-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.png"), [0x89, b'P', b'N', b'G', 0, 0]).unwrap();
        fs::write(dir.join("fake.png"), b"just text").unwrap();
        assert_eq!(read_image(&dir.join("a.png")).unwrap().0, "image/png");
        assert_eq!(
            read_image(&dir.join("fake.png")).unwrap_err(),
            AttachError::NotText
        );
        assert_eq!(
            read_image(&dir.join("none.png")).unwrap_err(),
            AttachError::Unreadable
        );
    }

    #[test]
    fn office_files_are_unpacked() {
        let Some(py) = crate::locate::find_python() else {
            eprintln!("python not available: skipping");
            return;
        };
        let dir = std::env::temp_dir().join(format!("cc-office-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let make = r#"
import sys, zipfile
d = sys.argv[1]
z = zipfile.ZipFile(d + "/a.docx", "w")
z.writestr("word/document.xml", "<w:document><w:body><w:p><w:r><w:t>Salut lume ăîșț</w:t></w:r></w:p><w:p><w:r><w:t>A doua &amp; ultima</w:t></w:r></w:p></w:body></w:document>")
z.close()
z = zipfile.ZipFile(d + "/b.xlsx", "w")
z.writestr("xl/workbook.xml", "<workbook/>")
z.writestr("xl/sharedStrings.xml", "<sst><si><t>Nume</t></si><si><t>Ana</t></si></sst>")
z.writestr("xl/worksheets/sheet1.xml", '<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>42</v></c></row><row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>7</v></c></row></sheetData></worksheet>')
z.close()
z = zipfile.ZipFile(d + "/c.pptx", "w")
z.writestr("ppt/slides/slide1.xml", "<p:sld><a:p><a:r><a:t>Primul slide</a:t></a:r></a:p></p:sld>")
z.close()
"#;
        let mut c = py.command();
        c.args(["-c", make]).arg(&dir);
        assert!(c.status().unwrap().success());
        let doc = extract_text(Some(&py), &dir.join("a.docx")).unwrap().text;
        assert!(doc.contains("Salut lume ăîșț\nA doua & ultima"), "{doc:?}");
        let sheet = extract_text(Some(&py), &dir.join("b.xlsx")).unwrap().text;
        assert!(
            sheet.contains("Nume\t42") && sheet.contains("Ana\t7"),
            "{sheet:?}"
        );
        let deck = extract_text(Some(&py), &dir.join("c.pptx")).unwrap().text;
        assert!(
            deck.contains("--- slide 1 ---") && deck.contains("Primul slide"),
            "{deck:?}"
        );
        // a zip that is not a document fails with a message instead of garbage
        fs::write(dir.join("bad.docx"), b"not a zip").unwrap();
        assert!(matches!(
            extract_text(Some(&py), &dir.join("bad.docx")),
            Err(AttachError::Failed(_))
        ));
        // without Python, an Office file says what is missing
        assert_eq!(
            extract_text(None, &dir.join("a.docx")).unwrap_err(),
            AttachError::NeedsTool("Python")
        );
    }

    #[test]
    fn folders_are_walked_and_junk_skipped() {
        let dir = std::env::temp_dir().join(format!("cc-collect-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        for d in ["a", "a/b", ".git", "node_modules", "z"] {
            fs::create_dir_all(dir.join(d)).unwrap();
        }
        for f in [
            "a/1.txt",
            "a/b/2.md",
            "z/3.pdf",
            ".git/x",
            "node_modules/y.js",
            ".hidden",
            "top.csv",
        ] {
            fs::write(dir.join(f), b"x").unwrap();
        }
        let c = collect(std::slice::from_ref(&dir), 100);
        let names: Vec<String> = c
            .files
            .iter()
            .map(|p| {
                p.strip_prefix(&dir)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        assert_eq!(names, ["a/1.txt", "a/b/2.md", "top.csv", "z/3.pdf"]);
        assert!(c.skipped >= 3 && !c.too_many);
        let few = collect(std::slice::from_ref(&dir), 2);
        assert!(few.too_many && few.files.len() == 2);
        let mixed = collect(&[dir.join("top.csv"), dir.join("a")], 10);
        assert_eq!(mixed.files.len(), 3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn big_text_is_split_at_lines_and_loses_nothing() {
        let text: String = (0..500).map(|i| format!("line number {i}\n")).collect();
        let parts = split_chunks(&text, 1500);
        assert!(parts.len() > 3);
        assert!(parts.iter().all(|p| p.chars().count() <= 1500));
        assert_eq!(parts.concat(), text);
        let one_line = "x".repeat(4500);
        let p2 = split_chunks(&one_line, 2000);
        assert_eq!(p2.concat(), one_line);
        assert_eq!(p2.len(), 3);
        assert!(split_chunks("", 2000).is_empty());
    }

    #[test]
    fn small_input_is_one_message_and_big_input_is_numbered_parts() {
        let docs = vec![("a.txt".to_string(), "hello\n".to_string())];
        assert_eq!(plan_parts("what?", &docs, 10_000).len(), 1);
        assert_eq!(plan_parts("only text", &[], 10).len(), 1);
        let big = vec![("b.txt".to_string(), "line of text\n".repeat(2000))];
        let parts = plan_parts("summarise", &big, 5000);
        assert!(parts.len() > 3);
        assert!(parts[0].contains("Part 1/"));
        assert!(parts.last().unwrap().ends_with("summarise"));
        assert!(
            parts[..parts.len() - 1]
                .iter()
                .all(|p| p.chars().count() <= 5000)
        );
    }

    #[test]
    fn unclosed_tags_by_the_thousand_do_not_take_forever() {
        let page = "<script>x ".repeat(20_000)
            + "<style>y".repeat(20_000).as_str()
            + "<head >".repeat(20_000).as_str()
            + "end";
        let t0 = std::time::Instant::now();
        let text = html_to_text(&page);
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(5),
            "{:?}",
            t0.elapsed()
        );
        assert!(text.contains("end"));
    }

    #[test]
    fn a_dash_name_is_not_taken_for_an_option() {
        assert_eq!(dash_safe(Path::new("-opw")), Path::new("./-opw"));
        assert_eq!(dash_safe(Path::new("/tmp/-opw")), Path::new("/tmp/-opw"));
    }
}
