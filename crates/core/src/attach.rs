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
pub const MAX_CHARS: usize = 200_000;
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
    let mut out = String::with_capacity(html.len() / 2);
    let (mut i, bytes) = (0, html.as_bytes());
    while i < bytes.len() {
        if bytes[i] == b'<' {
            let rest = &lower[i..];
            let skip_to = ["script", "style", "head"].iter().find_map(|t| {
                rest.strip_prefix('<')
                    .filter(|r| {
                        r.starts_with(t) && r[t.len()..].starts_with(['>', ' ', '\t', '\n'])
                    })
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
def text_of(xml):
    xml = re.sub(r"</(w:p|a:p|text:p|text:h|row|table:table-row)>", "\n", xml)
    xml = re.sub(r"<w:tab/>|</(a:tc|table:table-cell)>", "\t", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(xml)
out = []
if "word/document.xml" in names:
    out.append(text_of(z.read("word/document.xml").decode("utf-8", "replace")))
elif any(n.startswith("ppt/slides/slide") for n in names):
    slides = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[0]))
    for i, n in enumerate(slides, 1):
        out.append("--- slide %d ---\n%s" % (i, text_of(z.read(n).decode("utf-8", "replace"))))
elif "xl/workbook.xml" in names:
    shared = []
    if "xl/sharedStrings.xml" in names:
        for si in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode("utf-8", "replace"), re.S):
            shared.append(html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S))))
    sheets = sorted(n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
    for n in sheets:
        out.append("--- %s ---" % n.split("/")[-1])
        for row in re.findall(r"<row[^>]*>(.*?)</row>", z.read(n).decode("utf-8", "replace"), re.S):
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
    out.append(text_of(z.read("content.xml").decode("utf-8", "replace")))
else:
    sys.exit("not a document this reader knows")
sys.stdout.write("\n".join(out))
"#;

const PDF_SCRIPT: &str = r#"
import sys
try:
    from pypdf import PdfReader
except ImportError:
    sys.exit(3)
r = PdfReader(sys.argv[1])
sys.stdout.write("\n\n".join((p.extract_text() or "") for p in r.pages))
"#;

fn pdf_text(python: Option<&PythonCmd>, path: &Path) -> Result<String, AttachError> {
    // poppler first: fast and the most faithful to the layout
    let mut c = Command::new("pdftotext");
    c.args(["-layout", "-q"]).arg(path).arg("-");
    hide_window(&mut c);
    if let Ok(out) = run_capture(c, Duration::from_secs(60)) {
        if out.status.success() {
            return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
        }
    }
    let py = python.ok_or(AttachError::NeedsTool("pdftotext or pypdf"))?;
    let mut c = py.command();
    c.args(["-I", "-c", PDF_SCRIPT]).arg(path);
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
    c.args(["-I", "-c", OFFICE_SCRIPT]).arg(path);
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
        _ => 8 * 1024 * 1024,
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
z.writestr("word/document.xml", "<w:document><w:body><w:p><w:r><w:t>Salut lume</w:t></w:r></w:p><w:p><w:r><w:t>A doua &amp; ultima</w:t></w:r></w:p></w:body></w:document>")
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
        assert!(doc.contains("Salut lume\nA doua & ultima"), "{doc:?}");
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
}
