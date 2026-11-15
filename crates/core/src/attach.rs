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
}
