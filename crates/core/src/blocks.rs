//! Code blocks inside an answer: found, named and saved, so the friendly buttons under a
//! message ("Copy", "Save as file", "Open preview") know what they act on.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub lang: String,
    pub code: String,
}

impl Block {
    /// Proper file extension for the block's language tag.
    pub fn extension(&self) -> &str {
        match self.lang.to_ascii_lowercase().as_str() {
            "rust" | "rs" => "rs",
            "python" | "py" => "py",
            "javascript" | "js" => "js",
            "typescript" | "ts" => "ts",
            "html" | "htm" => "html",
            "css" => "css",
            "json" => "json",
            "csv" => "csv",
            "svg" => "svg",
            "markdown" | "md" => "md",
            "bash" | "sh" | "shell" => "sh",
            "yaml" | "yml" => "yml",
            "xml" => "xml",
            "sql" => "sql",
            "c" => "c",
            "cpp" | "c++" => "cpp",
            "java" => "java",
            "go" => "go",
            _ => "txt",
        }
    }

    /// Kinds that can be looked at outside the chat (a browser shows them).
    pub fn is_previewable(&self) -> bool {
        matches!(self.extension(), "html" | "svg" | "md")
    }

    /// HTML and SVG can run scripts, so the app asks before opening them.
    pub fn is_active(&self) -> bool {
        matches!(self.extension(), "html" | "svg")
    }
}

/// Fenced blocks (``` or ~~~, any length ≥ 3) in order; an unclosed block at the end of a
/// half-streamed answer is ignored.
pub fn extract(md: &str) -> Vec<Block> {
    let mut out = Vec::new();
    let mut open: Option<(char, usize, String, Vec<&str>)> = None;
    for line in md.lines() {
        let t = line.trim_start();
        let ch = t.chars().next();
        if let Some(c) = ch.filter(|c| *c == '`' || *c == '~') {
            let n = t.chars().take_while(|x| *x == c).count();
            if n >= 3 {
                match &open {
                    None => {
                        let lang = t[n..].split_whitespace().next().unwrap_or("").to_string();
                        open = Some((c, n, lang, Vec::new()));
                        continue;
                    }
                    Some((oc, on, _, _)) if *oc == c && n >= *on && t[n..].trim().is_empty() => {
                        let (_, _, lang, lines) = open.take().unwrap();
                        out.push(Block {
                            lang,
                            code: lines.join("\n"),
                        });
                        continue;
                    }
                    _ => {}
                }
            }
        }
        if let Some((_, _, _, lines)) = open.as_mut() {
            lines.push(line);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_blocks_and_ignores_the_unclosed_one() {
        let md = "text\n```rust\nfn a() {}\n```\nmore\n~~~html\n<b>x</b>\n~~~\n```py\nprint(1)";
        let b = extract(md);
        assert_eq!(b.len(), 2);
        assert_eq!(
            (b[0].lang.as_str(), b[0].code.as_str()),
            ("rust", "fn a() {}")
        );
        assert_eq!(b[1].extension(), "html");
        assert!(b[1].is_previewable() && b[1].is_active());
        assert!(!b[0].is_previewable());
    }

    #[test]
    fn a_longer_fence_is_not_closed_by_a_shorter_one() {
        let b = extract("````md\n```rust\nx\n```\n````\n");
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].code, "```rust\nx\n```");
        assert_eq!(b[0].extension(), "md");
    }
}
