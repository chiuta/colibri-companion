// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! LaTeX (`\cite[p. 3]{key}`, biblatex variants) and LyX (`.lyx` citation insets) to the Markdown the audit reads
//! (`[@key, p. 3]`). Plain Markdown passes through untouched.
//!
//! This is a reader for citations and headings, not a TeX engine: macros are not expanded, environments are not
//! understood, formatting commands lose their name and keep their argument. Every look-ahead (the closing `]` of an
//! optional argument, the closing `}` of a group) is limited to a window, so an unbalanced brace in a long file
//! cannot make the conversion quadratic.

/// How far (in characters) an argument may run.
const WINDOW: usize = 2_000;

const CITE_CMDS: &[&str] = &[
    "cite",
    "citep",
    "citet",
    "citealp",
    "citeauthor",
    "parencite",
    "textcite",
    "autocite",
    "footcite",
    "smartcite",
    "supercite",
    "nocite",
];

fn cite_group(keys: &str, post: &str) -> String {
    let page: String = post
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let ks: Vec<String> = keys
        .split(',')
        .map(|k| k.trim())
        .filter(|k| !k.is_empty())
        .map(|k| format!("@{k}"))
        .collect();
    if ks.is_empty() {
        return String::new();
    }
    let mut out = ks.join("; ");
    if !page.is_empty() {
        out = format!("{out}, p. {page}");
    }
    format!("[{out}]")
}

/// LaTeX and LyX sources to Markdown; anything else is returned as it is.
pub fn normalize_markup(text: &str) -> String {
    if text.contains("\\begin_body") && text.contains("\\begin_layout") {
        return lyx_to_md(text);
    }
    if text.contains("\\cite")
        || text.contains("\\parencite")
        || text.contains("\\textcite")
        || text.contains("\\begin{document}")
    {
        return latex_to_md(text);
    }
    text.to_string()
}

fn latex_to_md(src: &str) -> String {
    let body = src.split("\\begin{document}").nth(1).unwrap_or(src);
    let mut t: String = body
        .lines()
        .map(|l| {
            let mut prev = ' ';
            let mut cut = l.len();
            for (i, c) in l.char_indices() {
                if c == '%' && prev != '\\' {
                    cut = i;
                    break;
                }
                prev = c;
            }
            &l[..cut]
        })
        .collect::<Vec<_>>()
        .join("\n");
    t = t.replace("``", "“").replace("''", "”").replace("\\\\", " ");
    let cs: Vec<char> = t.chars().collect();
    let (mut out, mut i) = (String::new(), 0);
    while i < cs.len() {
        if cs[i] == '\\' {
            let mut j = i + 1;
            while j < cs.len() && cs[j].is_ascii_alphabetic() {
                j += 1;
            }
            let name: String = cs[i + 1..j].iter().take(40).collect();
            if j < cs.len() && cs[j] == '*' && !name.is_empty() {
                j += 1;
            }
            // optional arguments
            let mut opts: Vec<String> = Vec::new();
            let mut k = j;
            while k < cs.len() && cs[k] == '[' {
                let end = cs.len().min(k + WINDOW);
                if let Some(e) = cs[k..end].iter().position(|&c| c == ']') {
                    opts.push(cs[k + 1..k + e].iter().collect());
                    k += e + 1;
                } else {
                    break;
                }
            }
            let arg = |from: usize| -> Option<(String, usize)> {
                if from < cs.len() && cs[from] == '{' {
                    let end = cs.len().min(from + WINDOW);
                    let mut depth = 0;
                    for (n, &c) in cs[from..end].iter().enumerate() {
                        if c == '{' {
                            depth += 1
                        } else if c == '}' {
                            depth -= 1;
                            if depth == 0 {
                                return Some((
                                    cs[from + 1..from + n].iter().collect(),
                                    from + n + 1,
                                ));
                            }
                        }
                    }
                }
                None
            };
            if CITE_CMDS.contains(&name.as_str())
                && name != "nocite"
                && let Some((keys, end)) = arg(k)
            {
                out.push_str(&cite_group(
                    &keys,
                    opts.last().map(|s| s.as_str()).unwrap_or(""),
                ));
                i = end;
                continue;
            }
            if matches!(
                name.as_str(),
                "section" | "subsection" | "subsubsection" | "chapter" | "paragraph"
            ) && let Some((title, end)) = arg(k)
            {
                out.push_str(&format!("\n\n# {title}\n\n"));
                i = end;
                continue;
            }
            if matches!(
                name.as_str(),
                "begin"
                    | "end"
                    | "label"
                    | "ref"
                    | "bibliography"
                    | "bibliographystyle"
                    | "usepackage"
                    | "documentclass"
                    | "maketitle"
                    | "newcommand"
            ) && let Some((_, end)) = arg(k)
            {
                out.push(' ');
                i = end;
                continue;
            }
            if name.is_empty() {
                // escaped symbol such as \& or \%
                if j < cs.len() {
                    out.push(cs[j]);
                    i = j + 1;
                } else {
                    i = j;
                }
                continue;
            }
            i = j; // formatting command: drop the name, keep its argument text
            continue;
        }
        if cs[i] == '{' || cs[i] == '}' {
            i += 1;
            continue;
        }
        if cs[i] == '~' {
            out.push(' ');
            i += 1;
            continue;
        }
        out.push(cs[i]);
        i += 1;
    }
    out
}

fn lyx_to_md(src: &str) -> String {
    let body = src.split("\\begin_body").nth(1).unwrap_or(src);
    let (mut out, mut heading, mut start) = (String::new(), false, true);
    let mut lines = body.lines().peekable();
    while let Some(l) = lines.next() {
        let l = l.trim_end();
        if let Some(rest) = l.strip_prefix("\\begin_layout ") {
            heading = matches!(
                rest.trim(),
                "Chapter"
                    | "Section"
                    | "Subsection"
                    | "Subsubsection"
                    | "Part"
                    | "Title"
                    | "Paragraph"
                    | "Section*"
                    | "Subsection*"
                    | "Chapter*"
            );
            start = true;
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            continue;
        }
        if l.starts_with("\\begin_inset CommandInset citation") {
            let (mut key, mut after) = (String::new(), String::new());
            for x in lines.by_ref() {
                if x.starts_with("\\end_inset") {
                    break;
                }
                let q = |p: &str| {
                    x.strip_prefix(p)
                        .map(|v| v.trim().trim_matches('"').to_string())
                };
                if let Some(v) = q("key ") {
                    key = v;
                }
                if let Some(v) = q("after ") {
                    after = v;
                }
            }
            out.push_str(&cite_group(&key, &after));
            continue;
        }
        if let Some(q) = l.strip_prefix("\\begin_inset Quotes ") {
            let kind = q.trim();
            out.push(if kind.chars().nth(1) == Some('l') {
                '“'
            } else {
                '”'
            });
            for x in lines.by_ref() {
                if x.starts_with("\\end_inset") {
                    break;
                }
            }
            continue;
        }
        if l.starts_with("\\begin_inset") {
            // footnotes, floats...: skip nested content except plain text lines
            continue;
        }
        if l.starts_with('\\') || l.starts_with('#') || l.is_empty() {
            continue;
        }
        if heading && start {
            out.push_str("# ");
        }
        if !start && !out.ends_with(' ') && !out.ends_with('“') {
            out.push(' ');
        }
        out.push_str(l);
        start = false;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_passes_through() {
        let s = "A claim [@smith2019, p. 3].";
        assert_eq!(normalize_markup(s), s);
    }

    #[test]
    fn latex_citations_become_pandoc_groups() {
        let t = normalize_markup(
            "\\begin{document}\n\\section{Intro}\nThe rule \\cite[p. 12]{smith2019,doe2020} applies. % comment \\cite{hidden}\n100\\% sure~now \\textbf{bold}.",
        );
        assert!(t.contains("# Intro"), "{t}");
        assert!(t.contains("[@smith2019; @doe2020, p. 12]"), "{t}");
        assert!(!t.contains("hidden"));
        assert!(t.contains("100% sure now bold"), "{t}");
    }

    #[test]
    fn lyx_citation_insets_are_read() {
        let lyx = "\\begin_body\n\\begin_layout Section\nIntro\n\\end_layout\n\\begin_layout Standard\nThe rule\n\\begin_inset CommandInset citation\nLatexCommand cite\nafter \"p. 7\"\nkey \"smith2019\"\n\\end_inset\n applies.\n\\end_layout\n\\end_body";
        let t = normalize_markup(lyx);
        assert!(t.contains("# Intro"), "{t}");
        assert!(t.contains("[@smith2019, p. 7]"), "{t}");
    }

    #[test]
    fn unbalanced_input_is_bounded() {
        let t = std::time::Instant::now();
        let a = "\\cite{".repeat(50_000);
        let b = "\\cite[".repeat(50_000);
        let c = "{".repeat(100_000);
        for s in [a, b, c] {
            let _ = normalize_markup(&format!("\\begin{{document}}{s}"));
        }
        assert!(t.elapsed().as_secs() < 10, "{:?}", t.elapsed());
    }
}
