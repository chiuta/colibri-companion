//! Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf).
//! Citation hygiene: which keys a text cites, which of them the bibliography lacks, and which
//! entries nothing cites. Understands pandoc (`[@a; @b, p. 3]`, bare `@a`) and LaTeX
//! (`\cite{a,b}`, `\citep[p.~3]{a}`, biblatex `\parencite`, `\nocite` ...). Linear time.

use crate::bib::Entry;
use std::collections::{BTreeSet, HashSet};

/// Most keys one text may report (a hostile text cannot grow the report without bound).
pub const MAX_KEYS: usize = 100_000;

const CITE_COMMANDS: &[&str] = &[
    "cite",
    "citep",
    "citet",
    "citealp",
    "citealt",
    "citeauthor",
    "citeyear",
    "citeyearpar",
    "citenum",
    "parencite",
    "textcite",
    "autocite",
    "smartcite",
    "footcite",
    "supercite",
    "fullcite",
    "nocite",
    "cites",
    "parencites",
    "textcites",
    "autocites",
];

fn is_key_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | ':' | '.' | '-' | '/' | '+')
}

/// Trims what the key charset swallows but that ends a sentence, not a key.
fn trim_key(k: &str) -> &str {
    k.trim_end_matches(['.', ':', '-', '/', '+'])
}

/// Longest `[..]` or `{..}` group looked at; a longer one is not a citation (and a scan for a
/// closing bracket that never comes would otherwise make hostile text quadratic).
const MAX_GROUP: usize = 2048;

fn find_within(c: &[char], from: usize, close: char) -> Option<usize> {
    c[from..c.len().min(from + MAX_GROUP)]
        .iter()
        .position(|&x| x == close)
}

/// Every cited key, in order of first use, without repeats.
pub fn used_keys(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut add = |k: &str, out: &mut Vec<String>| {
        let k = trim_key(k);
        if !k.is_empty() && out.len() < MAX_KEYS && seen.insert(k.to_string()) {
            out.push(k.to_string());
        }
    };
    let mut i = 0;
    while i < c.len() {
        match c[i] {
            '@' if i == 0
                || !(c[i - 1].is_alphanumeric() || matches!(c[i - 1], '_' | '@' | '\\')) =>
            {
                // pandoc: @key, also inside [..]; "@" followed by "{" is a BibTeX entry header, not a citation
                let st = i + 1;
                let mut e = st;
                while e < c.len() && is_key_char(c[e]) {
                    e += 1;
                }
                if e > st && (c[st].is_alphanumeric() || c[st] == '_') && c.get(e) != Some(&'{') {
                    let k: String = c[st..e].iter().collect();
                    add(&k, &mut out);
                }
                i = e.max(i + 1);
            }
            '\\' => {
                let st = i + 1;
                let mut e = st;
                while e < c.len() && c[e].is_ascii_alphabetic() {
                    e += 1;
                }
                let name: String = c[st..e].iter().collect();
                i = e.max(i + 1);
                if !CITE_COMMANDS.contains(&name.as_str()) {
                    continue;
                }
                if c.get(i) == Some(&'*') {
                    i += 1;
                }
                // optional [..] arguments, then the {keys} group (repeated for \cites)
                loop {
                    while i < c.len() && c[i].is_whitespace() {
                        i += 1;
                    }
                    if c.get(i) == Some(&'[') {
                        match find_within(&c, i, ']') {
                            Some(p) => i += p + 1,
                            None => break,
                        }
                    } else if c.get(i) == Some(&'{') {
                        let Some(p) = find_within(&c, i, '}') else {
                            break;
                        };
                        let inner: String = c[i + 1..i + p].iter().collect();
                        for k in inner.split(',') {
                            add(k.trim(), &mut out);
                        }
                        i += p + 1;
                        if !name.ends_with('s') {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
            _ => i += 1,
        }
    }
    out
}

/// What a text cites against a bibliography.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Report {
    /// Keys the text cites that name an entry (the entry's own key, whatever case was written).
    pub resolved: Vec<String>,
    /// Keys the text cites that no entry has (not even as an alias).
    pub missing: Vec<String>,
    /// Entries nothing cites, by key.
    pub uncited: Vec<String>,
    /// Keys written with different capitals than the entry has (they resolve, but break in BibTeX).
    pub wrong_case: Vec<(String, String)>,
}

/// Compares the keys `text` cites with `entries`.
pub fn check(text: &str, entries: &[Entry]) -> Report {
    let mut by_exact = std::collections::HashMap::new();
    let mut by_lower = std::collections::HashMap::new();
    for e in entries {
        for k in std::iter::once(&e.key).chain(e.aliases.iter()) {
            by_exact.entry(k.as_str()).or_insert(e.key.as_str());
            by_lower.entry(k.to_lowercase()).or_insert(e.key.as_str());
        }
    }
    let mut r = Report::default();
    let mut hit: BTreeSet<&str> = BTreeSet::new();
    for k in used_keys(text) {
        if let Some(&canon) = by_exact.get(k.as_str()) {
            hit.insert(canon);
            r.resolved.push(canon.to_string());
        } else if let Some(&canon) = by_lower.get(&k.to_lowercase()) {
            hit.insert(canon);
            r.resolved.push(canon.to_string());
            r.wrong_case.push((k, canon.to_string()));
        } else {
            r.missing.push(k);
        }
    }
    r.resolved.dedup();
    let mut seen = HashSet::new();
    r.resolved.retain(|k| seen.insert(k.clone()));
    r.uncited = entries
        .iter()
        .filter(|e| !hit.contains(e.key.as_str()))
        .map(|e| e.key.clone())
        .collect();
    r
}

impl Report {
    /// Short human text, one line per finding.
    pub fn render(&self) -> String {
        let mut s = format!(
            "{} cited, {} missing, {} uncited\n",
            self.resolved.len(),
            self.missing.len(),
            self.uncited.len()
        );
        for k in &self.missing {
            s.push_str(&format!("missing: {k}\n"));
        }
        for (w, k) in &self.wrong_case {
            s.push_str(&format!("case: {w} -> {k}\n"));
        }
        for k in &self.uncited {
            s.push_str(&format!("uncited: {k}\n"));
        }
        s
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "resolved": self.resolved,
            "missing": self.missing,
            "uncited": self.uncited,
            "wrong_case": self.wrong_case.iter().map(|(a, b)| serde_json::json!([a, b])).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(key: &str) -> Entry {
        Entry {
            key: key.into(),
            ..Default::default()
        }
    }

    #[test]
    fn pandoc_and_latex_forms() {
        let t = "As shown [@smith2019, p. 4; @doe_2020] and @lee21 argued. \\citep[see][p.~3]{a1,b2} \\parencite*{c3}";
        assert_eq!(
            used_keys(t),
            ["smith2019", "doe_2020", "lee21", "a1", "b2", "c3"]
        );
    }

    #[test]
    fn emails_and_bibtex_headers_are_not_citations() {
        assert!(used_keys("write to me@example.org").is_empty());
        assert!(used_keys("@article{x, title={T}}").is_empty());
        assert_eq!(used_keys("see @smith2019."), ["smith2019"]);
    }

    #[test]
    fn report_finds_missing_uncited_and_case() {
        let r = check("[@a; @B; @zz]", &[e("a"), e("b"), e("c")]);
        assert_eq!(r.resolved, ["a", "b"]);
        assert_eq!(r.missing, ["zz"]);
        assert_eq!(r.uncited, ["c"]);
        assert_eq!(r.wrong_case, [("B".to_string(), "b".to_string())]);
        assert!(r.render().contains("missing: zz"));
    }

    #[test]
    fn aliases_resolve_to_the_kept_key() {
        let mut k = e("main");
        k.aliases = vec!["old".into()];
        let r = check("@old", &[k]);
        assert_eq!(r.resolved, ["main"]);
        assert!(r.uncited.is_empty());
    }

    #[test]
    fn hostile_text_stays_linear_and_bounded() {
        let t = "\\cite{".repeat(200_000);
        let t0 = std::time::Instant::now();
        let _ = used_keys(&t);
        let many: String = (0..150_000).map(|i| format!("@k{i} ")).collect();
        assert!(used_keys(&many).len() <= MAX_KEYS);
        let open = "\\cite[".repeat(50_000);
        let _ = used_keys(&open);
        assert!(t0.elapsed().as_secs() < 5);
    }
}
