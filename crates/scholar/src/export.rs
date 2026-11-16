//! Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf).
//! Reference lists and in-text citations in three small styles, as plain text or Markdown:
//! author-year (Harvard-like), numeric, and APA-like. Not a CSL engine: for journal-exact
//! formatting export CSL JSON (`csljson::to_json`) and let a CSL processor do it.

use crate::bib::Entry;
use crate::names::Person;
use crate::refs;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// Smith, J. (2019). Title. Journal, 3(2), 1-9.
    AuthorYear,
    /// [1] J. Smith, Title, Journal 3(2), 2019, 1-9.
    Numeric,
    /// Smith, J., & Doe, A. (2019). Title. *Journal*, *3*(2), 1-9. https://doi.org/..
    Apa,
}

impl Style {
    pub fn parse(s: &str) -> Option<Style> {
        match s.trim().to_ascii_lowercase().as_str() {
            "author-year" | "authoryear" | "harvard" => Some(Style::AuthorYear),
            "numeric" | "ieee" | "vancouver" => Some(Style::Numeric),
            "apa" => Some(Style::Apa),
            _ => None,
        }
    }
}

fn people(e: &Entry) -> (Vec<Person>, bool) {
    let l = e.people();
    (l.people, l.et_al)
}

fn dot(s: &str) -> String {
    let t = s.trim();
    if t.is_empty() || t.ends_with(['.', '?', '!']) {
        t.to_string()
    } else {
        format!("{t}.")
    }
}

fn initials(p: &Person) -> String {
    p.given()
        .split([' ', '-'])
        .filter_map(|w| w.chars().next())
        .filter(|c| c.is_alphabetic())
        .map(|c| format!("{c}."))
        .collect::<Vec<_>>()
        .join(" ")
}

fn family_with_particle(p: &Person) -> String {
    let (pa, f) = (p.particle(), p.family());
    if pa.is_empty() {
        f
    } else {
        format!("{pa} {f}")
    }
}

/// "Smith, J." (corporate names stay whole).
fn last_first(p: &Person) -> String {
    if p.is_literal() {
        return p.family();
    }
    let i = initials(p);
    if i.is_empty() {
        family_with_particle(p)
    } else {
        format!("{}, {i}", family_with_particle(p))
    }
}

/// "J. Smith".
fn first_last(p: &Person) -> String {
    if p.is_literal() {
        return p.family();
    }
    let i = initials(p);
    if i.is_empty() {
        family_with_particle(p)
    } else {
        format!("{i} {}", family_with_particle(p))
    }
}

fn join_names(v: &[String], et_al: bool, amp: &str, comma: bool) -> String {
    let mut v = v.to_vec();
    if et_al {
        v.push("et al.".into());
        return v.join(", ");
    }
    match v.len() {
        0 | 1 => v.join(""),
        2 if !comma => format!("{} {amp} {}", v[0], v[1]),
        _ => {
            let last = v.pop().unwrap_or_default();
            let c = if comma { "," } else { "" };
            format!("{}{c} {amp} {last}", v.join(", "))
        }
    }
}

fn volume_issue(e: &Entry) -> String {
    match (e.volume.is_empty(), e.number.is_empty()) {
        (true, true) => String::new(),
        (false, true) => e.volume.clone(),
        (true, false) => format!("({})", e.number),
        (false, false) => format!("{}({})", e.volume, e.number),
    }
}

fn pages(e: &Entry) -> String {
    e.pages
        .replace("--", "-")
        .replace('\u{2013}', "-")
        .trim()
        .to_string()
}

fn link(e: &Entry) -> String {
    if !e.doi.is_empty() {
        format!("https://doi.org/{}", e.doi)
    } else {
        e.url.clone()
    }
}

/// One reference line in `style`; `n` is the number for the numeric style.
pub fn reference(e: &Entry, style: Style, n: usize) -> String {
    let (p, et) = people(e);
    let year = if e.year.is_empty() {
        "n.d.".to_string()
    } else {
        e.year.clone()
    };
    let mut parts: Vec<String> = Vec::new();
    match style {
        Style::AuthorYear | Style::Apa => {
            let who: Vec<String> = p.iter().map(last_first).collect();
            let who = join_names(&who, et, "&", style == Style::Apa);
            if !who.is_empty() {
                parts.push(dot(&who));
            }
            parts.push(format!("({year})."));
            if !e.title.is_empty() {
                parts.push(dot(&e.title));
            }
            if !e.container.is_empty() {
                let mut c = if style == Style::Apa {
                    format!("*{}*", e.container)
                } else {
                    e.container.clone()
                };
                let vi = volume_issue(e);
                if !vi.is_empty() {
                    c.push_str(&format!(
                        ", {}",
                        if style == Style::Apa {
                            format!("*{}*", vi.split('(').next().unwrap_or(""))
                                + &vi[vi.find('(').unwrap_or(vi.len())..]
                        } else {
                            vi
                        }
                    ));
                }
                let pg = pages(e);
                if !pg.is_empty() {
                    c.push_str(&format!(", {pg}"));
                }
                parts.push(dot(&c));
            } else if !e.publisher.is_empty() {
                parts.push(dot(&e.publisher));
            }
        }
        Style::Numeric => {
            parts.push(format!("[{n}]"));
            let who: Vec<String> = p.iter().map(first_last).collect();
            let who = join_names(&who, et, "and", false);
            if !who.is_empty() {
                parts.push(format!("{who},"));
            }
            if !e.title.is_empty() {
                parts.push(format!("{},", e.title.trim_end_matches('.')));
            }
            let mut c = e.container.clone();
            let vi = volume_issue(e);
            if !vi.is_empty() {
                c.push_str(&format!(" {vi}"));
            }
            if !c.trim().is_empty() {
                parts.push(format!("{},", c.trim()));
            } else if !e.publisher.is_empty() {
                parts.push(format!("{},", e.publisher));
            }
            let mut tail = year.clone();
            let pg = pages(e);
            if !pg.is_empty() {
                tail.push_str(&format!(", {pg}"));
            }
            parts.push(dot(&tail));
        }
    }
    let l = link(e);
    if !l.is_empty() {
        parts.push(l);
    }
    parts.join(" ")
}

/// Reference list for `entries`, sorted for the author-year styles, in given order for numeric.
/// Markdown: one line per reference, numeric as an ordered list.
pub fn bibliography(entries: &[Entry], style: Style) -> String {
    let mut v: Vec<&Entry> = entries.iter().collect();
    if style != Style::Numeric {
        v.sort_by_cached_key(|e| {
            let (p, _) = people(e);
            let who = p
                .first()
                .map(|x| x.family().to_lowercase())
                .unwrap_or_else(|| e.title.to_lowercase());
            (who, e.year.clone(), e.key.clone())
        });
    }
    let mut s = String::new();
    for (i, e) in v.iter().enumerate() {
        if style == Style::Numeric {
            s.push_str(&reference(e, style, i + 1));
        } else {
            s.push_str(&reference(e, style, 0));
        }
        s.push_str("\n\n");
    }
    s.trim_end().to_string()
}

/// Rewrites pandoc citations in `text` for `style`: `[@a, p. 3; @b]` becomes "(Smith 2019, p. 3;
/// Doe 2020)" for the author-year styles and "[1, p. 3; 2]" for numeric. Numbers follow first
/// use. Returns the new text and the entries in the order they were cited (the list to print).
/// A key with no entry is left as written, so nothing is lost silently.
pub fn render_citations<'a>(
    text: &str,
    entries: &'a [Entry],
    style: Style,
) -> (String, Vec<&'a Entry>) {
    let mut by_key: HashMap<String, &Entry> = HashMap::new();
    for e in entries {
        for k in std::iter::once(&e.key).chain(e.aliases.iter()) {
            by_key.entry(k.to_lowercase()).or_insert(e);
        }
    }
    let mut cited: Vec<&Entry> = Vec::new();
    let mut number: HashMap<String, usize> = HashMap::new();
    let c: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '[' && c.get(i + 1) == Some(&'@') {
            let end = c[i..c.len().min(i + 2048)].iter().position(|&x| x == ']');
            if let Some(p) = end {
                let inner: String = c[i + 1..i + p].iter().collect();
                let mut cites = Vec::new();
                let mut ok = true;
                for part in inner.split(';') {
                    let part = part.trim().trim_start_matches('@');
                    let kend = part
                        .find(|ch: char| {
                            !(ch.is_alphanumeric()
                                || matches!(ch, '_' | ':' | '.' | '-' | '/' | '+'))
                        })
                        .unwrap_or(part.len());
                    let key = part[..kend].trim_end_matches(['.', ':', '-', '/', '+']);
                    let loc = part[kend..].trim().trim_start_matches(',').trim();
                    match by_key.get(&key.to_lowercase()) {
                        Some(e) => cites.push((*e, loc.to_string())),
                        None => ok = false,
                    }
                }
                if ok && !cites.is_empty() {
                    let mut pieces = Vec::new();
                    for (e, loc) in cites {
                        let n = *number.entry(e.key.clone()).or_insert_with(|| {
                            cited.push(e);
                            cited.len()
                        });
                        let base = if style == Style::Numeric {
                            n.to_string()
                        } else {
                            short(e)
                        };
                        pieces.push(if loc.is_empty() {
                            base
                        } else {
                            format!("{base}, {loc}")
                        });
                    }
                    let (o, cl) = if style == Style::Numeric {
                        ('[', ']')
                    } else {
                        ('(', ')')
                    };
                    out.push(o);
                    out.push_str(&pieces.join("; "));
                    out.push(cl);
                    i += p + 1;
                    continue;
                }
            }
        }
        out.push(c[i]);
        i += 1;
    }
    let _ = refs::MAX_KEYS;
    (out, cited)
}

/// "Smith 2019", "Smith & Doe 2019", "Smith et al. 2019".
fn short(e: &Entry) -> String {
    let (p, et) = people(e);
    let who = match p.len() {
        0 => e.title.clone(),
        1 if !et => p[0].family(),
        2 if !et => format!("{} & {}", p[0].family(), p[1].family()),
        _ => format!("{} et al.", p[0].family()),
    };
    if e.year.is_empty() {
        who
    } else {
        format!("{who} {}", e.year)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bib;

    fn sample() -> Vec<Entry> {
        bib::parse(
            "@article{smith2019, author={Smith, John and Doe, Anna}, title={On things}, journal={Journal of Stuff}, year={2019}, volume={3}, number={2}, pages={1--9}, doi={10.1000/xyz}}\n\
             @book{lee2020, author={Lee, Kim}, title={A Book}, publisher={Press}, year={2020}}\n",
        )
    }

    #[test]
    fn author_year_list() {
        let s = bibliography(&sample(), Style::AuthorYear);
        assert!(s.starts_with("Lee, K. (2020). A Book. Press."), "{s}");
        assert!(s.contains("Smith, J. & Doe, A. (2019). On things. Journal of Stuff, 3(2), 1-9. https://doi.org/10.1000/xyz"), "{s}");
    }

    #[test]
    fn numeric_and_apa() {
        let e = sample();
        let n = reference(&e[0], Style::Numeric, 7);
        assert!(
            n.starts_with("[7] J. Smith and A. Doe, On things, Journal of Stuff 3(2), 2019, 1-9."),
            "{n}"
        );
        let a = reference(&e[0], Style::Apa, 0);
        assert!(a.contains("Smith, J., & Doe, A. (2019)"), "{a}");
        assert!(a.contains("*Journal of Stuff*, *3*(2), 1-9"), "{a}");
    }

    #[test]
    fn citations_are_rewritten_in_first_use_order() {
        let e = sample();
        let (t, used) = render_citations(
            "A [@lee2020, p. 4; @smith2019] b [@nope] c [@lee2020].",
            &e,
            Style::Numeric,
        );
        assert_eq!(t, "A [1, p. 4; 2] b [@nope] c [1].");
        assert_eq!(used.len(), 2);
        let (t, _) = render_citations("x [@smith2019, p. 2]", &e, Style::AuthorYear);
        assert_eq!(t, "x (Smith & Doe 2019, p. 2)");
    }

    #[test]
    fn empty_and_hostile_inputs() {
        assert_eq!(bibliography(&[], Style::Apa), "");
        let bibl = sample();
        let (t, u) = render_citations(&"[@".repeat(100_000), &bibl, Style::Numeric);
        assert!(u.is_empty() && t.len() == 200_000);
        let e = Entry::default();
        let _ = reference(&e, Style::Apa, 1);
        assert_eq!(Style::parse("IEEE"), Some(Style::Numeric));
        assert_eq!(Style::parse("x"), None);
    }
}
