// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! Text handling for the audit: pulling citation groups out of a draft, splitting it into sentences, finding
//! quotations, and the normal form used to compare words. Every scan here is linear in the input; the original's
//! quadratic cases (an unmatched `«`, a long word full of full stops, thousands of unclosed `[`) are bounded.

use crate::unitext::fold_text;

/// First private-use code point used as the marker of the n-th citation group in a draft. The index travels in the
/// marker, so a sentence that is dropped (nothing but a marker) cannot shift the groups of the sentences after it.
const MARK_BASE: u32 = 0xF0000;
/// Number of distinct markers (groups beyond this are not extracted).
pub const MAX_MARKS: usize = 0xFFFD0;

pub fn mark_for(idx: usize) -> Option<char> {
    if idx < MAX_MARKS {
        char::from_u32(MARK_BASE + idx as u32)
    } else {
        None
    }
}

pub fn mark_index(c: char) -> Option<usize> {
    let u = c as u32;
    (MARK_BASE..MARK_BASE + MAX_MARKS as u32)
        .contains(&u)
        .then(|| (u - MARK_BASE) as usize)
}

fn is_mark(c: char) -> bool {
    mark_index(c).is_some()
}

/// Removes characters that would be mistaken for the markers (and `\u{1}`, the original's marker).
pub fn strip_marks(s: &str) -> String {
    s.chars().filter(|&c| c != '\u{1}' && !is_mark(c)).collect()
}

/// Longest bracket group (in characters) read as a citation group.
pub const MAX_GROUP_CHARS: usize = 600;

/// Replaces every `[...]` group that `parse` accepts (non-empty result) with a marker; returns the marked text, the
/// groups in marker order, and whether `max_groups` cut the extraction short (later groups stay as plain text).
pub fn extract_groups<T>(
    text: &str,
    parse: &dyn Fn(&str) -> Vec<T>,
    max_groups: usize,
) -> (String, Vec<Vec<T>>, bool) {
    let chars: Vec<char> = text.chars().collect();
    let (mut out, mut groups, mut i, mut capped) = (String::new(), Vec::new(), 0, false);
    while i < chars.len() {
        if chars[i] == '[' && !capped {
            // look for the closing bracket, stopping at another '[' or at the window end
            let end = chars.len().min(i + 1 + MAX_GROUP_CHARS);
            let mut j = i + 1;
            while j < end && chars[j] != ']' && chars[j] != '[' {
                j += 1;
            }
            if j < end && chars[j] == ']' {
                let inner: String = chars[i + 1..j].iter().collect();
                let g = parse(&inner);
                if !g.is_empty() {
                    match (groups.len() < max_groups)
                        .then(|| mark_for(groups.len()))
                        .flatten()
                    {
                        Some(m) => {
                            out.push(m);
                            groups.push(g);
                            i = j + 1;
                            continue;
                        }
                        None => capped = true,
                    }
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    (out, groups, capped)
}

fn is_abbrev(w: &str) -> bool {
    let w = w.trim_start_matches(['(', '"', '“']);
    let lw = w.to_lowercase();
    matches!(
        lw.as_str(),
        "al" | "cf"
            | "vs"
            | "etc"
            | "e.g"
            | "i.e"
            | "fig"
            | "eq"
            | "no"
            | "dr"
            | "prof"
            | "p"
            | "pp"
            | "ed"
            | "eds"
            | "cap"
            | "art"
            | "nr"
            | "pag"
    ) || (w.chars().count() == 1 && w.chars().all(|c| c.is_uppercase()))
}

/// Is the word that ends `cur` (just before the full stop `punct` that was pushed last) an abbreviation? Looks at the
/// last few characters only, so the cost does not depend on the length of the sentence so far.
fn abbrev_before(cur: &str, punct: char) -> bool {
    let mut it = cur.chars().rev().peekable();
    let mut skipped = 0;
    while it.peek() == Some(&punct) {
        it.next();
        skipped += 1;
        if skipped > 3 {
            return false; // "....." is not an abbreviation
        }
    }
    let mut rev: Vec<char> = Vec::with_capacity(13);
    for c in it {
        if c.is_whitespace() {
            break;
        }
        rev.push(c);
        if rev.len() > 12 {
            return false; // longer than any abbreviation we know
        }
    }
    let w: String = rev.into_iter().rev().collect();
    is_abbrev(&w)
}

/// Splits into sentences; a citation marker right after the full stop stays with the sentence it supports.
/// Pieces without a letter or digit are dropped (a marker alone is not a sentence; the groups are looked up by the
/// index inside the marker, so nothing shifts).
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split("\n\n") {
        let para = para.lines().map(|l| l.trim()).collect::<Vec<_>>().join(" ");
        let cs: Vec<char> = para.chars().collect();
        let (mut cur, mut i) = (String::new(), 0);
        while i < cs.len() {
            cur.push(cs[i]);
            if matches!(cs[i], '.' | '!' | '?' | '…') && !abbrev_before(&cur, cs[i]) {
                let mut j = i + 1;
                while j < cs.len() && (matches!(cs[j], '"' | '”' | '»' | ')') || is_mark(cs[j]))
                {
                    cur.push(cs[j]);
                    j += 1;
                }
                let mut k = j;
                while k < cs.len() && cs[k] == ' ' {
                    k += 1;
                }
                if k < cs.len() && is_mark(cs[k]) && k > j {
                    // "claim. [@a]": the marker belongs to the claim before it
                    cur.push(cs[k]);
                    j = k + 1;
                }
                if j >= cs.len() || cs.get(j).is_some_and(|c| c.is_whitespace()) {
                    out.push(cur.trim().to_string());
                    cur.clear();
                    i = j;
                    continue;
                }
                i = j;
                continue;
            }
            i += 1;
        }
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
    }
    out.into_iter()
        .filter(|s| s.chars().any(|c| c.is_alphanumeric()))
        .collect()
}

/// How far (in characters) a quotation may run before its closing mark.
const MAX_QUOTE_CHARS: usize = 2_000;

/// Quotations of at least four words, at most `max` per call.
pub fn quotes(s: &str, max: usize) -> Vec<String> {
    let cs: Vec<char> = s.chars().collect();
    // positions of every possible closing mark, so that finding the next one is a binary search and an unclosed
    // opening mark costs O(log n) instead of a scan to the end of the text
    let positions = |ch: char| -> Vec<usize> {
        cs.iter()
            .enumerate()
            .filter(|(_, c)| **c == ch)
            .map(|(i, _)| i)
            .collect()
    };
    let (straight, right_curly, left_curly, guillemet) = (
        positions('"'),
        positions('”'),
        positions('“'),
        positions('»'),
    );
    let next = |v: &[usize], from: usize| -> Option<usize> {
        v.get(v.partition_point(|&p| p < from)).copied()
    };
    let mut v = Vec::new();
    let mut i = 0;
    while i < cs.len() && v.len() < max {
        let closer = match cs[i] {
            '"' => next(&straight, i + 1),
            '“' => [next(&right_curly, i + 1), next(&straight, i + 1)]
                .into_iter()
                .flatten()
                .min(),
            '«' => next(&guillemet, i + 1),
            '„' => [next(&right_curly, i + 1), next(&left_curly, i + 1)]
                .into_iter()
                .flatten()
                .min(),
            _ => {
                i += 1;
                continue;
            }
        };
        match closer {
            Some(e) if e - i - 1 <= MAX_QUOTE_CHARS => {
                let q: String = cs[i + 1..e].iter().filter(|c| !is_mark(**c)).collect();
                if q.split_whitespace().count() >= 4 {
                    v.push(q);
                }
                i = e + 1;
            }
            _ => i += 1,
        }
    }
    v
}

/// The form in which words are compared: folded (accents, ligatures, case), everything that is not a letter or digit
/// is one space, no leading or trailing space.
pub fn norm(s: &str) -> String {
    let f = fold_text(s);
    let mut out = String::with_capacity(f.len());
    let mut space = true;
    for c in f.chars() {
        if c.is_alphanumeric() {
            out.push(c);
            space = false;
        } else if !space {
            out.push(' ');
            space = true;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Word-boundary containment of one normalised text in another (`hay` and `needle` come from `norm`).
pub fn contains_words(hay: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let mut from = 0;
    while let Some(p) = hay[from..].find(needle) {
        let s = from + p;
        let e = s + needle.len();
        let before = s == 0 || hay.as_bytes()[s - 1] == b' ';
        let after = e == hay.len() || hay.as_bytes()[e] == b' ';
        if before && after {
            return true;
        }
        // move one character forward (needle is non-empty, s is a char boundary)
        from = s + hay[s..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_carry_their_index() {
        assert_eq!(mark_index(mark_for(0).unwrap()), Some(0));
        assert_eq!(mark_index(mark_for(12345).unwrap()), Some(12345));
        assert_eq!(mark_index('a'), None);
        assert!(mark_for(MAX_MARKS).is_none());
        assert_eq!(strip_marks("a\u{1}b\u{f0000}c"), "abc");
    }

    #[test]
    fn groups_are_extracted_and_bounded() {
        let p = |s: &str| -> Vec<String> {
            s.strip_prefix('@')
                .map(|k| vec![k.to_string()])
                .unwrap_or_default()
        };
        let (t, g, capped) = extract_groups("a [@x] b [not] c [@y]", &p, 10);
        assert_eq!(g, vec![vec!["x".to_string()], vec!["y".to_string()]]);
        assert!(!capped);
        assert_eq!(t.chars().filter(|c| is_mark(*c)).count(), 2);
        let (_, g, capped) = extract_groups("[@a][@b][@c]", &p, 2);
        assert_eq!(g.len(), 2);
        assert!(capped);
        // unclosed brackets and a '[' inside a group
        let (_, g, _) = extract_groups("[[@a] [@b", &p, 10);
        assert_eq!(g.len(), 1);
        // a group longer than the window is not a group
        let long = format!("[@{}]", "k".repeat(MAX_GROUP_CHARS + 10));
        assert!(extract_groups(&long, &p, 10).1.is_empty());
    }

    #[test]
    fn sentences_keep_markers_and_abbreviations() {
        let m = mark_for(0).unwrap();
        let t = format!(
            "See Dr. Smith, e.g. the case. It held {m}. Next claim here {m}\n\nNew paragraph?"
        );
        let s = sentences(&t);
        assert_eq!(s.len(), 4, "{s:?}");
        assert!(s[0].starts_with("See Dr. Smith"));
        let t2 = format!("It held. {m} Next sentence.");
        let s2 = sentences(&t2);
        assert_eq!(s2.len(), 2);
        assert!(s2[0].ends_with(m));
    }

    #[test]
    fn a_lone_marker_does_not_shift_the_groups_after_it() {
        // the original numbered markers by order of appearance, so a sentence consisting only of a marker
        // (dropped for having no letters) gave every later sentence the wrong group
        let (t, g, _) = extract_groups(
            "[@a]\n\nA real claim about the thing [@b].",
            &|s: &str| {
                s.strip_prefix('@')
                    .map(|k| vec![k.to_string()])
                    .unwrap_or_default()
            },
            10,
        );
        let ss = sentences(&t);
        assert_eq!(ss.len(), 1);
        let idx: Vec<usize> = ss[0].chars().filter_map(mark_index).collect();
        assert_eq!(g[idx[0]], vec!["b".to_string()]);
    }

    #[test]
    fn quotes_need_four_words_and_a_closer() {
        assert_eq!(
            quotes("he said \"the quick brown fox\" and \"no\"", 10),
            vec!["the quick brown fox"]
        );
        assert_eq!(
            quotes("«one two three four» x", 10),
            vec!["one two three four"]
        );
        assert!(quotes("«never closed one two three four", 10).is_empty());
        assert_eq!(quotes("\"a b c d\" \"e f g h\" \"i j k l\"", 2).len(), 2);
    }

    #[test]
    fn norm_and_word_containment() {
        assert_eq!(norm("  Școala,  NAȚIONALĂ! "), "scoala nationala");
        assert_eq!(norm("\u{fb01}nal—word"), "final word");
        assert_eq!(norm("..."), "");
        assert!(contains_words("the rat sat", "rat sat"));
        assert!(!contains_words("the rather big", "the rat"));
        assert!(!contains_words("the rat", ""));
        assert!(contains_words("x é y", "é"));
    }

    #[test]
    fn pathological_inputs_stay_linear() {
        let t = std::time::Instant::now();
        let dots = "a.".repeat(300_000);
        let _ = sentences(&dots);
        let _ = sentences(&".".repeat(300_000));
        let _ = quotes(&"«".repeat(200_000), 100);
        let _ = quotes(&"\"".repeat(200_000), 100);
        let p = |_: &str| -> Vec<u8> { vec![] };
        let _ = extract_groups(&"[".repeat(300_000), &p, 10);
        let _ = extract_groups(&format!("[{}", "x".repeat(300_000)), &p, 10);
        assert!(t.elapsed().as_secs() < 10, "{:?}", t.elapsed());
    }
}
