//! Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//!
//! Prompts and parsers for model-assisted checks on top of retrieval: re-ranking passages, and testing whether
//! each cited passage really supports the sentence that cites it.
//!
//! This module contains no model and makes no network call: the caller sends `SYSTEM_RERANK` / `SYSTEM_VERIFY`
//! to whatever model it uses and hands the reply to the parsers. A model can be wrong here too, and a reply that
//! cannot be read becomes `Unclear` / `None`, never a guess. These are aids, not proof.

use std::collections::BTreeSet;

pub const SYSTEM_RERANK: &str = "You rate how useful each passage is for answering the query. Output exactly one line per passage \
in the form `n: score`, where score is an integer from 0 (irrelevant) to 10 (directly answers). No other text.";

pub const SYSTEM_VERIFY: &str = "You are a strict citation checker. You get excerpts and ONE sentence that cites them. \
Decide whether the excerpts, and nothing else, support everything the sentence claims. \
Answer with exactly one word: SUPPORTED, PARTIAL or UNSUPPORTED.";

/// Longest `[n]` group (in characters) that is still read as a citation marker.
const MAX_GROUP: usize = 64;
/// Passages beyond this many are ignored by the parsers (allocation bound against a hostile `n`).
const MAX_PASSAGES: usize = 100_000;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Verdict {
    Supported,
    Partial,
    Unsupported,
    Unclear,
}

/// Scores per passage (index = passage number - 1) from lines like `3: 8`, `[2] 5`, `1) 10`. Out-of-range passage
/// numbers and scores above 10 are ignored; for a repeated passage number the last line wins.
pub fn parse_scores(text: &str, n: usize) -> Vec<Option<u8>> {
    let n = n.min(MAX_PASSAGES);
    let mut out = vec![None; n];
    for line in text.lines() {
        let mut digits = line
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty());
        if let (Some(a), Some(b)) = (digits.next(), digits.next())
            && let (Ok(i), Ok(s)) = (a.parse::<usize>(), b.parse::<u8>())
            && i >= 1
            && i <= n
            && s <= 10
        {
            out[i - 1] = Some(s);
        }
    }
    out
}

/// Order for the passages: best score first, ties and unscored ones (neutral 5) keep their retrieval order.
/// `None` when the model gave no usable scores at all (the caller keeps the original order).
pub fn rerank_order(text: &str, n: usize) -> Option<Vec<usize>> {
    let sc = parse_scores(text, n);
    if sc.iter().all(|s| s.is_none()) {
        return None;
    }
    let mut idx: Vec<usize> = (0..sc.len()).collect();
    idx.sort_by_key(|&i| std::cmp::Reverse(sc[i].map(i32::from).unwrap_or(5)));
    Some(idx)
}

/// Reads the model's one-word verdict. Only the start of a short reply is looked at: a long rambling answer that
/// merely contains the word "supported" is `Unclear`, not `Supported`.
pub fn parse_verdict(text: &str) -> Verdict {
    let head: String = text
        .trim()
        .chars()
        .take(120)
        .collect::<String>()
        .to_uppercase();
    let first: String = head
        .chars()
        .skip_while(|c| !c.is_alphabetic())
        .take_while(|c| c.is_alphabetic())
        .collect();
    let short = text.trim().chars().count() <= 120;
    match first.as_str() {
        "SUPPORTED" => Verdict::Supported,
        "PARTIAL" | "PARTIALLY" => Verdict::Partial,
        "UNSUPPORTED" | "NOT" => Verdict::Unsupported,
        _ if short && head.contains("UNSUPPORTED") => Verdict::Unsupported,
        _ if short && head.contains("PARTIAL") => Verdict::Partial,
        _ if short && head.contains("SUPPORTED") => Verdict::Supported,
        _ => Verdict::Unclear,
    }
}

/// If `b[i]` opens a `[n]`, `[n, m]` or `[n; m]` group, returns its numbers and the index of `]`. A number too
/// large for `usize` is returned as `usize::MAX`, so it is reported as out of range instead of vanishing.
fn bracket_group(b: &[char], i: usize) -> Option<(Vec<usize>, usize)> {
    if b.get(i) != Some(&'[') {
        return None;
    }
    let mut j = i + 1;
    while j < b.len() && j - i < MAX_GROUP && b[j] != ']' && b[j] != '[' {
        j += 1;
    }
    if j >= b.len() || b[j] != ']' {
        return None;
    }
    let inner: String = b[i + 1..j].iter().collect();
    let mut nums = Vec::new();
    for p in inner.split([',', ';']).map(str::trim) {
        if p.is_empty() || !p.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        nums.push(p.parse::<usize>().unwrap_or(usize::MAX));
    }
    Some((nums, j))
}

/// Numbers appearing inside `[..]` brackets, e.g. "[2, 5]" -> {2, 5}. Numbers that do not fit `usize` appear as
/// `usize::MAX`.
pub fn cited_numbers(text: &str) -> BTreeSet<usize> {
    let b: Vec<char> = text.chars().collect();
    let mut out = BTreeSet::new();
    let mut i = 0;
    while i < b.len() {
        if let Some((nums, end)) = bracket_group(&b, i) {
            out.extend(nums);
            i = end;
        }
        i += 1;
    }
    out
}

/// Sentences that carry citations, with their valid excerpt numbers (1..=n_hits), in order, at most `max`.
/// Sentences whose only numbers are out of range are left out here (the audit reports them separately).
pub fn cite_sentences(answer: &str, n_hits: usize, max: usize) -> Vec<(String, Vec<usize>)> {
    let mut out = Vec::new();
    'paras: for para in answer.split('\n') {
        let mut cur = String::new();
        let chars: Vec<char> = para.chars().collect();
        let mut i = 0;
        let mut depth = 0i32; // do not split inside [1, 2]
        while i < chars.len() {
            if out.len() >= max {
                break 'paras;
            }
            let c = chars[i];
            cur.push(c);
            if c == '[' {
                depth += 1;
            } else if c == ']' {
                depth -= 1;
            }
            let end = matches!(c, '.' | '?' | '!')
                && depth <= 0
                && chars
                    .get(i + 1)
                    .is_none_or(|n| n.is_whitespace() || *n == '[');
            // a citation right after the full stop belongs to this sentence: "claim. [1]"
            if end {
                let mut j = i + 1;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                if chars.get(j) == Some(&'[')
                    && chars.get(j + 1).is_some_and(|d| d.is_ascii_digit())
                {
                    i += 1;
                    continue;
                }
                push_sentence(&mut out, &cur, n_hits);
                cur.clear();
            }
            i += 1;
        }
        push_sentence(&mut out, &cur, n_hits);
    }
    out.truncate(max);
    out
}

fn push_sentence(out: &mut Vec<(String, Vec<usize>)>, s: &str, n_hits: usize) {
    let s = s.trim();
    if s.is_empty() {
        return;
    }
    let nums: Vec<usize> = cited_numbers(s)
        .into_iter()
        .filter(|&n| n >= 1 && n <= n_hits)
        .collect();
    if !nums.is_empty() {
        out.push((s.to_string(), nums));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scores_in_various_shapes() {
        let s = parse_scores("1: 9\n[2] 3\n3) 10\nfoo\n9: 4\n2: 99", 3);
        assert_eq!(s, vec![Some(9), Some(3), Some(10)]); // 99 is out of range, passage 9 does not exist
    }

    #[test]
    fn reranks_stably_and_gives_up_on_garbage() {
        assert_eq!(rerank_order("1: 2\n2: 9\n3: 2", 3), Some(vec![1, 0, 2]));
        assert_eq!(rerank_order("1: 7\n3: 7", 3), Some(vec![0, 2, 1])); // passage 2 unscored = neutral 5
        assert_eq!(rerank_order("no idea", 3), None);
        assert_eq!(rerank_order("1: 5", 0), None);
    }

    #[test]
    fn parses_verdicts() {
        assert_eq!(parse_verdict("SUPPORTED"), Verdict::Supported);
        assert_eq!(parse_verdict("  Partial."), Verdict::Partial);
        assert_eq!(parse_verdict("Partially supported"), Verdict::Partial);
        assert_eq!(parse_verdict("Not supported"), Verdict::Unsupported);
        assert_eq!(
            parse_verdict("The sentence is UNSUPPORTED by the excerpt"),
            Verdict::Unsupported
        );
        assert_eq!(parse_verdict("¯\\_(ツ)_/¯"), Verdict::Unclear);
        assert_eq!(parse_verdict(""), Verdict::Unclear);
        // a long reply that only mentions the word somewhere is not a verdict
        let long = format!("{} supported", "blah ".repeat(60));
        assert_eq!(parse_verdict(&long), Verdict::Unclear);
    }

    #[test]
    fn finds_cited_sentences() {
        let a = "The authority bears the burden [1]. Costs may be imposed [2, 3]. No citation here. Ghost claim [9].\nSecond line. With trailing cite. [2]";
        let s = cite_sentences(a, 3, 10);
        assert_eq!(s.len(), 3);
        assert_eq!(
            s[0],
            ("The authority bears the burden [1].".to_string(), vec![1])
        );
        assert_eq!(s[1].1, vec![2, 3]);
        assert_eq!(s[2].0, "With trailing cite. [2]");
        assert_eq!(cite_sentences(a, 3, 1).len(), 1);
    }

    #[test]
    fn huge_numbers_are_kept_as_out_of_range() {
        let n = cited_numbers("claim [99999999999999999999999999] and [2]");
        assert!(n.contains(&usize::MAX));
        assert!(n.contains(&2));
        // and are never mistaken for a valid excerpt
        assert!(cite_sentences("claim [99999999999999999999999999].", 3, 5).is_empty());
    }

    #[test]
    fn hostile_input_stays_bounded() {
        let bomb = "[".repeat(200_000);
        let t = std::time::Instant::now();
        assert!(cited_numbers(&bomb).is_empty());
        assert!(cite_sentences(&bomb, 3, 10).is_empty());
        let many = "x [1]. ".repeat(100_000);
        assert_eq!(cite_sentences(&many, 3, 50).len(), 50);
        assert_eq!(parse_scores("1: 1", usize::MAX).len(), MAX_PASSAGES);
        assert!(t.elapsed().as_secs() < 10);
    }
}
