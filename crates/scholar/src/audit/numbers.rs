// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! Figures in a sentence: digits and number words (English and Romanian), compared by value rather than by spelling.
//! A swapped figure keeps almost every other word, so plain term overlap cannot see it.
//!
//! Differences from the original: the arithmetic on number words saturates (the original overflowed on
//! `hundred hundred hundred ...`), and the work on one sentence is bounded by the caller's cap on its length.

use super::text::norm;
use std::collections::HashSet;

#[derive(Clone, Copy, PartialEq)]
enum Tok {
    Add(u32),
    Hundred,
    Scale(u32),
    And,
    Point,
    Half,
}

/// Number words (English, Romanian without diacritics), unambiguous only: "o" and "a" are not numbers here.
fn tok(w: &str) -> Option<Tok> {
    use Tok::*;
    Some(match w {
        "one" | "unu" | "una" => Add(1),
        "two" | "doi" | "doua" | "twice" => Add(2),
        "three" | "trei" => Add(3),
        "four" | "patru" => Add(4),
        "five" | "cinci" => Add(5),
        "six" | "sase" => Add(6),
        "seven" | "sapte" => Add(7),
        "eight" | "opt" => Add(8),
        "nine" | "noua" => Add(9),
        "ten" | "zece" => Add(10),
        "eleven" | "unsprezece" => Add(11),
        "twelve" | "doisprezece" => Add(12),
        "thirteen" | "treisprezece" => Add(13),
        "fourteen" | "paisprezece" => Add(14),
        "fifteen" | "cincisprezece" => Add(15),
        "sixteen" | "saisprezece" => Add(16),
        "seventeen" | "saptesprezece" => Add(17),
        "eighteen" | "optsprezece" => Add(18),
        "nineteen" | "nouasprezece" => Add(19),
        "twenty" | "douazeci" => Add(20),
        "thirty" | "treizeci" => Add(30),
        "forty" | "patruzeci" => Add(40),
        "fifty" | "cincizeci" => Add(50),
        "sixty" | "saizeci" => Add(60),
        "seventy" | "saptezeci" => Add(70),
        "eighty" | "optzeci" => Add(80),
        "ninety" | "nouazeci" => Add(90),
        "hundred" | "suta" | "sute" => Hundred,
        "thousand" | "mie" | "mii" => Scale(1000),
        "million" | "milion" | "milioane" => Scale(1_000_000),
        "and" | "si" => And,
        "point" | "virgula" => Point,
        "half" => Half,
        _ => return None,
    })
}

/// Numbers written in words, as (the words, the value as digits): "twenty five" -> 25, "one hundred and twelve" ->
/// 112, "three and a half" -> 3.5, "one point two" -> 1.2, "sapte" -> 7. `norm` has already turned hyphens into
/// spaces. Values that do not fit 64 bits saturate.
pub fn number_words(s: &str) -> Vec<(String, String)> {
    use Tok::*;
    let n = norm(s);
    let ws: Vec<&str> = n.split(' ').collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < ws.len() {
        let Some(t0) = tok(ws[i]) else {
            i += 1;
            continue;
        };
        if matches!(t0, And | Point) {
            i += 1;
            continue;
        }
        let (mut total, mut cur, mut j, mut frac) = (0u64, 0u64, i, None::<String>);
        let mut used = i;
        let mut seen = false;
        while j < ws.len() {
            match tok(ws[j]) {
                Some(Add(v)) => {
                    cur = cur.saturating_add(v as u64);
                    seen = true;
                    used = j;
                }
                Some(Hundred) => {
                    cur = cur.max(1).saturating_mul(100);
                    seen = true;
                    used = j;
                }
                Some(Scale(m)) => {
                    total = total.saturating_add(cur.max(1).saturating_mul(m as u64));
                    cur = 0;
                    seen = true;
                    used = j;
                }
                Some(Half) if !seen => {
                    cur = 0;
                    frac = Some("0.5".into());
                    used = j;
                    break;
                }
                Some(And) if seen => {
                    // "and" joins numerals ("a hundred and twelve") or introduces "a half" ("three and a half");
                    // otherwise the number ended
                    match (ws.get(j + 1).copied().and_then(tok), ws.get(j + 2).copied()) {
                        (Some(Add(_) | Hundred | Scale(_)), _) => {}
                        (_, Some("half")) if ws[j + 1] == "a" => {
                            frac = Some(".5".into());
                            used = j + 2;
                            break;
                        }
                        _ => break,
                    }
                }
                Some(Point) if seen => {
                    let mut d = String::new();
                    let mut k = j + 1;
                    while let Some(Some(Add(v @ 0..=9))) = ws.get(k).map(|w| tok(w)) {
                        d.push_str(&v.to_string());
                        k += 1;
                    }
                    if !d.is_empty() {
                        frac = Some(format!(".{d}"));
                        used = k - 1;
                    }
                    break;
                }
                _ => break,
            }
            j += 1;
        }
        let whole = total.saturating_add(cur);
        let val = match frac {
            Some(f) if f == "0.5" => f,
            Some(f) => format!("{whole}{f}"),
            None => whole.to_string(),
        };
        out.push((ws[i..=used].join(" "), val));
        i = used + 1;
    }
    out
}

/// Digit figures in `s`, normalised: thousands separators removed ("1,240" = "1240" = "1 240"), decimal comma read
/// as a point.
pub fn figures(s: &str) -> Vec<String> {
    let cs: Vec<char> = s.chars().collect();
    let digit = |k: usize| k < cs.len() && cs[k].is_ascii_digit();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        if !cs[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut groups: Vec<String> = vec![String::new()];
        let mut seps: Vec<char> = Vec::new();
        while j < cs.len() {
            if digit(j) {
                if let Some(g) = groups.last_mut() {
                    g.push(cs[j]);
                }
                j += 1;
            } else if matches!(cs[j], '.' | ',') && digit(j + 1) {
                seps.push(cs[j]);
                groups.push(String::new());
                j += 1;
            } else if matches!(cs[j], ' ' | '\u{a0}' | '\u{202f}')
                && digit(j + 1)
                && digit(j + 2)
                && digit(j + 3)
                && !digit(j + 4)
                && groups[0].len() <= 3
                && groups.iter().skip(1).all(|g| g.len() == 3)
            {
                seps.push(' ');
                groups.push(String::new());
                j += 1;
            } else {
                break;
            }
        }
        let triples = groups.len() > 1
            && groups[0].len() <= 3
            && groups[1..].iter().all(|g| g.len() == 3)
            && seps.iter().all(|c| *c == seps[0]);
        let thousands = triples && (seps[0] == ' ' || seps[0] == ',' || groups.len() > 2);
        let n = if thousands || groups.len() == 1 {
            groups.concat()
        } else {
            let last = groups.len() - 1;
            format!("{}.{}", groups[..last].concat(), groups[last])
        };
        out.push(n);
        i = j.max(i + 1);
    }
    out
}

/// Numbers (digits or number words) that the sentence states but the cited passage does not contain.
///
/// Figures are compared by value, not by spelling: "4", "four" and "patru" are the same number, "twenty-five" is 25,
/// "three and a half" is 3.5, "71.3" is one number (not "71" and "3"), so a correct citation is not flagged for
/// writing a digit where the source has a word, and a re-ordered figure cannot hide.
pub fn missing_numbers(sentence: &str, passage: &str) -> Vec<String> {
    let have: HashSet<String> = {
        let mut v: HashSet<String> = figures(passage).into_iter().collect();
        v.extend(number_words(passage).into_iter().map(|(_, val)| val));
        v
    };
    let mut miss: Vec<String> = figures(sentence)
        .into_iter()
        .filter(|n| !have.contains(n))
        .collect();
    miss.extend(
        number_words(sentence)
            .into_iter()
            .filter(|(_, v)| !have.contains(v))
            .map(|(t, _)| t),
    );
    miss.dedup();
    miss
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Vec<String> {
        number_words(s).into_iter().map(|(_, v)| v).collect()
    }

    #[test]
    fn digits_by_value() {
        assert_eq!(
            figures("1,240 and 1 240 and 1240"),
            vec!["1240", "1240", "1240"]
        );
        assert_eq!(figures("71.3 then 71,3"), vec!["71.3", "71.3"]);
        assert_eq!(figures("1.240.000 items"), vec!["1240000"]);
        assert_eq!(figures("in 2019, 12 people"), vec!["2019", "12"]);
        assert_eq!(figures("no figures"), Vec::<String>::new());
    }

    #[test]
    fn words_by_value() {
        assert_eq!(w("twenty-five"), vec!["25"]);
        assert_eq!(w("one hundred and twelve"), vec!["112"]);
        assert_eq!(w("three and a half"), vec![".5".replace('.', "3.")]);
        assert_eq!(w("one point two"), vec!["1.2"]);
        assert_eq!(w("sapte"), vec!["7"]);
        assert_eq!(w("two thousand five hundred"), vec!["2500"]);
        assert_eq!(w("the quick brown fox"), Vec::<String>::new());
    }

    #[test]
    fn a_swapped_figure_is_found_and_a_respelled_one_is_not() {
        assert!(
            missing_numbers(
                "The fine was 500 euros.",
                "The fine was five hundred euros."
            )
            .is_empty()
        );
        assert_eq!(
            missing_numbers("The fine was 700 euros.", "The fine was 500 euros."),
            vec!["700"]
        );
        assert_eq!(
            missing_numbers("A sentence with four items", "had 5 items"),
            vec!["four"]
        );
    }

    #[test]
    fn huge_number_words_saturate_instead_of_overflowing() {
        let s = "one ".to_string() + &"hundred ".repeat(40);
        let _ = number_words(&s);
        let m = "million ".repeat(50);
        let _ = number_words(&m);
        let big = "nine ".repeat(10_000);
        assert_eq!(number_words(&big).len(), 1);
    }

    #[test]
    fn long_runs_of_digits_and_separators_are_fine() {
        let t = std::time::Instant::now();
        let _ = figures(&"1,".repeat(200_000));
        let _ = figures(&"9".repeat(200_000));
        let _ = figures(&"1 234 ".repeat(100_000));
        assert!(t.elapsed().as_secs() < 5);
    }
}
