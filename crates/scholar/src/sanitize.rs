//! Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//!
//! Neutralising text that comes from outside (a `.bib` someone sent you, an archive record, a draft, a
//! model answer, a model name in a receipt).
//!
//! Three things go wrong when such text is copied into files, prompts or onto a terminal as it is:
//! * control characters: NUL truncates or corrupts files for other tools, ESC starts a terminal escape sequence;
//! * bidirectional overrides (U+202E…), invisible characters and Unicode "tag" characters (U+E0000…, invisible text
//!   that language models read but people do not) change what a reader sees, so a key or title can look like
//!   something else in a terminal, in a diff or in a prompt;
//! * structure characters (a newline in a RIS value, an unbalanced `}` in a BibTeX value, a `\` or `}` in a LaTeX
//!   citation key) that let a value break out of its field and become new fields or commands.
//!
//! Nothing here changes ordinary text: letters, digits, punctuation, accents and every script, including ZWJ and
//! ZWNJ, which some scripts need, and the directional marks LRM/RLM/ALM that mixed-direction text uses, pass
//! through (`key` is stricter: a key is an identifier, so every invisible character goes).

/// True for characters that must never be copied from outside text into a file or a terminal line.
fn is_hostile(c: char) -> bool {
    matches!(c,
        '\u{0}'..='\u{8}' | '\u{b}' | '\u{c}' | '\u{e}'..='\u{1f}' | '\u{7f}'..='\u{9f}'
        | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}' // bidi embeddings, overrides, isolates
        | '\u{200b}' | '\u{2060}' | '\u{feff}' | '\u{180e}' // zero-width space, word joiner, BOM
        | '\u{2061}'..='\u{2064}' // invisible math operators
        | '\u{fff9}'..='\u{fffb}' // interlinear annotation
        | '\u{e0000}'..='\u{e007f}') // Unicode "tag" characters: invisible ASCII smuggling
}

/// Invisible formatting characters that are fine in running text but never in an identifier.
fn is_invisible_format(c: char) -> bool {
    matches!(c,
        '\u{ad}' | '\u{61c}' | '\u{200c}'..='\u{200f}' | '\u{115f}' | '\u{1160}'
        | '\u{3164}' | '\u{ffa0}' | '\u{fe00}'..='\u{fe0f}' | '\u{e0100}'..='\u{e01ef}')
}

/// Single-line text: hostile characters are dropped, line breaks and tabs become one space.
pub fn line(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if matches!(c, '\n' | '\r' | '\t' | '\u{2028}' | '\u{2029}') {
            if !space && !o.is_empty() {
                o.push(' ');
            }
            space = true;
            continue;
        }
        if is_hostile(c) {
            continue;
        }
        space = false;
        o.push(c);
    }
    o.trim().to_string()
}

/// Multi-line text (an answer, a draft): hostile characters are dropped, `\n` and `\t` are kept, `\r\n` and lone
/// `\r` become `\n`.
pub fn text(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '\r' => {
                if it.peek() == Some(&'\n') {
                    it.next();
                }
                o.push('\n');
            }
            '\u{2028}' | '\u{2029}' => o.push('\n'),
            '\n' | '\t' => o.push(c),
            c if is_hostile(c) => {}
            c => o.push(c),
        }
    }
    o
}

/// A value for a BibTeX `{…}` field: single line, and braces balanced so it cannot close its own field.
pub fn bib_value(s: &str) -> String {
    let t = line(s);
    let mut o = String::with_capacity(t.len() + 2);
    let mut depth = 0usize;
    let mut escaped = false;
    for c in t.chars() {
        match c {
            '{' if !escaped => {
                depth += 1;
                o.push(c);
            }
            '}' if !escaped => {
                if depth > 0 {
                    depth -= 1;
                    o.push(c);
                } // a `}` with nothing to close is dropped
            }
            c => o.push(c),
        }
        escaped = c == '\\' && !escaped;
    }
    // a trailing backslash would escape the closing brace of the field
    while o.ends_with('\\') {
        o.pop();
    }
    for _ in 0..depth {
        o.push('}');
    }
    o
}

/// A citation key: what BibTeX, Biber and LaTeX accept inside `\cite{…}` without surprises. Whitespace, controls,
/// bidi and invisible characters and the characters `{}(),="#%\'~^$&` are dropped.
pub fn key(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !c.is_whitespace()
                && !is_hostile(*c)
                && !is_invisible_format(*c)
                && !matches!(
                    c,
                    '{' | '}'
                        | '('
                        | ')'
                        | ','
                        | '='
                        | '"'
                        | '#'
                        | '%'
                        | '\\'
                        | '\''
                        | '~'
                        | '^'
                        | '$'
                        | '&'
                )
        })
        .collect()
}

/// Text that goes on a terminal line (warnings, diagnostics): hostile characters dropped, line breaks kept as spaces.
pub fn term(s: &str) -> String {
    line(s)
}

/// `line`, cut to at most `max` characters (never in the middle of a character), with an ellipsis when cut.
pub fn short(s: &str, max: usize) -> String {
    let l = line(s);
    if l.chars().count() <= max {
        return l;
    }
    let mut o: String = l.chars().take(max.saturating_sub(1)).collect();
    o.push('…');
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_text_is_untouched() {
        for s in [
            "Ada Lovelace",
            "Școala Națională — Ștefan cel Mare",
            "Zoë Müller-Çelik, 東京大学",
            "می‌خواهم", // contains ZWNJ (U+200C), needed in Persian
            "a b",
            "אבג\u{200f}abc", // RLM between scripts
        ] {
            assert_eq!(line(s), s);
        }
    }

    #[test]
    fn controls_bidi_and_invisibles_go() {
        assert_eq!(
            line("a\u{0}b\u{1b}[31mc\u{202e}d\u{200b}e\u{feff}f"),
            "ab[31mcdef"
        );
        assert_eq!(line("x\u{85}y\u{7f}z"), "xyz");
    }

    #[test]
    fn unicode_tag_characters_are_dropped() {
        // "ignore previous instructions" smuggled as invisible tag characters
        let hidden: String = "ignore"
            .chars()
            .map(|c| char::from_u32(0xe0000 + c as u32).unwrap())
            .collect();
        assert_eq!(line(&format!("safe{hidden}text")), "safetext");
        assert_eq!(text(&format!("a{hidden}\nb")), "a\nb");
    }

    #[test]
    fn line_breaks_become_one_space() {
        assert_eq!(line("a\r\n\r\nb\tc"), "a b c");
        assert_eq!(line("\n  a  \n"), "a");
        // a RIS value cannot start a new tag line
        assert!(!line("title\nER  - \nTY  - BOOK").contains('\n'));
    }

    #[test]
    fn text_keeps_paragraphs_and_normalises_newlines() {
        assert_eq!(text("a\r\nb\rc\u{2028}d\n\ne\tf"), "a\nb\nc\nd\n\ne\tf");
        assert_eq!(text("x\u{0}\u{1b}y"), "xy");
    }

    #[test]
    fn bibtex_values_cannot_close_their_field() {
        assert_eq!(bib_value("x}, note = {injected"), "x, note = {injected}");
        assert_eq!(bib_value("a {b} c"), "a {b} c");
        assert_eq!(bib_value("open {{ never closed"), "open {{ never closed}}");
        assert_eq!(bib_value("ends with backslash\\"), "ends with backslash");
        assert_eq!(bib_value("}}}"), "");
        // result is always balanced
        for s in [
            "}{", "{{}", "}{}{", "a\\}b", "{\\}", "\\{\\}", "\\\\}", "{\\",
        ] {
            let v = bib_value(s);
            assert_balanced(&v, s);
        }
    }

    /// Counts braces the way BibTeX reads them: a backslash makes the next character literal.
    fn assert_balanced(v: &str, src: &str) {
        let (mut d, mut esc) = (0i32, false);
        for c in v.chars() {
            if !esc {
                match c {
                    '{' => d += 1,
                    '}' => {
                        d -= 1;
                        assert!(d >= 0, "{src:?} -> {v:?}")
                    }
                    _ => {}
                }
            }
            esc = c == '\\' && !esc;
        }
        assert!(!esc, "{src:?} -> {v:?} ends escaped");
        assert_eq!(d, 0, "{src:?} -> {v:?}");
    }

    #[test]
    fn escaped_braces_are_literal_and_stay_balanced() {
        // `\}` is a literal brace for BibTeX: it must not close anything, and a `\{` must not open anything
        let v = bib_value("a\\}b");
        assert_balanced(&v, "a\\}b");
        let w = bib_value("x\\{y");
        assert_balanced(&w, "x\\{y");
    }

    #[test]
    fn keys_lose_everything_that_could_break_latex() {
        assert_eq!(key("smith2019"), "smith2019");
        assert_eq!(key("doe:2020-a.b/c+d_e"), "doe:2020-a.b/c+d_e");
        assert_eq!(key("a}\\write18{id}{"), "awrite18id");
        assert_eq!(key("x\u{202e}y z"), "xyz");
        assert_eq!(key("kéy_ü"), "kéy_ü");
        assert_eq!(key("a\u{200f}b\u{ad}c\u{200d}d\u{fe0f}"), "abcd");
    }

    #[test]
    fn short_cuts_on_character_boundaries() {
        assert_eq!(short("abc", 5), "abc");
        assert_eq!(short("ștefan cel mare", 6), "ștefa…");
        assert_eq!(short("😀😀😀😀", 3), "😀😀…");
        assert_eq!(short("x", 0), "…");
    }

    #[test]
    fn hostile_input_never_panics_and_output_has_no_hostile_chars() {
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let pool: Vec<char> = "ab{}\\\n\r\t\u{0}\u{1b}\u{202e}\u{200b}\u{e0041}\u{85}é😀 ,=#"
            .chars()
            .collect();
        for _ in 0..2000 {
            let n = (next() % 24) as usize;
            let input: String = (0..n)
                .map(|_| pool[(next() as usize) % pool.len()])
                .collect();
            let l = line(&input);
            assert!(l.chars().all(|c| !is_hostile(c) && c != '\n' && c != '\r'));
            assert_balanced(&bib_value(&input), &input);
            assert!(key(&input).chars().all(|c| !c.is_whitespace()));
            let _ = (text(&input), short(&input, 3));
        }
    }
}
