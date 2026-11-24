//! Language tools for a Romanian-first, eight-language product (EN, RO, FR, IT, ES, PT, DE, HU).
//!
//! Four parts, all std-only and independent of the model code:
//! 1. **Normalisation** of messy input ([`normalize`]): Romanian comma-below vs cedilla, Hungarian
//!    double acute, mojibake repair (UTF-8 read as Windows-1252, Latin-2 read as Latin-1), ligatures,
//!    invisible characters, quotes/dashes — with a change report and exact [`Normalized::revert`].
//!    Also [`strip_diacritics`] and Turkish-safe case mapping.
//! 2. **Language identification** ([`detect`], [`detect_segments`]) over 27 languages with embedded
//!    stop-word and distinctive-letter profiles. See "Accuracy" below.
//! 3. **Answer-language lock** ([`LangLock`], [`LangGuard`]): per-language token masks and logit biases
//!    built from a tokenizer vocabulary, usable with [`crate::sampling`], and a streaming drift watcher.
//! 4. **Romanian helpers**: number/date/currency formatting and parsing, and a conservative
//!    diacritics restoration pass ([`restore_ro`]) with explicit confidence flags.
//!
//! # Accuracy (honest limits)
//! * Identification uses small hand-written profiles (stop-words and distinctive letters), not a
//!   trained n-gram model. It is reliable for sentences of roughly five words or more in the eight
//!   product languages; one- or two-word inputs often have low confidence or no answer on purpose.
//! * Norwegian vs Danish and Czech vs Slovak overlap heavily and are the weakest pairs. Arabic
//!   script is reported as Arabic (Persian and Urdu are not distinguished), Devanagari as Hindi.
//! * The Romanian restoration lexicon holds ~650 accented word forms (about 750 lookup keys with
//!   the plain-word list; fewer than the 1500 originally planned) plus a few suffix heuristics
//!   (`-ație`, `-țiune`, `-tăți`, `-ează`, `-ească`). Rare words, names and jargon pass through
//!   unchanged. Ambiguous forms (`fata`, `va`, `sa`, `ca`, `alta`, `data`…) are resolved only by a
//!   handful of context rules; otherwise the letters all readings agree on are restored and the word
//!   is flagged, unless you ask for [`RestoreMode::Guess`]. Accuracy has not been measured on a corpus.
//! * Mojibake repair is deliberately conservative: it only rewrites sequences that decode to
//!   valid UTF-8 or that carry an unambiguous Romanian Latin-2 marker.
use crate::sampling::TokenMask;
use std::collections::HashMap;
use std::sync::OnceLock;

// ================================================================================ languages

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Lang {
    En,
    Ro,
    Fr,
    It,
    Es,
    Pt,
    De,
    Hu,
    Nl,
    Pl,
    Cs,
    Sk,
    Sv,
    Da,
    No,
    Fi,
    Tr,
    Ru,
    Uk,
    Bg,
    El,
    Ar,
    He,
    Zh,
    Ja,
    Ko,
    Hi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Script {
    Latin,
    Cyrillic,
    Greek,
    Arabic,
    Hebrew,
    Devanagari,
    Han,
    Kana,
    Hangul,
    Other,
    Common,
}

impl Script {
    pub fn bit(self) -> u16 {
        match self {
            Script::Latin => 1,
            Script::Cyrillic => 2,
            Script::Greek => 4,
            Script::Arabic => 8,
            Script::Hebrew => 16,
            Script::Devanagari => 32,
            Script::Han => 64,
            Script::Kana => 128,
            Script::Hangul => 256,
            Script::Other => 512,
            Script::Common => 0,
        }
    }
}

impl Lang {
    pub const ALL: [Lang; 27] = [
        Lang::En,
        Lang::Ro,
        Lang::Fr,
        Lang::It,
        Lang::Es,
        Lang::Pt,
        Lang::De,
        Lang::Hu,
        Lang::Nl,
        Lang::Pl,
        Lang::Cs,
        Lang::Sk,
        Lang::Sv,
        Lang::Da,
        Lang::No,
        Lang::Fi,
        Lang::Tr,
        Lang::Ru,
        Lang::Uk,
        Lang::Bg,
        Lang::El,
        Lang::Ar,
        Lang::He,
        Lang::Zh,
        Lang::Ja,
        Lang::Ko,
        Lang::Hi,
    ];
    /// The eight languages the product ships in.
    pub const PRODUCT: [Lang; 8] = [
        Lang::En,
        Lang::Ro,
        Lang::Fr,
        Lang::It,
        Lang::Es,
        Lang::Pt,
        Lang::De,
        Lang::Hu,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ro => "ro",
            Lang::Fr => "fr",
            Lang::It => "it",
            Lang::Es => "es",
            Lang::Pt => "pt",
            Lang::De => "de",
            Lang::Hu => "hu",
            Lang::Nl => "nl",
            Lang::Pl => "pl",
            Lang::Cs => "cs",
            Lang::Sk => "sk",
            Lang::Sv => "sv",
            Lang::Da => "da",
            Lang::No => "no",
            Lang::Fi => "fi",
            Lang::Tr => "tr",
            Lang::Ru => "ru",
            Lang::Uk => "uk",
            Lang::Bg => "bg",
            Lang::El => "el",
            Lang::Ar => "ar",
            Lang::He => "he",
            Lang::Zh => "zh",
            Lang::Ja => "ja",
            Lang::Ko => "ko",
            Lang::Hi => "hi",
        }
    }
    /// Accepts `ro`, `RO`, `ro-RO`, `ro_RO`, and a few ISO 639-2 / legacy codes.
    pub fn from_code(s: &str) -> Option<Lang> {
        let l = s.trim().to_ascii_lowercase();
        let head = l.split(['-', '_']).next().unwrap_or("");
        Some(match head {
            "en" | "eng" => Lang::En,
            "ro" | "ron" | "rum" | "mo" => Lang::Ro,
            "fr" | "fra" | "fre" => Lang::Fr,
            "it" | "ita" => Lang::It,
            "es" | "spa" => Lang::Es,
            "pt" | "por" => Lang::Pt,
            "de" | "deu" | "ger" => Lang::De,
            "hu" | "hun" => Lang::Hu,
            "nl" | "nld" | "dut" => Lang::Nl,
            "pl" | "pol" => Lang::Pl,
            "cs" | "ces" | "cze" => Lang::Cs,
            "sk" | "slk" | "slo" => Lang::Sk,
            "sv" | "swe" => Lang::Sv,
            "da" | "dan" => Lang::Da,
            "no" | "nb" | "nn" | "nor" => Lang::No,
            "fi" | "fin" => Lang::Fi,
            "tr" | "tur" => Lang::Tr,
            "ru" | "rus" => Lang::Ru,
            "uk" | "ukr" => Lang::Uk,
            "bg" | "bul" => Lang::Bg,
            "el" | "ell" | "gre" => Lang::El,
            "ar" | "ara" => Lang::Ar,
            "he" | "iw" | "heb" => Lang::He,
            "zh" | "zho" | "chi" => Lang::Zh,
            "ja" | "jpn" => Lang::Ja,
            "ko" | "kor" => Lang::Ko,
            "hi" | "hin" => Lang::Hi,
            _ => return None,
        })
    }
    pub fn is_product(self) -> bool {
        Lang::PRODUCT.contains(&self)
    }
    /// Bitmask of the scripts this language is normally written in.
    pub fn scripts(self) -> u16 {
        match self {
            Lang::Ru | Lang::Uk | Lang::Bg => Script::Cyrillic.bit(),
            Lang::El => Script::Greek.bit(),
            Lang::Ar => Script::Arabic.bit(),
            Lang::He => Script::Hebrew.bit(),
            Lang::Hi => Script::Devanagari.bit(),
            Lang::Zh => Script::Han.bit(),
            Lang::Ja => Script::Han.bit() | Script::Kana.bit(),
            Lang::Ko => Script::Hangul.bit() | Script::Han.bit(),
            _ => Script::Latin.bit(),
        }
    }
    pub fn script(self) -> Script {
        match self {
            Lang::Ru | Lang::Uk | Lang::Bg => Script::Cyrillic,
            Lang::El => Script::Greek,
            Lang::Ar => Script::Arabic,
            Lang::He => Script::Hebrew,
            Lang::Hi => Script::Devanagari,
            Lang::Zh => Script::Han,
            Lang::Ja => Script::Kana,
            Lang::Ko => Script::Hangul,
            _ => Script::Latin,
        }
    }
}

/// Script of one character. Non-letters are [`Script::Common`].
pub fn script_of(c: char) -> Script {
    let u = c as u32;
    if u < 0x80 {
        return if c.is_ascii_alphabetic() {
            Script::Latin
        } else {
            Script::Common
        };
    }
    if !c.is_alphabetic() {
        return Script::Common;
    }
    match u {
        0xAA
        | 0xBA
        | 0xC0..=0x24F
        | 0x1E00..=0x1EFF
        | 0x2C60..=0x2C7F
        | 0xA720..=0xA7FF
        | 0xFF21..=0xFF3A
        | 0xFF41..=0xFF5A => {
            if u == 0xD7 || u == 0xF7 {
                Script::Common
            } else {
                Script::Latin
            }
        }
        0x370..=0x3FF | 0x1F00..=0x1FFF => Script::Greek,
        0x400..=0x52F | 0x1C80..=0x1C8F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => Script::Cyrillic,
        0x590..=0x5FF | 0xFB1D..=0xFB4F => Script::Hebrew,
        0x600..=0x6FF | 0x750..=0x77F | 0x8A0..=0x8FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => {
            Script::Arabic
        }
        0x900..=0x97F => Script::Devanagari,
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => Script::Kana,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F => Script::Han,
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => Script::Hangul,
        _ => Script::Other,
    }
}

// ================================================================================ normalisation

/// Which Romanian s/t form to produce. Comma-below (ș ț) is the Unicode-correct one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoForm {
    CommaBelow,
    Cedilla,
    Keep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HuFix {
    Auto,
    Always,
    Never,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DashMode {
    Keep,
    Hyphens,
    All,
}

#[derive(Clone, Debug)]
pub struct NormOptions {
    pub ro: RoForm,
    pub hu: HuFix,
    pub mojibake: bool,
    pub ligatures: bool,
    pub invisible: bool,
    pub spaces: bool,
    pub quotes: bool,
    pub dashes: DashMode,
    pub compose: bool,
}
impl Default for NormOptions {
    fn default() -> Self {
        NormOptions {
            ro: RoForm::CommaBelow,
            hu: HuFix::Auto,
            mojibake: true,
            ligatures: true,
            invisible: true,
            spaces: true,
            quotes: true,
            dashes: DashMode::Hyphens,
            compose: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ChangeKind {
    RoForm,
    HuDoubleAcute,
    Mojibake,
    Latin2,
    Ligature,
    Invisible,
    Space,
    Quote,
    Dash,
    Compose,
}

/// One edit. `start` is the byte offset of `after` in the *output* text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    pub kind: ChangeKind,
    pub start: usize,
    pub before: String,
    pub after: String,
}

#[derive(Clone, Debug, Default)]
pub struct Normalized {
    pub text: String,
    pub changes: Vec<Change>,
}

impl Normalized {
    /// Exactly reconstructs the input that was passed to [`normalize_with`].
    pub fn revert(&self) -> String {
        let mut s = self.text.clone();
        for c in self.changes.iter().rev() {
            s.replace_range(c.start..c.start + c.after.len(), &c.before);
        }
        s
    }
    pub fn is_unchanged(&self) -> bool {
        self.changes.is_empty()
    }
    pub fn counts(&self) -> Vec<(ChangeKind, usize)> {
        let mut m: HashMap<ChangeKind, usize> = HashMap::new();
        for c in &self.changes {
            *m.entry(c.kind).or_default() += 1;
        }
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort();
        v
    }
    /// Human-readable diagnostic, one line per change kind.
    pub fn report(&self) -> String {
        if self.changes.is_empty() {
            return "no changes".to_string();
        }
        let mut out = String::new();
        for (k, n) in self.counts() {
            let ex = self.changes.iter().find(|c| c.kind == k).unwrap();
            out.push_str(&format!(
                "{:?}: {} (e.g. {:?} -> {:?})\n",
                k, n, ex.before, ex.after
            ));
        }
        out
    }
}

/// Windows-1252 byte for a character, as a mis-decoding of UTF-8 would have produced it.
fn cp1252_byte(c: char) -> Option<u8> {
    let u = c as u32;
    if (0xA0..=0xFF).contains(&u) || (0x80..=0x9F).contains(&u) {
        return Some(u as u8);
    }
    const T: [(u32, u8); 27] = [
        (0x20AC, 0x80),
        (0x201A, 0x82),
        (0x0192, 0x83),
        (0x201E, 0x84),
        (0x2026, 0x85),
        (0x2020, 0x86),
        (0x2021, 0x87),
        (0x02C6, 0x88),
        (0x2030, 0x89),
        (0x0160, 0x8A),
        (0x2039, 0x8B),
        (0x0152, 0x8C),
        (0x017D, 0x8E),
        (0x2018, 0x91),
        (0x2019, 0x92),
        (0x201C, 0x93),
        (0x201D, 0x94),
        (0x2022, 0x95),
        (0x2013, 0x96),
        (0x2014, 0x97),
        (0x02DC, 0x98),
        (0x2122, 0x99),
        (0x0161, 0x9A),
        (0x203A, 0x9B),
        (0x0153, 0x9C),
        (0x017E, 0x9E),
        (0x0178, 0x9F),
    ];
    T.iter().find(|(k, _)| *k == u).map(|(_, b)| *b)
}

/// Tries to read a UTF-8 sequence that was mis-decoded as Windows-1252/Latin-1 at `i`.
/// Returns (decoded char, number of input chars consumed).
fn unmojibake_at(chars: &[char], i: usize) -> Option<(char, usize)> {
    let b0 = cp1252_byte(chars[i])?;
    let need = match b0 {
        0xC2..=0xDF => 1,
        0xE0..=0xEF => 2,
        0xF0..=0xF4 => 3,
        _ => return None,
    };
    let mut bytes = vec![b0];
    for k in 1..=need {
        let b = cp1252_byte(*chars.get(i + k)?)?;
        if !(0x80..=0xBF).contains(&b) {
            return None;
        }
        bytes.push(b);
    }
    let s = std::str::from_utf8(&bytes).ok()?;
    let c = s.chars().next()?;
    if c.is_control() || c == '\u{FFFD}' || (0xE000..=0xF8FF).contains(&(c as u32)) {
        return None;
    }
    Some((c, need + 1))
}

fn compose_pair(base: char, mark: char) -> Option<char> {
    const T: [(u32, &str); 11] = [
        (
            0x301,
            "aá eé ií oó uú yý AÁ EÉ IÍ OÓ UÚ YÝ cć nń sś zź lĺ rŕ CĆ NŃ SŚ ZŹ",
        ),
        (0x300, "aà eè iì oò uù AÀ EÈ IÌ OÒ UÙ"),
        (0x302, "aâ eê iî oô uû AÂ EÊ IÎ OÔ UÛ cĉ"),
        (0x303, "aã nñ oõ AÃ NÑ OÕ"),
        (0x308, "aä eë iï oö uü yÿ AÄ EË IÏ OÖ UÜ"),
        (0x327, "cç sş tţ CÇ SŞ TŢ"),
        (0x326, "sș tț SȘ TȚ"),
        (0x306, "aă AĂ gğ GĞ"),
        (0x30C, "cč sš zž eě rř nň dď tť CČ SŠ ZŽ EĚ RŘ NŇ DĎ TŤ"),
        (0x30B, "oő uű OŐ UŰ"),
        (0x30A, "aå uů AÅ UŮ"),
    ];
    let m = mark as u32;
    let (_, pairs) = T.iter().find(|(k, _)| *k == m)?;
    pairs.split(' ').find_map(|p| {
        let mut it = p.chars();
        let (b, r) = (it.next()?, it.next()?);
        if b == base { Some(r) } else { None }
    })
}

fn is_latin_letter(c: char) -> bool {
    c.is_alphabetic() && script_of(c) == Script::Latin
}

/// Normalises with default options.
pub fn normalize(text: &str) -> Normalized {
    normalize_with(text, &NormOptions::default())
}

pub fn normalize_with(text: &str, o: &NormOptions) -> Normalized {
    let chars: Vec<char> = text.chars().collect();
    let hu_on = match o.hu {
        HuFix::Always => true,
        HuFix::Never => false,
        HuFix::Auto => {
            chars
                .iter()
                .any(|&c| c == 'ő' || c == 'ű' || c == 'Ő' || c == 'Ű')
                || {
                    let d = detect(text);
                    d.lang == Some(Lang::Hu) && d.reliable
                }
        }
    };
    // Latin-2 read as Latin-1 evidence: in-word º ª þ Þ not following a digit.
    let mut l2 = 0usize;
    for (i, &c) in chars.iter().enumerate() {
        if matches!(c, 'º' | 'ª' | 'þ' | 'Þ')
            && chars.get(i + 1).is_some_and(|n| n.is_alphabetic())
            && !(i > 0 && chars[i - 1].is_ascii_digit())
        {
            l2 += 1;
        }
    }
    let l2_on = l2 > 0;
    let mut out = String::with_capacity(text.len());
    let mut changes: Vec<Change> = vec![];
    let push = |out: &mut String,
                changes: &mut Vec<Change>,
                kind: ChangeKind,
                before: String,
                after: &str| {
        changes.push(Change {
            kind,
            start: out.len(),
            before,
            after: after.to_string(),
        });
        out.push_str(after);
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // 1. mojibake (UTF-8 seen as cp1252)
        if o.mojibake
            && c as u32 >= 0x80
            && let Some((d, n)) = unmojibake_at(&chars, i)
        {
            let before: String = chars[i..i + n].iter().collect();
            // repaired char may itself need Romanian canonicalisation
            let after = post_char(d, o);
            push(&mut out, &mut changes, ChangeKind::Mojibake, before, &after);
            i += n;
            continue;
        }
        // 2. combining marks composed with the previous base
        if o.compose
            && i + 1 < chars.len()
            && ('\u{300}'..='\u{36F}').contains(&chars[i + 1])
            && let Some(r) = compose_pair(c, chars[i + 1])
        {
            let before: String = chars[i..i + 2].iter().collect();
            let r2 = ro_form(r, o.ro);
            push(
                &mut out,
                &mut changes,
                ChangeKind::Compose,
                before,
                &r2.to_string(),
            );
            i += 2;
            continue;
        }
        // 3. Latin-2 mis-decoding, Romanian
        if l2_on {
            let r = match c {
                'º' => Some('ş'),
                'ª' => Some('Ş'),
                'þ' => Some('ţ'),
                'Þ' => Some('Ţ'),
                'ã' => Some('ă'),
                'Ã' => Some('Ă'),
                _ => None,
            };
            if let Some(r) = r {
                let fixed = ro_form(r, o.ro);
                let marker = matches!(c, 'º' | 'ª' | 'þ' | 'Þ');
                let skip = marker
                    && (!chars.get(i + 1).is_some_and(|n| n.is_alphabetic())
                        || (i > 0 && chars[i - 1].is_ascii_digit()));
                if !skip {
                    push(
                        &mut out,
                        &mut changes,
                        ChangeKind::Latin2,
                        c.to_string(),
                        &fixed.to_string(),
                    );
                    i += 1;
                    continue;
                }
            }
        }
        // 4. Hungarian double acute
        if hu_on {
            let r = match c {
                'õ' => Some('ő'),
                'Õ' => Some('Ő'),
                'û' => Some('ű'),
                'Û' => Some('Ű'),
                _ => None,
            };
            if let Some(r) = r {
                push(
                    &mut out,
                    &mut changes,
                    ChangeKind::HuDoubleAcute,
                    c.to_string(),
                    &r.to_string(),
                );
                i += 1;
                continue;
            }
        }
        // 5. Romanian s/t form
        if o.ro != RoForm::Keep {
            let r = ro_form(c, o.ro);
            if r != c {
                push(
                    &mut out,
                    &mut changes,
                    ChangeKind::RoForm,
                    c.to_string(),
                    &r.to_string(),
                );
                i += 1;
                continue;
            }
        }
        // 6. ligatures
        if o.ligatures {
            let l = match c {
                'ﬀ' => "ff",
                'ﬁ' => "fi",
                'ﬂ' => "fl",
                'ﬃ' => "ffi",
                'ﬄ' => "ffl",
                'ﬅ' | 'ﬆ' => "st",
                _ => "",
            };
            if !l.is_empty() {
                push(
                    &mut out,
                    &mut changes,
                    ChangeKind::Ligature,
                    c.to_string(),
                    l,
                );
                i += 1;
                continue;
            }
        }
        // 7. invisible characters
        if o.invisible {
            let drop = match c {
                '\u{200B}' | '\u{2060}' | '\u{FEFF}' | '\u{00AD}' => true,
                '\u{200C}' | '\u{200D}' => {
                    let p = i.checked_sub(1).map(|k| chars[k]);
                    let n = chars.get(i + 1).copied();
                    p.is_some_and(is_latin_letter) && n.is_some_and(is_latin_letter)
                }
                _ => false,
            };
            if drop {
                push(
                    &mut out,
                    &mut changes,
                    ChangeKind::Invisible,
                    c.to_string(),
                    "",
                );
                i += 1;
                continue;
            }
        }
        // 8. spaces
        if o.spaces
            && matches!(
                c,
                '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}'
            )
        {
            push(
                &mut out,
                &mut changes,
                ChangeKind::Space,
                c.to_string(),
                " ",
            );
            i += 1;
            continue;
        }
        // 9. quotes
        if o.quotes {
            let q = match c {
                '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => "'",
                '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' => "\"",
                _ => "",
            };
            if !q.is_empty() {
                push(&mut out, &mut changes, ChangeKind::Quote, c.to_string(), q);
                i += 1;
                continue;
            }
        }
        // 10. dashes
        if o.dashes != DashMode::Keep {
            let hy = matches!(c, '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2212}');
            let long = matches!(c, '\u{2013}' | '\u{2014}' | '\u{2015}');
            if hy || (long && o.dashes == DashMode::All) {
                push(&mut out, &mut changes, ChangeKind::Dash, c.to_string(), "-");
                i += 1;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    Normalized { text: out, changes }
}

/// Quote, dash and Romanian-form mapping applied to a character that a repair step just produced.
fn post_char(c: char, o: &NormOptions) -> String {
    if o.quotes {
        match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => return "'".into(),
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' => return "\"".into(),
            _ => {}
        }
    }
    if o.dashes != DashMode::Keep {
        let hy = matches!(c, '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2212}');
        let long = matches!(c, '\u{2013}' | '\u{2014}' | '\u{2015}');
        if hy || (long && o.dashes == DashMode::All) {
            return "-".into();
        }
    }
    ro_form(c, o.ro).to_string()
}

fn ro_form(c: char, f: RoForm) -> char {
    match f {
        RoForm::Keep => c,
        RoForm::CommaBelow => match c {
            'ş' => 'ș',
            'Ş' => 'Ș',
            'ţ' => 'ț',
            'Ţ' => 'Ț',
            _ => c,
        },
        RoForm::Cedilla => match c {
            'ș' => 'ş',
            'Ș' => 'Ş',
            'ț' => 'ţ',
            'Ț' => 'Ţ',
            _ => c,
        },
    }
}

// ---------------------------------------------------------------------------- case & diacritics

/// Lowercase with Turkish/Azeri dotted/dotless i rules: `I`→`ı`, `İ`→`i`.
pub fn to_lower_tr(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'I' => o.push('ı'),
            'İ' => o.push('i'),
            _ => o.extend(c.to_lowercase()),
        }
    }
    o
}
/// Uppercase with Turkish rules: `i`→`İ`, `ı`→`I`.
pub fn to_upper_tr(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'i' => o.push('İ'),
            'ı' => o.push('I'),
            _ => o.extend(c.to_uppercase()),
        }
    }
    o
}
pub fn to_lower_lang(s: &str, l: Lang) -> String {
    if l == Lang::Tr {
        to_lower_tr(s)
    } else {
        s.to_lowercase()
    }
}
pub fn to_upper_lang(s: &str, l: Lang) -> String {
    if l == Lang::Tr {
        to_upper_tr(s)
    } else {
        s.to_uppercase()
    }
}
/// Case-insensitive equality under the language's case rules.
pub fn eq_fold(a: &str, b: &str, l: Lang) -> bool {
    to_lower_lang(a, l) == to_lower_lang(b, l)
}

fn fold_table() -> &'static HashMap<char, &'static str> {
    static T: OnceLock<HashMap<char, &'static str>> = OnceLock::new();
    T.get_or_init(|| {
        let groups: [(&str, &str); 40] = [
            ("A", "ÀÁÂÃÄÅĀĂĄǍ"),
            ("a", "àáâãäåāăąǎ"),
            ("C", "ÇĆĈĊČ"),
            ("c", "çćĉċč"),
            ("D", "ĎĐÐ"),
            ("d", "ďđð"),
            ("E", "ÈÉÊËĒĔĖĘĚ"),
            ("e", "èéêëēĕėęě"),
            ("G", "ĜĞĠĢ"),
            ("g", "ĝğġģ"),
            ("H", "ĤĦ"),
            ("h", "ĥħ"),
            ("I", "ÌÍÎÏĨĪĬĮİ"),
            ("i", "ìíîïĩīĭįı"),
            ("J", "Ĵ"),
            ("j", "ĵ"),
            ("K", "Ķ"),
            ("k", "ķ"),
            ("L", "ĹĻĽĿŁ"),
            ("l", "ĺļľŀł"),
            ("N", "ÑŃŅŇ"),
            ("n", "ñńņň"),
            ("O", "ÒÓÔÕÖØŌŎŐ"),
            ("o", "òóôõöøōŏő"),
            ("R", "ŔŖŘ"),
            ("r", "ŕŗř"),
            ("S", "ŚŜŞŠȘ"),
            ("s", "śŝşšș"),
            ("T", "ŢŤŦȚ"),
            ("t", "ţťŧț"),
            ("U", "ÙÚÛÜŨŪŬŮŰŲ"),
            ("u", "ùúûüũūŭůűų"),
            ("W", "Ŵ"),
            ("w", "ŵ"),
            ("Y", "ÝŶŸ"),
            ("y", "ýÿŷ"),
            ("Z", "ŹŻŽ"),
            ("z", "źżž"),
            ("ss", "ß"),
            ("ae", "æ"),
        ];
        let mut m = HashMap::new();
        for (r, set) in groups {
            for c in set.chars() {
                m.insert(c, r);
            }
        }
        m.insert('Æ', "AE");
        m.insert('œ', "oe");
        m.insert('Œ', "OE");
        m.insert('Þ', "Th");
        m.insert('þ', "th");
        m
    })
}

/// Removes diacritics for accent-insensitive search: `Școală` → `Scoala`, `Straße` → `Strasse`.
/// Combining marks are dropped; letters without a Latin base (Cyrillic, Greek, CJK…) are kept.
pub fn strip_diacritics(s: &str) -> String {
    let t = fold_table();
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        if ('\u{300}'..='\u{36F}').contains(&c) {
            continue;
        }
        match t.get(&c) {
            Some(r) => o.push_str(r),
            None => o.push(c),
        }
    }
    o
}

// ================================================================================ identification

struct RawProfile {
    lang: Lang,
    words: &'static str,
    suffixes: &'static str,
    chars: &'static [(&'static str, f32)],
}

// Words: trailing `*` = strong marker (x2), `~` = weak (x0.5). Diacritic-free spellings are derived automatically.
const RAW: &[RawProfile] = &[
    RawProfile {
        lang: Lang::En,
        suffixes: "ing ould ough ight ness",
        chars: &[],
        words: "the* of and to a in is that it was for on are with as be this at by have from or an but not you all can her his they will one would there their what so up out if about who get which go me when make like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us we he she do has had were been am my hello* hi hey thanks* thank please* yes morning",
    },
    RawProfile {
        lang: Lang::Ro,
        suffixes: "ului ilor ește ație iune ează ească ității",
        chars: &[("ășțţ", 3.0), ("şâî", 0.9)],
        words: "și* în* de la cu pe un o nu* că ca să sa* este* sunt* pentru* care mai ce se din al ale dar sau mi am ai are au fost fi foarte bine cum unde când cine acest această acesta aceste acestea dacă după între fără despre prin spre până peste sub lui lor ei ele el ea eu tu noi voi vă va te ne mă ți îți îmi își tot toate mult puțin ceva nimic aici acolo acum atunci mâine azi ieri bună* ziua* seara dimineața mulțumesc* mulțumim vreau vrei vrea pot poți poate trebuie face faceți* faci fac era vor mea meu mele tău ta sale său unei unui unor ci nici deci așa încă însă deși",
    },
    RawProfile {
        lang: Lang::Fr,
        suffixes: "eux ique aient eau",
        chars: &[("çœùëÿ", 2.0), ("àâêèîïôû", 0.7), ("é", 0.3)],
        words: "le la les un une des du de et est en que qui dans pour pas sur avec ce il elle nous vous ils elles je tu ne se au aux par plus mais ou où comme être avoir fait faire sont ont était très bien merci* bonjour* salut oui non cette ces son sa ses leur leurs mon ma mes ton ta tes notre votre aussi tout tous toute encore alors donc car sans sous entre chez depuis peut peux veux voudrais j~ l~ d~ qu~ c~ n~ s~ t~ m~",
    },
    RawProfile {
        lang: Lang::It,
        suffixes: "zione mente ità ssimo",
        chars: &[("ìòù", 1.0), ("àè", 0.5)],
        words: "il lo la le gli* i un una uno di del della* dei delle che e è non per con su da in sono ho hai ha abbiamo avete hanno come ma se più questo questa questi queste quello quella anche molto bene grazie* ciao* buongiorno* buonasera* prego sì io tu lui lei noi voi loro mi ti ci vi si mio mia tuo tua suo sua nostro vostro essere avere fare fatto cosa dove quando chi perché* quindi però sempre ancora già tutto tutti tutte nel nella nei nelle sul sulla al alla agli alle dal dalla degli dagli allo dallo nello sullo dell~ all~ un~",
    },
    RawProfile {
        lang: Lang::Es,
        suffixes: "ción ando iendo idad",
        chars: &[("ñ¿¡", 3.0), ("áíóú", 0.6), ("é", 0.2)],
        words: "el la los las un una unos unas de del que y en es no por con para su sus se lo como más pero sí si al a le les me te nos mi tu yo él ella nosotros ellos ellas esto esta estos estas ese esa eso muy bien gracias* hola* buenos* días* buenas tardes noches está están ser estar haber hacer hecho tiene tengo tienen qué* cómo dónde cuándo quién porque cuando donde también ya todo todos todas hay* había fue era son soy somos sin sobre entre hasta desde durante usted ustedes",
    },
    RawProfile {
        lang: Lang::Pt,
        suffixes: "ção ções idade",
        chars: &[("ãõ", 3.0), ("çâêô", 0.8), ("à", 0.5)],
        words: "o a os as um uma uns umas de do da dos das que e é em no na nos nas por para com não* se ao à aos às como mais mas foi são* ser estar ter tem tenho têm* eu tu ele ela nós vocês eles elas isto isso este esta esse essa muito bem obrigado* obrigada* olá* bom boa dia tarde noite você* também já tudo todos todas há havia fazer feito quando onde quem porque então sim ainda sobre entre até desde pelo pela pelos pelas",
    },
    RawProfile {
        lang: Lang::De,
        suffixes: "ung keit heit lich isch schaft chen",
        chars: &[("ß", 3.0), ("äöü", 1.0)],
        words: "der die das den dem des ein eine einen einem einer eines und ist nicht zu in von mit für auf dass* es sich auch an als aber oder wie wir sie er ich du ihr sind war hat haben wird werden kann können muss sehr gut danke* hallo guten* tag morgen abend bitte ja nein wenn dann noch nur schon mehr diese dieser dieses bei nach aus über unter vor zwischen durch gegen ohne um mein dein sein unser euer",
    },
    RawProfile {
        lang: Lang::Hu,
        suffixes: "ság ség ban ben nak nek val vel ként",
        chars: &[("őű", 4.0), ("öü", 0.6), ("áéíóú", 0.5)],
        words: "a az és* hogy* nem* egy is van volt vagy de ez ezt azt ami ahol amikor mint meg el be ki fel le már még csak nagyon jó jól köszönöm* szia* szép* napot* kívánok igen* nincs lesz kell lehet vagyok vagyunk vagytok vannak voltam én te ő mi ti ők nekem neked neki nekünk nektek nekik hogyan* mit* miért* hol mikor mennyi",
    },
    RawProfile {
        lang: Lang::Nl,
        suffixes: "heid lijk ische",
        chars: &[("ĳ", 2.0)],
        words: "de het* een en van in is dat op te zijn voor met niet aan er maar om ook als dan nog bij wel uit naar heeft hebben was waren wordt worden kan kunnen moet zal zou ik je jij hij zij we wij jullie ze mijn jouw haar ons onze hun dit deze die wat wie waar wanneer hoe waarom heel erg goed dank* bedankt* hallo goedemorgen alstublieft* ja nee geen niets alles veel meer",
    },
    RawProfile {
        lang: Lang::Pl,
        suffixes: "ość cji ski",
        chars: &[("ąćęłńśźż", 3.0), ("ó", 0.8)],
        words: "i w nie* na się* z do to że jest jak ale o po co tak za od przez dla czy są był była było będzie być mam masz ma mamy macie mają ja ty on ona my wy oni one mnie ciebie jego jej nas was ich bardzo dobrze dziękuję* cześć* dzień* dobry* proszę też już jeszcze tylko który która które kiedy gdzie dlaczego",
    },
    RawProfile {
        lang: Lang::Cs,
        suffixes: "ost ství ový ání ění",
        chars: &[("ěřůťďň", 2.5), ("čšžý", 1.0), ("áéíóú", 0.3)],
        words: "a v se* na je že to s z do o i ale jako* by bylo byl byla jsou jsem* jsi jsme jste který která které co kdy kde proč ano ne také už ještě jen velmi* dobře děkuji* ahoj* dobrý* den* prosím* mám máš má máme máte mají já ty on ona my vy oni pro od po při bez před mezi pod nad podle",
    },
    RawProfile {
        lang: Lang::Sk,
        suffixes: "osť ový ánie",
        chars: &[("ľĺŕô", 3.0), ("ä", 1.5), ("čšžý", 0.8)],
        words: "a v sa* na je že to s z do o i ale ako* by bolo bol bola sú* som* si sme ste ktorý ktorá ktoré čo kedy kde prečo áno nie tiež už ešte len veľmi* dobre ďakujem* ahoj* dobrý* deň* prosím* mám máš má máme máte majú ja ty on ona my vy oni pre od po pri bez pred medzi pod nad podľa",
    },
    RawProfile {
        lang: Lang::Sv,
        suffixes: "ning het ande",
        chars: &[("åäö", 1.2)],
        words: "och* att* det är* en som på av för med inte* jag* har vi om så men du han hon de den ett till från var kan ska vill kommer mycket bra tack* hej god dag kväll snälla ja nej också redan bara när där här vad vem varför hur alla något ingen mig dig sig oss er min din sin vår",
    },
    RawProfile {
        lang: Lang::Da,
        suffixes: "hed ing",
        chars: &[("æø", 2.5), ("å", 1.2)],
        words: "og at det er en som på af* for med ikke jeg har vi om så men du han hun de den et til fra var kan skal vil kommer meget godt tak* hej dag aften venligst ja nej også allerede bare når hvor her hvad* hvem hvorfor hvordan alle noget ingen mig dig* sig os jer min din sin vores",
    },
    RawProfile {
        lang: Lang::No,
        suffixes: "het ing",
        chars: &[("æø", 2.5), ("å", 1.2)],
        words: "og at det er en som på av* for med ikke jeg har vi om så men du han hun de den et til fra var kan skal vil kommer veldig bra takk* hei* god dag kveld vær snill ja nei også allerede bare når hvor her hva* hvem hvorfor hvordan alle noe ingen meg* deg* seg oss dere min din sin vår",
    },
    RawProfile {
        lang: Lang::Fi,
        suffixes: "inen ssa ssä lla llä nen ista",
        chars: &[("äö", 1.0)],
        words: "ja on ei että* se oli kun mutta hän olen minä sinä tämä joka kuin tai myös ovat niin vain mitä kiitos* hyvää* päivää* huomenta moi* terve olla ole ollut kyllä en et emme ette he me te mikä kuka missä milloin miksi miten kaikki jotain mitään paljon hyvin sitten vielä jo",
    },
    RawProfile {
        lang: Lang::Tr,
        suffixes: "lar ler dır dir mak mek",
        chars: &[("ğı", 4.0), ("şçöü", 0.8)],
        words: "ve bir bu için* ile de da ben sen o biz siz onlar çok değil* ama gibi ne mi mı mu daha olarak var yok merhaba* teşekkürler* teşekkür* nasılsın* nasıl evet hayır lütfen* şu ki çünkü ancak veya hem her bütün kendi sonra önce kadar ise olan oldu olur",
    },
    RawProfile {
        lang: Lang::Ru,
        suffixes: "",
        chars: &[("ыэё", 2.0), ("ъ", 0.3)],
        words: "и в не на я что он с как это но они мы вы от по из за для же бы был была было были есть быть привет* здравствуйте* спасибо* пожалуйста* да нет очень хорошо все всё так уже еще ещё только когда где почему кто его её их мой твой наш ваш этот эта эти то тут там",
    },
    RawProfile {
        lang: Lang::Uk,
        suffixes: "",
        chars: &[("іїєґ", 3.0)],
        words: "і в не на я що він з як це але вони ми ви від по із за для же би був була було були є бути привіт* дякую* будь ласка так ні дуже добре все вже ще тільки коли де чому хто його її їх мій твій наш ваш цей ця ці то тут там та й",
    },
    RawProfile {
        lang: Lang::Bg,
        suffixes: "",
        chars: &[("ъ", 1.5)],
        words: "и в не на аз че той с как това но те ние вие от по из за да е са бил била било били има съм си сме сте здравей* здравейте* благодаря* моля много добре всичко вече още само кога къде защо кой",
    },
];

struct Prof {
    lang: Lang,
    words: HashMap<String, f32>,
    chars: Vec<(char, f32)>,
    suffixes: Vec<&'static str>,
}

fn profiles() -> &'static Vec<Prof> {
    static P: OnceLock<Vec<Prof>> = OnceLock::new();
    P.get_or_init(|| {
        RAW.iter()
            .map(|r| {
                let mut words: HashMap<String, f32> = HashMap::new();
                for w in r.words.split_whitespace() {
                    let (w, wt) = if let Some(x) = w.strip_suffix('*') {
                        (x, 2.0)
                    } else if let Some(x) = w.strip_suffix('~') {
                        (x, 0.5)
                    } else {
                        (w, 1.0)
                    };
                    words.insert(w.to_string(), wt);
                }
                let extra: Vec<(String, f32)> = words
                    .iter()
                    .filter_map(|(k, v)| {
                        let f = strip_diacritics(k);
                        if f != *k { Some((f, v * 0.9)) } else { None }
                    })
                    .collect();
                for (k, v) in extra {
                    words.entry(k).or_insert(v);
                }
                let mut chars = vec![];
                for (set, w) in r.chars {
                    for c in set.chars() {
                        if c != ' ' {
                            chars.push((c, *w));
                        }
                    }
                }
                Prof {
                    lang: r.lang,
                    words,
                    chars,
                    suffixes: r.suffixes.split_whitespace().collect(),
                }
            })
            .collect()
    })
}

#[derive(Clone, Debug)]
pub struct Detection {
    /// Best guess, `None` when there is no evidence at all (digits only, one unknown word, …).
    pub lang: Option<Lang>,
    /// 0..1, margin-based: low for short or ambiguous input. Not a calibrated probability.
    pub confidence: f32,
    /// `confidence >= 0.5` and enough evidence; safe to act on automatically.
    pub reliable: bool,
    pub script: Script,
    pub letters: usize,
    /// Evidence shares, best first (top 5), summing to ≤ 1.
    pub scores: Vec<(Lang, f32)>,
}
impl Detection {
    pub fn prob(&self, l: Lang) -> f32 {
        self.scores
            .iter()
            .find(|(k, _)| *k == l)
            .map_or(0.0, |(_, p)| *p)
    }
}

fn lower_char(c: char) -> char {
    match c {
        'İ' => 'i',
        'I' => 'i',
        _ => c.to_lowercase().next().unwrap_or(c),
    }
}

fn words_of(text: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphabetic() {
            cur.push(lower_char(c));
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn script_index(s: Script) -> usize {
    match s {
        Script::Latin => 0,
        Script::Cyrillic => 1,
        Script::Greek => 2,
        Script::Arabic => 3,
        Script::Hebrew => 4,
        Script::Devanagari => 5,
        Script::Han => 6,
        Script::Kana => 7,
        Script::Hangul => 8,
        Script::Other => 9,
        Script::Common => 10,
    }
}
const SCRIPTS: [Script; 10] = [
    Script::Latin,
    Script::Cyrillic,
    Script::Greek,
    Script::Arabic,
    Script::Hebrew,
    Script::Devanagari,
    Script::Han,
    Script::Kana,
    Script::Hangul,
    Script::Other,
];

/// Identifies the language of `text`. Never panics; empty or letterless input gives `lang: None`.
pub fn detect(text: &str) -> Detection {
    let mut n = [0usize; 11];
    for c in text.chars() {
        if c.is_alphabetic() {
            n[script_index(script_of(c))] += 1;
        }
    }
    let letters: usize = n[..10].iter().sum();
    let none = |script: Script| Detection {
        lang: None,
        confidence: 0.0,
        reliable: false,
        script,
        letters,
        scores: vec![],
    };
    if letters == 0 {
        return none(Script::Common);
    }
    let (mut best, mut bn) = (Script::Other, 0usize);
    for s in SCRIPTS {
        let k = if s == Script::Han || s == Script::Kana {
            0
        } else {
            n[script_index(s)]
        };
        if k > bn {
            best = s;
            bn = k;
        }
    }
    let cjk = n[script_index(Script::Han)] + n[script_index(Script::Kana)];
    if cjk > bn {
        best = if n[script_index(Script::Kana)] > 0 {
            Script::Kana
        } else {
            Script::Han
        };
        bn = cjk;
    }
    let share = bn as f32 / letters as f32;
    let single = |l: Lang, cap: f32| {
        let conf = (cap * share * (0.55 + 0.09 * (bn.min(5) as f32))).min(cap);
        Detection {
            lang: Some(l),
            confidence: conf,
            reliable: conf >= 0.5 && bn >= 2,
            script: best,
            letters,
            scores: vec![(l, share)],
        }
    };
    match best {
        Script::Greek => single(Lang::El, 0.97),
        Script::Arabic => single(Lang::Ar, 0.9),
        Script::Hebrew => single(Lang::He, 0.97),
        Script::Devanagari => single(Lang::Hi, 0.9),
        Script::Hangul => single(Lang::Ko, 0.97),
        Script::Kana | Script::Han => {
            if n[script_index(Script::Kana)] > 0 && n[script_index(Script::Kana)] * 10 >= cjk {
                single(Lang::Ja, 0.97)
            } else if n[script_index(Script::Hangul)] > cjk {
                single(Lang::Ko, 0.9)
            } else {
                single(Lang::Zh, 0.85)
            }
        }
        Script::Other | Script::Common => none(best),
        Script::Latin | Script::Cyrillic => {
            let words = words_of(text);
            let mut scores: Vec<(Lang, f32)> = vec![];
            for p in profiles().iter().filter(|p| p.lang.script() == best) {
                let mut s = 0f32;
                for w in &words {
                    if let Some(wt) = p.words.get(w) {
                        s += *wt;
                        continue;
                    }
                    let nchars = w.chars().count();
                    if nchars >= 5 && p.suffixes.iter().any(|sf| w.ends_with(sf)) {
                        s += 0.7;
                    }
                }
                for (c, wt) in &p.chars {
                    let k = text.chars().filter(|x| lower_char(*x) == *c).count();
                    s += *wt * (k.min(4) as f32);
                }
                scores.push((p.lang, s));
            }
            scores.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            let top = scores.first().map_or(0.0, |x| x.1);
            if top <= 0.0 {
                return none(best);
            }
            let second = scores.get(1).map_or(0.0, |x| x.1);
            let sum: f32 = scores.iter().map(|x| x.1).sum();
            let r = ((top - second) / (top + 1.0)).clamp(0.0, 1.0);
            let evidence = (1.0 - (-top / 2.5).exp()).clamp(0.0, 1.0);
            let confidence = r * evidence;
            let lang = scores[0].0;
            let tie = (top - second).abs() < 1e-6;
            Detection {
                lang: if tie { None } else { Some(lang) },
                confidence: if tie { 0.0 } else { confidence },
                reliable: !tie && confidence >= 0.5 && top >= 2.0,
                script: best,
                letters,
                scores: scores.iter().take(5).map(|(l, s)| (*l, s / sum)).collect(),
            }
        }
    }
}

/// Convenience: the language if `detect` is reliable.
pub fn detect_lang(text: &str) -> Option<Lang> {
    let d = detect(text);
    if d.reliable { d.lang } else { None }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
    pub lang: Option<Lang>,
    pub confidence: f32,
}

fn script_group(c: char) -> Option<Script> {
    if !c.is_alphabetic() {
        return None;
    }
    Some(match script_of(c) {
        Script::Kana => Script::Han,
        s => s,
    })
}

/// Splits mixed-language text into segments (sentence and script boundaries). Pieces without reliable
/// evidence (names, single words) join the preceding language, or the following one at the start.
pub fn detect_segments(text: &str) -> Vec<Segment> {
    // 1. cut points
    let mut cuts = vec![0usize];
    let mut cur_group: Option<Script> = None;
    let mut after_term = false;
    for (i, c) in text.char_indices() {
        if after_term && !c.is_whitespace() {
            cuts.push(i);
            after_term = false;
            cur_group = None;
        }
        if matches!(
            c,
            '.' | '!' | '?' | '…' | '。' | '！' | '？' | '\n' | ';' | '؟' | '।'
        ) {
            after_term = true;
        }
        if let Some(g) = script_group(c) {
            if let Some(p) = cur_group
                && p != g
                && cuts.last() != Some(&i)
            {
                cuts.push(i);
            }
            cur_group = Some(g);
        }
    }
    cuts.push(text.len());
    cuts.dedup();
    // 2. detect each piece
    struct P {
        s: usize,
        e: usize,
        lang: Option<Lang>,
        conf: f32,
        letters: usize,
    }
    let mut ps: Vec<P> = vec![];
    for w in cuts.windows(2) {
        let (s, e) = (w[0], w[1]);
        if s >= e {
            continue;
        }
        let d = detect(&text[s..e]);
        let ok = d.reliable;
        ps.push(P {
            s,
            e,
            lang: if ok { d.lang } else { None },
            conf: d.confidence,
            letters: d.letters,
        });
    }
    // 3. fill unresolved
    let mut last: Option<Lang> = None;
    for p in ps.iter_mut() {
        if p.lang.is_some() {
            last = p.lang;
        } else if last.is_some() {
            p.lang = last;
            p.conf *= 0.5;
        }
    }
    let mut next: Option<Lang> = None;
    for p in ps.iter_mut().rev() {
        if p.lang.is_some() {
            next = p.lang;
        } else if next.is_some() {
            p.lang = next;
            p.conf *= 0.5;
        }
    }
    // 4. merge
    let mut out: Vec<Segment> = vec![];
    let mut wsum: Vec<(f32, f32)> = vec![];
    for p in ps {
        let w = p.letters.max(1) as f32;
        if let (Some(l), Some(prev)) = (out.last_mut(), wsum.last_mut())
            && l.lang == p.lang
        {
            l.end = p.e;
            prev.0 += p.conf * w;
            prev.1 += w;
            l.confidence = prev.0 / prev.1;
            continue;
        }
        out.push(Segment {
            start: p.s,
            end: p.e,
            lang: p.lang,
            confidence: p.conf,
        });
        wsum.push((p.conf * w, w));
    }
    out
}

// ================================================================================ answer-language lock

/// Letters that are characteristic of `l` (used for the soft logit bias).
pub fn marker_chars(l: Lang) -> &'static str {
    match l {
        Lang::Ro => "ăâîșțĂÂÎȘȚ",
        Lang::Fr => "àâçéèêëîïôùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŸŒÆ",
        Lang::It => "àèéìòùÀÈÉÌÒÙ",
        Lang::Es => "áéíóúñüÁÉÍÓÚÑÜ",
        Lang::Pt => "ãõáâàçéêíóôúÃÕÁÂÀÇÉÊÍÓÔÚ",
        Lang::De => "äöüßÄÖÜ",
        Lang::Hu => "áéíóöőúüűÁÉÍÓÖŐÚÜŰ",
        Lang::Pl => "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
        Lang::Cs => "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ",
        Lang::Sk => "áäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ",
        Lang::Sv => "åäöÅÄÖ",
        Lang::Da | Lang::No => "æøåÆØÅ",
        Lang::Fi => "äöÄÖ",
        Lang::Tr => "çğıöşüÇĞİÖŞÜ",
        Lang::Nl => "ëïéËÏÉ",
        _ => "",
    }
}

/// Characters that should *not* appear in an answer in `target` (letters of other product languages).
fn foreign_chars(target: Lang) -> String {
    let own = marker_chars(target);
    let mut s = String::new();
    for l in Lang::PRODUCT {
        if l != target {
            for c in marker_chars(l).chars() {
                if !own.contains(c) && c != 'é' && c != 'É' && !s.contains(c) {
                    s.push(c);
                }
            }
        }
    }
    s
}

#[derive(Clone, Copy, Default, Debug)]
struct TokInfo {
    scripts: u16,
    marks: u8,
    cedilla: u8,
    foreign: u8,
}

fn partial_script(lead: u8, second: Option<u8>) -> Script {
    let s2 = second.unwrap_or(0);
    match lead {
        0xCD => {
            if second.is_some_and(|s| s >= 0xB0) {
                Script::Greek
            } else {
                Script::Common
            }
        }
        0xCE | 0xCF => Script::Greek,
        0xD0..=0xD3 => Script::Cyrillic,
        0xD4 => {
            if second.is_some_and(|s| s >= 0xB0) {
                Script::Other
            } else {
                Script::Cyrillic
            }
        }
        0xD5 => Script::Other,
        0xD6 => {
            if s2 >= 0x90 {
                Script::Hebrew
            } else {
                Script::Other
            }
        }
        0xD7 => Script::Hebrew,
        0xD8..=0xDB => Script::Arabic,
        0xDC..=0xDF => Script::Other,
        0xE0 => {
            if matches!(s2, 0xA4 | 0xA5) {
                Script::Devanagari
            } else {
                Script::Other
            }
        }
        0xE1 => match s2 {
            0x84..=0x87 => Script::Hangul,
            0xBC..=0xBF => Script::Greek,
            _ => Script::Common,
        },
        0xE3 => match s2 {
            0x81..=0x83 => Script::Kana,
            0x84..=0x86 => Script::Hangul,
            0x80 => Script::Common,
            _ => Script::Han,
        },
        0xE4..=0xE9 => Script::Han,
        0xEA => match second {
            Some(0x99) => Script::Cyrillic,
            Some(s) if s < 0xB0 => Script::Common,
            _ => Script::Hangul,
        },
        0xEB..=0xED => Script::Hangul,
        0xF0 => {
            if (0xA0..=0xAF).contains(&s2) {
                Script::Han
            } else {
                Script::Common
            }
        }
        _ => Script::Common,
    }
}

fn classify_token(b: &[u8], marks: &str, foreign: &str) -> TokInfo {
    let mut t = TokInfo::default();
    for chunk in b.utf8_chunks() {
        for c in chunk.valid().chars() {
            if c.is_alphabetic() {
                t.scripts |= script_of(c).bit();
            }
            if marks.contains(c) {
                t.marks = t.marks.saturating_add(1);
            }
            if foreign.contains(c) {
                t.foreign = t.foreign.saturating_add(1);
            }
            if matches!(c, 'ş' | 'ţ' | 'Ş' | 'Ţ') {
                t.cedilla = t.cedilla.saturating_add(1);
            }
        }
        let inv = chunk.invalid();
        if let Some(&lead) = inv.first()
            && lead >= 0xC2
        {
            t.scripts |= partial_script(lead, inv.get(1).copied()).bit();
        }
    }
    t
}

/// Per-token script and diacritic facts for one target language, built once per vocabulary.
/// Use [`LangLock::mask`] as a hard filter and [`LangLock::bias`] as a soft push.
#[derive(Clone, Debug)]
pub struct LangLock {
    pub target: Lang,
    allowed: u16,
    deny_cedilla: bool,
    info: Vec<TokInfo>,
}

impl LangLock {
    /// `bytes[id]` is the byte string of token `id` (empty for special tokens).
    pub fn from_token_bytes(bytes: &[Vec<u8>], target: Lang) -> LangLock {
        Self::build(bytes.len(), |i| &bytes[i], target)
    }
    /// Builds from a loaded tokenizer (byte-level BPE, SentencePiece and `<0xNN>` byte tokens are decoded).
    pub fn from_tokenizer(tok: &crate::tokenizer::Tokenizer, target: Lang) -> LangLock {
        let trie = crate::grammar::TokenTrie::from_tokenizer(tok);
        Self::build(trie.vocab_size(), |i| trie.token_bytes(i as u32), target)
    }
    fn build<'a>(n: usize, f: impl Fn(usize) -> &'a [u8], target: Lang) -> LangLock {
        let marks = marker_chars(target);
        let foreign = foreign_chars(target);
        let info = (0..n)
            .map(|i| classify_token(f(i), marks, &foreign))
            .collect();
        // Latin names, code and units are always acceptable inside a non-Latin answer.
        let allowed = target.scripts() | Script::Latin.bit();
        LangLock {
            target,
            allowed,
            deny_cedilla: false,
            info,
        }
    }
    /// Latin-script targets: other scripts are forbidden. Non-Latin targets: Latin stays allowed.
    /// Call this to allow one more script (e.g. Han inside a Japanese or Korean answer is already allowed).
    pub fn allow(mut self, s: Script) -> Self {
        self.allowed |= s.bit();
        self
    }
    /// Forbid Latin letters too (only sensible for non-Latin targets).
    pub fn forbid_latin(mut self) -> Self {
        if !self.target.script().eq(&Script::Latin) {
            self.allowed &= !Script::Latin.bit();
        }
        self
    }
    /// For Romanian: also forbid the cedilla forms ş ţ in the mask.
    pub fn deny_ro_cedilla(mut self, on: bool) -> Self {
        self.deny_cedilla = on;
        self
    }
    /// Rebuilds `allowed` for strict Latin-only answers (default for Latin targets already).
    pub fn vocab_size(&self) -> usize {
        self.info.len()
    }
    pub fn token_allowed(&self, id: u32) -> bool {
        match self.info.get(id as usize) {
            Some(t) => {
                t.scripts & !self.allowed == 0
                    && !(self.deny_cedilla && self.target == Lang::Ro && t.cedilla > 0)
            }
            None => false,
        }
    }
    /// Hard mask for [`crate::sampling`]: false for tokens with letters from a forbidden script.
    /// Special tokens (no bytes) and punctuation/digits/emoji stay allowed.
    pub fn mask(&self) -> TokenMask {
        TokenMask::from_fn(self.info.len(), |id| self.token_allowed(id))
    }
    pub fn forbidden_count(&self) -> usize {
        (0..self.info.len() as u32)
            .filter(|&i| !self.token_allowed(i))
            .count()
    }
    /// Soft bias: `+boost` for tokens that carry the target language's letters, `-penalty` for letters
    /// of other product languages, and for Romanian a stronger push away from cedilla forms.
    pub fn bias_with(&self, boost: f32, foreign_penalty: f32) -> Vec<(u32, f32)> {
        let mut v = vec![];
        for (id, t) in self.info.iter().enumerate() {
            if t.scripts & !self.allowed != 0 {
                continue;
            }
            let mut b = 0f32;
            if t.marks > 0 {
                b += boost;
            }
            if t.foreign > 0 {
                b -= foreign_penalty;
            }
            if self.target == Lang::Ro && t.cedilla > 0 {
                b -= boost * 2.0;
            }
            if b != 0.0 {
                v.push((id as u32, b));
            }
        }
        v
    }
    pub fn bias(&self, boost: f32) -> Vec<(u32, f32)> {
        self.bias_with(boost, boost)
    }
}

// ---------------------------------------------------------------------------- drift watcher

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DriftKind {
    Script(Script),
    Language(Lang),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drift {
    pub kind: DriftKind,
    pub at_char: usize,
    pub confidence: f32,
}

/// Watches streamed output and reports when it leaves the target language. Code fences are ignored.
/// Language drift needs two consecutive failed checks (so a quoted foreign phrase is not an alarm);
/// off-script drift fires as soon as a few foreign-script letters are in the window.
#[derive(Clone, Debug)]
pub struct LangGuard {
    target: Lang,
    window: usize,
    min_letters: usize,
    off_limit: usize,
    check_every: usize,
    buf: String,
    buf_chars: usize,
    seen: usize,
    since_check: usize,
    ticks: u8,
    fence: bool,
    pending: Vec<u8>,
    strikes: u8,
    active: Option<Drift>,
}

impl LangGuard {
    pub fn new(target: Lang) -> LangGuard {
        LangGuard {
            target,
            window: 240,
            min_letters: 40,
            off_limit: 6,
            check_every: 24,
            buf: String::new(),
            buf_chars: 0,
            seen: 0,
            since_check: 0,
            ticks: 0,
            fence: false,
            pending: vec![],
            strikes: 0,
            active: None,
        }
    }
    pub fn with_window(mut self, chars: usize) -> Self {
        self.window = chars.max(40);
        self
    }
    pub fn with_off_script_limit(mut self, n: usize) -> Self {
        self.off_limit = n.max(1);
        self
    }
    pub fn reset(&mut self) {
        *self = LangGuard::new(self.target)
            .with_window(self.window)
            .with_off_script_limit(self.off_limit);
    }
    pub fn chars_seen(&self) -> usize {
        self.seen
    }
    pub fn current(&self) -> Option<Drift> {
        self.active
    }

    /// Feeds raw token bytes; incomplete UTF-8 at the end is carried to the next call.
    pub fn feed_bytes(&mut self, b: &[u8]) -> Option<Drift> {
        self.pending.extend_from_slice(b);
        let pending = std::mem::take(&mut self.pending);
        let mut text = String::new();
        let mut tail: Vec<u8> = vec![];
        let mut chunks = pending.utf8_chunks().peekable();
        while let Some(ch) = chunks.next() {
            text.push_str(ch.valid());
            if chunks.peek().is_none() && !ch.invalid().is_empty() && ch.invalid().len() < 4 {
                tail = ch.invalid().to_vec();
            }
        }
        self.pending = tail;
        self.feed(&text)
    }

    pub fn feed(&mut self, s: &str) -> Option<Drift> {
        for c in s.chars() {
            if c == '`' {
                self.ticks += 1;
                if self.ticks == 3 {
                    self.fence = !self.fence;
                    self.ticks = 0;
                }
                continue;
            }
            self.ticks = 0;
            if self.fence {
                continue;
            }
            self.buf.push(c);
            self.buf_chars += 1;
            self.seen += 1;
            self.since_check += 1;
        }
        if self.buf_chars > self.window * 2 {
            let drop = self.buf_chars - self.window;
            let cut = self
                .buf
                .char_indices()
                .nth(drop)
                .map_or(self.buf.len(), |(i, _)| i);
            self.buf.drain(..cut);
            self.buf_chars -= drop;
        }
        if self.since_check < self.check_every {
            return None;
        }
        self.since_check = 0;
        self.check()
    }

    fn check(&mut self) -> Option<Drift> {
        let tail: String = {
            let skip = self.buf_chars.saturating_sub(self.window);
            self.buf.chars().skip(skip).collect()
        };
        let allowed = self.target.scripts() | Script::Latin.bit();
        let allowed = if self.target.script() == Script::Latin {
            Script::Latin.bit()
        } else {
            allowed
        };
        let mut off: HashMap<u16, usize> = HashMap::new();
        let mut letters = 0usize;
        for c in tail.chars() {
            if !c.is_alphabetic() {
                continue;
            }
            letters += 1;
            let b = script_of(c).bit();
            if b != 0 && b & allowed == 0 {
                *off.entry(b).or_default() += 1;
            }
        }
        let off_total: usize = off.values().sum();
        let found = if off_total >= self.off_limit {
            let (&bit, _) = off.iter().max_by_key(|(_, n)| **n).unwrap();
            let sc = SCRIPTS
                .iter()
                .copied()
                .find(|s| s.bit() == bit)
                .unwrap_or(Script::Other);
            Some(Drift {
                kind: DriftKind::Script(sc),
                at_char: self.seen,
                confidence: (off_total as f32 / letters.max(1) as f32).min(1.0),
            })
        } else if letters >= self.min_letters {
            let d = detect(&tail);
            match d.lang {
                Some(l)
                    if d.reliable
                        && l != self.target
                        && d.prob(self.target) < 0.25
                        && d.confidence >= 0.6 =>
                {
                    self.strikes += 1;
                    if self.strikes >= 2 {
                        Some(Drift {
                            kind: DriftKind::Language(l),
                            at_char: self.seen,
                            confidence: d.confidence,
                        })
                    } else {
                        None
                    }
                }
                _ => {
                    self.strikes = 0;
                    None
                }
            }
        } else {
            None
        };
        match found {
            Some(dr) => {
                if self.active.is_none() {
                    self.active = Some(dr);
                    return Some(dr);
                }
                self.active = Some(dr);
                None
            }
            None => {
                if letters >= self.min_letters.min(12) && self.strikes == 0 {
                    self.active = None;
                }
                None
            }
        }
    }
}

// ================================================================================ Romanian formatting

#[allow(clippy::manual_is_multiple_of)]
fn group_thousands(digits: &str) -> String {
    let b = digits.as_bytes();
    let mut o = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in b.iter().enumerate() {
        if i > 0 && (b.len() - i) % 3 == 0 {
            o.push('.');
        }
        o.push(*c as char);
    }
    o
}

/// `1234567` → `1.234.567`.
pub fn format_int_ro(n: i64) -> String {
    let s = n.unsigned_abs().to_string();
    let g = group_thousands(&s);
    if n < 0 { format!("-{g}") } else { g }
}

/// `1234.5, 2` → `1.234,50`. Non-finite values are returned as `NaN`, `∞`, `-∞`.
pub fn format_decimal_ro(x: f64, decimals: usize) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-∞".into() } else { "∞".into() };
    }
    let s = format!("{:.*}", decimals.min(15), x.abs());
    let (int, frac) = match s.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (s.as_str(), None),
    };
    let mut o = String::new();
    if x.is_sign_negative() && s.bytes().any(|b| b != b'0' && b != b'.') {
        o.push('-');
    }
    o.push_str(&group_thousands(int));
    if let Some(f) = frac {
        o.push(',');
        o.push_str(f);
    }
    o
}

/// `12.5` → `12,5%`.
pub fn format_percent_ro(x: f64, decimals: usize) -> String {
    format!("{}%", format_decimal_ro(x, decimals))
}

/// `n` is in the smallest unit (bani, euro cents). `RON`/`lei` → `1.234,56 lei`; whole amounts use
/// the Romanian singular and "de" rules (`1 leu`, `2 lei`, `20 de lei`, `0 lei`). `EUR` and `USD` likewise.
/// Other currency codes are printed as `1.234,56 CODE`.
pub fn format_money_minor_ro(minor: i64, currency: &str) -> String {
    let neg = minor < 0;
    let m = minor.unsigned_abs();
    let (whole, cents) = (m / 100, m % 100);
    let sign = if neg { "-" } else { "" };
    let code = currency.trim().to_ascii_uppercase();
    let (one, many, has_de) = match code.as_str() {
        "RON" | "LEI" | "LEU" => ("leu", "lei", true),
        "EUR" | "EURO" | "€" => ("euro", "euro", true),
        "USD" | "$" => ("dolar", "dolari", true),
        _ => ("", "", false),
    };
    if !has_de {
        return format!(
            "{sign}{},{:02} {}",
            group_thousands(&whole.to_string()),
            cents,
            if code.is_empty() { "RON" } else { &code }
        );
    }
    if cents != 0 {
        return format!(
            "{sign}{},{:02} {}",
            group_thousands(&whole.to_string()),
            cents,
            many
        );
    }
    let n = group_thousands(&whole.to_string());
    if whole == 1 {
        return format!("{sign}{n} {one}");
    }
    let r = whole % 100;
    if whole != 0 && (r == 0 || r >= 20) {
        format!("{sign}{n} de {many}")
    } else {
        format!("{sign}{n} {many}")
    }
}

/// Rounds to two decimals (half away from zero) and formats; see [`format_money_minor_ro`].
pub fn format_money_ro(amount: f64, currency: &str) -> String {
    if !amount.is_finite() {
        return format_decimal_ro(amount, 2);
    }
    format_money_minor_ro((amount * 100.0).round() as i64, currency)
}

/// Parses Romanian numbers: `1.234,56`, `-12,5`, `1 234,5`, `3,14`, `1234`. A dot is a thousands
/// separator and must be followed by exactly three digits, so `1.234` is 1234 (not 1.234).
pub fn parse_number_ro(s: &str) -> Option<f64> {
    let t = s.trim();
    let (neg, t) = match t.strip_prefix('-').or_else(|| t.strip_prefix('−')) {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if t.is_empty() {
        return None;
    }
    let (int, frac) = match t.split_once(',') {
        Some((i, f)) => (i, Some(f)),
        None => (t, None),
    };
    if frac.is_some_and(|f| f.contains(',') || !f.bytes().all(|b| b.is_ascii_digit())) {
        return None;
    }
    let groups: Vec<&str> = int.split(['.', ' ', '\u{00A0}', '\u{202F}']).collect();
    if groups
        .iter()
        .any(|g| g.is_empty() || !g.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    if groups.len() > 1 && (groups[0].len() > 3 || groups[1..].iter().any(|g| g.len() != 3)) {
        return None;
    }
    let joined: String = groups.concat();
    let v: f64 = format!(
        "{}{}{}",
        joined,
        if frac.is_some() { "." } else { "" },
        frac.unwrap_or("")
    )
    .parse()
    .ok()?;
    Some(if neg { -v } else { v })
}

pub const MONTHS_RO: [&str; 12] = [
    "ianuarie",
    "februarie",
    "martie",
    "aprilie",
    "mai",
    "iunie",
    "iulie",
    "august",
    "septembrie",
    "octombrie",
    "noiembrie",
    "decembrie",
];
/// Index 0 = Sunday, matching [`weekday`].
pub const WEEKDAYS_RO: [&str; 7] = [
    "duminică",
    "luni",
    "marți",
    "miercuri",
    "joi",
    "vineri",
    "sâmbătă",
];

pub fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
pub fn days_in_month(y: i32, m: u8) -> u8 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}
/// Day of week for a valid Gregorian date: 0 = Sunday … 6 = Saturday.
pub fn weekday(y: i32, m: u8, d: u8) -> u8 {
    const T: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let yy = if m < 3 { y - 1 } else { y };
    ((yy + yy.div_euclid(4) - yy.div_euclid(100)
        + yy.div_euclid(400)
        + T[(m as usize - 1) % 12]
        + d as i32)
        .rem_euclid(7)) as u8
}

/// `(2026, 11, 24)` → `24 noiembrie 2026`, or `marți, 24 noiembrie 2026` with `with_weekday`. `None` if invalid.
pub fn format_date_ro(y: i32, m: u8, d: u8, with_weekday: bool) -> Option<String> {
    if !(1..=12).contains(&m) || d == 0 || d > days_in_month(y, m) {
        return None;
    }
    let base = format!("{} {} {}", d, MONTHS_RO[m as usize - 1], y);
    Some(if with_weekday {
        format!("{}, {}", WEEKDAYS_RO[weekday(y, m, d) as usize], base)
    } else {
        base
    })
}

fn month_from_word(w: &str) -> Option<u8> {
    let f = strip_diacritics(w).to_lowercase();
    let f = f.trim_end_matches('.');
    if f.len() < 3 {
        return None;
    }
    MONTHS_RO
        .iter()
        .position(|m| m.starts_with(f) && (f.len() >= 3))
        .map(|i| i as u8 + 1)
        .or(if f == "sept" { Some(9) } else { None })
}

/// Parses `24 noiembrie 2026`, `marți, 24 nov. 2026`, `24.11.2026`, `24/11/2026`, `2026-11-24`
/// (diacritics and case ignored). Two-digit years are rejected as ambiguous.
pub fn parse_date_ro(s: &str) -> Option<(i32, u8, u8)> {
    let t = s.trim();
    let valid = |y: i32, m: u8, d: u8| {
        if (1..=12).contains(&m) && d >= 1 && d <= days_in_month(y, m) {
            Some((y, m, d))
        } else {
            None
        }
    };
    let num: Vec<&str> = t.split(['.', '/', '-']).map(str::trim).collect();
    if num.len() == 3
        && num
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
    {
        let n: Vec<i32> = num.iter().map(|p| p.parse().unwrap_or(-1)).collect();
        if num[0].len() == 4 {
            return valid(n[0], n[1] as u8, n[2] as u8);
        }
        if num[2].len() == 4 {
            return valid(n[2], n[1] as u8, n[0] as u8);
        }
        return None;
    }
    let toks: Vec<&str> = t
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|x| !x.is_empty())
        .collect();
    let (mut d, mut m, mut y) = (None, None, None);
    for tok in toks {
        if tok.bytes().all(|b| b.is_ascii_digit()) {
            if tok.len() == 4 {
                y = tok.parse::<i32>().ok();
            } else if tok.len() <= 2 {
                d = tok.parse::<u8>().ok();
            } else {
                return None;
            }
        } else if WEEKDAYS_RO
            .iter()
            .any(|w| strip_diacritics(w) == strip_diacritics(&tok.to_lowercase()))
        {
            continue;
        } else {
            m = Some(month_from_word(tok)?);
        }
    }
    valid(y?, m?, d?)
}

// ================================================================================ Romanian diacritics restoration

/// Forms whose diacritic-free spelling is not another common word. Keys are derived by stripping.
const LEX_FORMS: &str = "
și în îl îi îți îmi își mă ți aș ești sunteți ați aveți faceți puteți vreți dați dăm mergeți spuneți veniți ieșiți scrieți citiți folosiți încercați învățați înțelegeți găsiți căutați rugăm mulțumim mulțumesc mulțumit mulțumire mulțumiri bună
după către fără între până încă însă deși așa așadar totuși oricând oricât câte câți câțiva câteva câteodată cândva niciodată odată altădată astăzi mâine poimâine alaltăieri curând târziu încet nicăieri lângă înainte înaintea înapoi înăuntru împreună împotriva împotrivă înspre încotro încolo întotdeauna întâi întâia întrucât încât fiindcă datorită măcar bineînțeles firește
cărui căror cărei acești aceștia aceleași același aceeași aceiași când cât câtă tău noastră voastră dumneavoastră ție într dintr
începe începem începeți începând început începutul încep încerc încerci încearcă încerca încercăm încercat încercare încercări înțeleg înțelegi înțelege înțelegem înțeles înțelegere înțelegerea
învăț învățăm învățat învățare învățarea învățământ învățământul învățător găsesc găsește găsim găsit găsi găsire știu știi știe știm știți știut știință știința științific științifică științifice știri știre
făcut făcută făcând făcea făceam făceau mergând spunând rămân rămâne rămas rămânem rămâneți rămânând plecăm plecați ajutăm ajutați cumpăr cumpărat cumpărăm cumpărați cumpărături
vând vândut vânzare vânzări vânzător vânzători plătesc plătește plătit plătim plătiți plătire plăți primește primiți băut mănânc mănâncă mâncare mâncarea mâncat mâncăm mâncați mâncăruri
muncește locuiește locuință locuințe lucrează lucrăm lucrați lucrări lucrările oprește opriți închide închid închidem închis închidere închiderea închiși pornește
folosește folosință utilizează funcționează funcționare funcționarea funcționalitate funcționalități funcționalitățile funcție funcții funcțiile funcțional funcționale
descărcare descărcări descărcarea descărcat descărcăm descărcați încărcare încărcări încărcat încărcăm încărcarea încărcător salvează selectează selecție șterge șters ștergere ștergerea ștergem ștergeți
modificări adăugare adăugat adăugări adăugarea adăugăm adăugați setări setările configurări opțiune opțiuni opțiunea opțiunile opțional opționale
aplicație aplicații aplicația aplicațiile actualizări actualizările actualizează fișier fișiere fișierul fișierele fișierelor rețea rețele rețeaua rețelele rețelelor conectează
confidențial confidențiale confidențialitate confidențialitatea licențe licențele condiții condițiile condiție condiția generează răspuns răspunsul răspunsuri răspunsurile răspunde răspund răspunzi răspundem
întrebare întrebări întrebarea întrebările întreabă întreba întreb întrebat întrebăm întrebați căutare căutări căutarea căutat căutăm înălțime înălțimea râuri
douăzeci nouăzeci șase șapte șaizeci șaptezeci șaptesprezece șaisprezece nouăsprezece jumătate jumătăți jumătatea marți
părinți părinte părinții frați frații soție soții soț soția soțul soțului căsătorit căsătorie căsătoria căsătoriei bărbat bărbați bărbatul băiat băieți băiatul băiatului tânăr tânărul bătrân bătrâni bătrânul
școli școlar studenți studenții lecție lecții lecția lecțiile educație educația cunoștințe cunoștințele cunoaște cunoaștere cunoașterea cărți cărțile cărții povești poveștile cântec cântece cântăreț câștig câștigat
angajați preț prețuri cumpărare
oraș orașe orașul județ județe județul număr numărul poștal călătorie călătorii călător stații stație rezervări
munți câmp câmpie pădurea însorit sănătate sănătos pește roșii castraveți căpșuni lămâie prăjitură prăjituri zahăr oțet
frumoși urât urâtă ușor ușoară greșit greșeală greșeli greșesc greșește soluție soluții rezolvă explicație explicații explică informație informații informațiile informează
performanțe eficiență optimizări îmbunătățire îmbunătățiri îmbunătățit îmbunătăți calități cantități capacități posibilități necesități activități universități facilități societăți autorități oportunități identități
organizație organizații situație situații relație relații operație operații națiune națiuni poziție poziții tradiție tradiții producție producții instituție instituții acțiune acțiuni secțiune secțiuni direcție direcții protecție protejează
excepție excepții atenție intenție intenții emoție emoții documentație documentația ediție ediții colecție colecții contribuție contribuții distribuție distribuții comunicații administrație
România român români românii românesc București Iași Brașov Timișoara Constanța Ploiești Galați Bacău Chișinău Târgu Mureș Pitești Buzău Botoșani Focșani Târgoviște Călărași Reșița Zalău Bistrița Dunărea Carpați Franța Elveția
Ștefan Ștefania Cătălin Cătălina Răzvan Ionuț Mădălin Mădălina Vlăduț Dănuț Răducu
părți dacă
";

/// Feminine nouns and verbs with an inner diacritic and a final `ă`: both the `-ă` form and the
/// `-a` form (definite article / infinitive) are registered, so the inner diacritics are restored
/// and only the last letter stays ambiguous.
const LEX_PAIRS_A: &str = "
țară viață școală mână față dimineață ceață gheață învățătură băutură săptămână primăvară zăpadă ușă mașină grădină țigară cămașă ceașcă lingură furculiță brânză friptură
sâmbătă înghețată făină domnișoară amiază bătrână cântăreață închisă înaltă cumpără învață câștigă
";

/// Ambiguous groups: stripped spelling is also a real word. Preferred form first.
const LEX_GROUPS: &str = "
să|sa că|ca vă|va altă|alta nouă|noua două|doua dată|data stă|sta
față|fată|fața|fata rău|râu română|romana stânga|stângă dreaptă|dreapta
caută|căuta adaugă|adăuga continuă|continua licență|licența politică|politica parolă|parola
inteligență|inteligența muzică|muzica joacă|joaca placă|placa grafică|grafica
";

/// Plain words (no diacritics in the correct spelling) used only for the coverage estimate.
const LEX_PLAIN: &str = "
de la cu pe un o nu ce se din al ale dar sau mai am ai are au fost fi foarte bine cum unde cine care este sunt era vor eu tu el ea noi voi ei ele lui lor mea meu mele ne te mi pentru prin spre sub peste cel cea cei cele unei unui unor acest
acum aici acolo atunci azi ieri tot nimic ceva nimeni oricine nici nicio niciun fiecare orice doar numai chiar deci vreau vrei vrea vrem pot poate putem trebuie face fac faci facem merg mergem spun spune dau da poftim scuze salut
ziua seara noaptea bun bune buni bani timp om oameni lucru lucruri loc locuri parte unde cand cum mult multe mare mari mici nou noi vechi calculator model modele program date sistem
";

struct Lex {
    map: HashMap<String, Vec<String>>,
}

fn lex() -> &'static Lex {
    static L: OnceLock<Lex> = OnceLock::new();
    L.get_or_init(|| {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for f in LEX_FORMS.split_whitespace() {
            let f = f.trim_end_matches('?').to_lowercase();
            map.entry(strip_diacritics(&f))
                .or_insert_with(|| vec![f.clone()]);
        }
        for f in LEX_PAIRS_A.split_whitespace() {
            let f = f.to_lowercase();
            let a: String = {
                let mut c: Vec<char> = f.chars().collect();
                if let Some(l) = c.last_mut() {
                    *l = 'a';
                }
                c.into_iter().collect()
            };
            map.insert(strip_diacritics(&f), vec![f.clone(), a]);
        }
        for g in LEX_GROUPS.split_whitespace() {
            let forms: Vec<String> = g.split('|').map(|x| x.to_lowercase()).collect();
            map.insert(strip_diacritics(&forms[0]), forms);
        }
        for w in LEX_PLAIN.split_whitespace() {
            map.entry(w.to_string())
                .or_insert_with(|| vec![w.to_string()]);
        }
        Lex { map }
    })
}

/// Number of lexicon entries (keys), for documentation and tests.
pub fn ro_lexicon_size() -> usize {
    lex().map.len()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreMode {
    /// Ambiguous words keep their spelling except for letters all readings agree on; they are flagged.
    Conservative,
    /// Ambiguous words take the most frequent reading (still flagged).
    Guess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoteKind {
    Lexicon,
    Context,
    Heuristic,
    Ambiguous,
}

/// One restored (or flagged) word. Offsets are byte offsets into [`Restored::text`].
#[derive(Clone, Debug, PartialEq)]
pub struct RestoreNote {
    pub start: usize,
    pub end: usize,
    pub original: String,
    pub restored: String,
    pub kind: NoteKind,
}

#[derive(Clone, Debug, Default)]
pub struct Restored {
    pub text: String,
    pub notes: Vec<RestoreNote>,
    /// Alphabetic words seen, and how many of those were recognised (lexicon, plain list, or already accented).
    pub words: usize,
    pub known: usize,
    pub changed: usize,
}
impl Restored {
    pub fn coverage(&self) -> f32 {
        if self.words == 0 {
            0.0
        } else {
            self.known as f32 / self.words as f32
        }
    }
    /// True when the lexicon recognised under 60% of the words (specialised text, or not Romanian at all).
    pub fn low_confidence(&self) -> bool {
        self.words >= 3 && self.coverage() < 0.6
    }
    pub fn flagged(&self) -> impl Iterator<Item = &RestoreNote> {
        self.notes
            .iter()
            .filter(|n| matches!(n.kind, NoteKind::Ambiguous | NoteKind::Heuristic))
    }
}

const HEUR: &[(&str, &str)] = &[
    ("atiilor", "ațiilor"),
    ("atiile", "ațiile"),
    ("atiei", "ației"),
    ("atii", "ații"),
    ("atie", "ație"),
    ("itii", "iții"),
    ("itie", "iție"),
    ("tiunea", "țiunea"),
    ("tiunii", "țiunii"),
    ("tiunile", "țiunile"),
    ("tiuni", "țiuni"),
    ("tiune", "țiune"),
    ("tatii", "tății"),
    ("tati", "tăți"),
    ("eaza", "ează"),
    ("easca", "ească"),
];

const SA_PREV: &[&str] = &[
    "ca", "poate", "vrut", "putut", "reusit", "trebuit", "invat", "invata", "decis", "uitat",
    "timp", "gata", "vreau", "vrei", "vrea", "vrem", "vreti", "pot", "poti", "putem", "puteti",
    "trebuie", "incerc", "incerca", "incearca", "rog", "las", "lasa", "nevoie", "fara", "pana",
    "inainte", "loc", "doresc", "dorim", "ar", "fie",
];
const SA_NEXT: &[&str] = &[
    "nu", "fie", "fii", "fiu", "se", "te", "ne", "va", "imi", "iti", "isi", "il", "le", "mai",
];
const CA_PREV: &[&str] = &[
    "stiu", "stii", "stie", "stim", "stiti", "spun", "spui", "spune", "spunem", "zic", "zice",
    "cred", "crezi", "crede", "vad", "vezi", "vede", "inteleg", "intelegi", "aud", "sper", "speri",
    "nu", "pentru", "banuiesc", "simt", "observ", "spus", "zis", "crezut",
];
const CA_NEXT: &[&str] = &[
    "nu", "am", "ai", "are", "avem", "aveti", "au", "este", "e", "sunt", "esti", "suntem",
    "sunteti", "a", "voi", "vei", "vom",
];
const VA_YES: &[&str] = &[
    "rog",
    "multumesc",
    "spun",
    "trimit",
    "doresc",
    "pot",
    "ofer",
    "recomand",
    "felicit",
    "urez",
    "astept",
    "intreb",
    "anunt",
    "informez",
    "amintesc",
    "propun",
    "sugerez",
    "salut",
    "cer",
    "ajut",
    "explic",
    "arat",
    "multumim",
    "rugam",
    "asteptam",
    "uram",
    "trimitem",
    "oferim",
    "anuntam",
    "informam",
    "invit",
    "invitam",
];
const VA_NO: &[&str] = &[
    "fi",
    "avea",
    "face",
    "putea",
    "merge",
    "trebui",
    "veni",
    "ajunge",
    "dura",
    "costa",
    "primi",
    "deveni",
    "ramane",
    "functiona",
];

fn context_pick<'a>(
    key: &str,
    forms: &'a [String],
    prev: Option<&str>,
    next: Option<&str>,
) -> Option<&'a str> {
    let has = |l: &[&str], w: Option<&str>| w.is_some_and(|x| l.contains(&x));
    let find = |f: &str| forms.iter().find(|x| x.as_str() == f).map(String::as_str);
    match key {
        "sa" if has(SA_PREV, prev) || has(SA_NEXT, next) => find("să"),
        "ca" if next == Some("sa") => find("ca"),
        "ca" if has(CA_PREV, prev) || has(CA_NEXT, next) => find("că"),
        "va" if has(VA_YES, next) => find("vă"),
        "va" if has(VA_NO, next) => find("va"),
        _ => {
            // feminine -ă / -a pair: after the indefinite article "o" the -ă form is the one
            if prev == Some("o")
                && forms.len() == 2
                && forms[0].ends_with('ă')
                && forms[1].ends_with('a')
            {
                return Some(forms[0].as_str());
            }
            None
        }
    }
}

fn match_case(orig: &str, restored: &str) -> String {
    let letters: Vec<char> = orig.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase()) {
        return restored.to_uppercase();
    }
    if letters.first().is_some_and(|c| c.is_uppercase()) {
        let mut it = restored.chars();
        return match it.next() {
            Some(f) => f.to_uppercase().chain(it).collect(),
            None => String::new(),
        };
    }
    restored.to_string()
}

const RO_LETTERS: &str = "ăâîșțşţĂÂÎȘȚŞŢ";

/// Restores Romanian diacritics in text typed without them (`Buna ziua, ce mai faceti?` →
/// `Bună ziua, ce mai faceți?`). Words already carrying diacritics, URLs, e-mails, paths, code
/// (backticks) and words glued to digits or underscores are left alone. See the module docs for
/// coverage and the accuracy limits; check [`Restored::low_confidence`] and [`Restored::flagged`].
pub fn restore_ro(text: &str, mode: RestoreMode) -> Restored {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    // protected: whitespace-delimited chunks that look like URLs, e-mails or paths; and backtick spans
    let mut prot = vec![false; chars.len()];
    let mut i = 0;
    while i < chars.len() {
        if chars[i].1.is_whitespace() {
            i += 1;
            continue;
        }
        let s = i;
        while i < chars.len() && !chars[i].1.is_whitespace() {
            i += 1;
        }
        let chunk: String = chars[s..i].iter().map(|x| x.1).collect();
        if chunk.contains("://")
            || chunk.contains('@')
            || chunk.starts_with("www.")
            || chunk.contains('\\')
            || chunk.contains('_')
            || chunk.matches('/').count() >= 2
        {
            for p in prot.iter_mut().take(i).skip(s) {
                *p = true;
            }
        }
    }
    let mut in_code = false;
    for (k, (_, c)) in chars.iter().enumerate() {
        if *c == '`' {
            in_code = !in_code;
            prot[k] = true;
        } else if in_code {
            prot[k] = true;
        }
    }
    // runs of letters
    struct Run {
        a: usize,
        b: usize,
        word: String,
        skip: bool,
    }
    let mut runs: Vec<Run> = vec![];
    let mut k = 0;
    while k < chars.len() {
        if !chars[k].1.is_alphabetic() {
            k += 1;
            continue;
        }
        let s = k;
        while k < chars.len() && chars[k].1.is_alphabetic() {
            k += 1;
        }
        let word: String = chars[s..k].iter().map(|x| x.1).collect();
        let glued = (s > 0 && (chars[s - 1].1.is_ascii_digit() || chars[s - 1].1 == '_'))
            || (k < chars.len() && (chars[k].1.is_ascii_digit() || chars[k].1 == '_'));
        let skip = glued || prot[s..k].iter().any(|p| *p);
        runs.push(Run {
            a: s,
            b: k,
            word,
            skip,
        });
    }
    let is_ro_text = {
        let d = detect(text);
        d.lang == Some(Lang::Ro) && d.reliable
    };
    let lx = lex();
    let mut out = String::with_capacity(text.len() + 16);
    let mut res = Restored::default();
    let mut cursor = 0usize; // char index
    let keys: Vec<Option<String>> = runs
        .iter()
        .map(|r| {
            if !r.skip && r.word.is_ascii() {
                Some(r.word.to_ascii_lowercase())
            } else {
                None
            }
        })
        .collect();
    // known-count pass so heuristics can decide whether the text is Romanian
    let mut known_pre = 0usize;
    for (r, key) in runs.iter().zip(&keys) {
        if r.word.chars().any(|c| RO_LETTERS.contains(c))
            || key.as_ref().is_some_and(|k| lx.map.contains_key(k))
        {
            known_pre += 1;
        }
    }
    let heur_ok = is_ro_text || (runs.len() >= 3 && known_pre * 3 >= runs.len());
    for (idx, r) in runs.iter().enumerate() {
        for &(_, c) in &chars[cursor..r.a] {
            out.push(c);
        }
        let orig_end = r.b;
        let start_out = out.len();
        res.words += 1;
        let emit_orig = |out: &mut String| {
            for &(_, c) in &chars[r.a..orig_end] {
                out.push(c);
            }
        };
        if r.word.chars().any(|c| RO_LETTERS.contains(c)) {
            res.known += 1;
            emit_orig(&mut out);
            cursor = orig_end;
            continue;
        }
        let Some(key) = &keys[idx] else {
            emit_orig(&mut out);
            cursor = orig_end;
            continue;
        };
        let prev = idx
            .checked_sub(1)
            .and_then(|p| keys[p].as_deref())
            .filter(|_| {
                runs[idx - 1].b + 1 >= r.a
                    || chars[runs[idx - 1].b..r.a]
                        .iter()
                        .all(|x| !matches!(x.1, '.' | '!' | '?' | '\n' | ';'))
            });
        let next = keys.get(idx + 1).and_then(|k| k.as_deref()).filter(|_| {
            chars[r.b..runs[idx + 1].a]
                .iter()
                .all(|x| !matches!(x.1, '.' | '!' | '?' | '\n' | ';'))
        });
        let mut replacement: Option<(String, NoteKind)> = None;
        if let Some(forms) = lx.map.get(key) {
            res.known += 1;
            if forms.len() == 1 {
                if forms[0] != *key {
                    replacement = Some((forms[0].clone(), NoteKind::Lexicon));
                }
            } else if let Some(f) = context_pick(key, forms, prev, next) {
                if f != *key {
                    replacement = Some((f.to_string(), NoteKind::Context));
                }
            } else if mode == RestoreMode::Guess {
                if forms[0] != *key {
                    replacement = Some((forms[0].clone(), NoteKind::Ambiguous));
                } else {
                    res.notes.push(RestoreNote {
                        start: start_out,
                        end: start_out + r.word.len(),
                        original: r.word.clone(),
                        restored: r.word.clone(),
                        kind: NoteKind::Ambiguous,
                    });
                }
            } else {
                let cand: Vec<Vec<char>> = forms.iter().map(|f| f.chars().collect()).collect();
                let kc: Vec<char> = key.chars().collect();
                if cand.iter().all(|c| c.len() == kc.len()) {
                    let mut agree: Vec<char> = kc.clone();
                    let mut disputed = false;
                    for p in 0..kc.len() {
                        let c0 = cand[0][p];
                        if cand.iter().all(|c| c[p] == c0) {
                            agree[p] = c0;
                        } else {
                            disputed = true;
                        }
                    }
                    let s: String = agree.into_iter().collect();
                    if s != *key {
                        replacement = Some((s.clone(), NoteKind::Lexicon));
                    }
                    if disputed {
                        let shown = match_case(&r.word, &s);
                        // flagged below once the final text is emitted
                        let st = start_out;
                        out.push_str(&shown);
                        res.notes.push(RestoreNote {
                            start: st,
                            end: out.len(),
                            original: r.word.clone(),
                            restored: shown.clone(),
                            kind: NoteKind::Ambiguous,
                        });
                        if s != *key {
                            res.changed += 1;
                        }
                        cursor = orig_end;
                        continue;
                    }
                }
            }
        } else if heur_ok && key.len() >= 6 {
            for (suf, rep) in HEUR {
                if let Some(stem) = key.strip_suffix(suf) {
                    replacement = Some((format!("{stem}{rep}"), NoteKind::Heuristic));
                    break;
                }
            }
        }
        match replacement {
            Some((rep, kind)) => {
                let shown = match_case(&r.word, &rep);
                out.push_str(&shown);
                res.notes.push(RestoreNote {
                    start: start_out,
                    end: out.len(),
                    original: r.word.clone(),
                    restored: shown,
                    kind,
                });
                res.changed += 1;
            }
            None => emit_orig(&mut out),
        }
        cursor = orig_end;
    }
    for &(_, c) in &chars[cursor..] {
        out.push(c);
    }
    res.text = out;
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ normalisation
    #[test]
    fn ro_forms_both_directions() {
        let n = normalize("Şcoală, ţară, şi ŢARA Ş");
        assert_eq!(n.text, "Școală, țară, și ȚARA Ș");
        let o = NormOptions {
            ro: RoForm::Cedilla,
            ..Default::default()
        };
        let m = normalize_with("școală țară Ș Ț", &o);
        assert_eq!(m.text, "şcoală ţară Ş Ţ");
        assert_eq!(
            normalize_with(
                "ş ș",
                &NormOptions {
                    ro: RoForm::Keep,
                    ..Default::default()
                }
            )
            .text,
            "ş ș"
        );
    }

    #[test]
    fn mojibake_utf8_as_cp1252() {
        assert_eq!(normalize("Ã¢ntrebare").text, "ântrebare");
        assert_eq!(normalize("fiÅŸier").text, "fișier");
        assert_eq!(normalize("cÄƒtre").text, "către");
        assert_eq!(normalize("È™tiu È›ara").text, "știu țara");
        assert_eq!(normalize("cafÃ©").text, "café");
        assert_eq!(
            normalize("it â€œquotedâ€\u{9D} here").text,
            "it \"quoted\" here"
        );
        // legitimate text is untouched
        assert!(normalize("Ä is a letter, Ã too").is_unchanged());
        assert!(
            normalize("Österreich Ãrvíztűrő")
                .text
                .starts_with("Österreich")
        );
    }

    #[test]
    fn latin2_read_as_latin1() {
        assert_eq!(normalize("Bunã ziua, ºi þara").text, "Bună ziua, și țara");
        // ordinals and Portuguese are safe
        assert!(normalize("1º lugar, são João, ação").is_unchanged());
        assert!(normalize("a 2ª vez").is_unchanged());
    }

    #[test]
    fn hungarian_double_acute() {
        assert_eq!(
            normalize("Köszönöm, hogy õ és û itt van, nem baj").text,
            "Köszönöm, hogy ő és ű itt van, nem baj"
        );
        let kept = "Elle est sûre, c'est une vérité et une idée du pays";
        assert!(normalize(kept).is_unchanged());
        assert_eq!(
            normalize_with(
                "õ",
                &NormOptions {
                    hu: HuFix::Always,
                    ..Default::default()
                }
            )
            .text,
            "ő"
        );
        assert!(normalize("pões limões são").is_unchanged());
    }

    #[test]
    fn invisible_spaces_quotes_dashes_ligatures() {
        let n = normalize(
            "a\u{200B}b\u{FEFF}c\u{00AD}d \u{00A0}e\u{202F}f “x” ‘y’ „z” a–b a‐b a−b ﬁne ﬂow ﬃ",
        );
        assert_eq!(
            n.text,
            "abcd  e f \"x\" 'y' \"z\" a–b a-b a-b fine flow ffi"
        );
        let all = normalize_with(
            "a–b a—b",
            &NormOptions {
                dashes: DashMode::All,
                ..Default::default()
            },
        );
        assert_eq!(all.text, "a-b a-b");
        // ZWNJ between Latin letters goes, between Arabic letters stays
        assert_eq!(normalize("a\u{200C}b").text, "ab");
        assert_eq!(normalize("ب\u{200C}ب").text, "ب\u{200C}ب");
    }

    #[test]
    fn combining_marks_compose() {
        assert_eq!(
            normalize("s\u{0326}i t\u{0327}ara a\u{0306} e\u{0301}").text,
            "și țara ă é"
        );
        assert_eq!(normalize("o\u{030B}").text, "ő");
    }

    #[test]
    fn revert_is_exact() {
        for s in [
            "Şcoală Ã¢ntrebare \u{200B}x “q” ﬁ",
            "Bunã ºi þ 1º",
            "õ û köszönöm hogy nem",
            "plain ASCII",
            "",
            "a\u{0326}s\u{0326}",
        ] {
            let n = normalize(s);
            assert_eq!(n.revert(), s, "{s:?}");
        }
        let n = normalize("Şcoală Ã¢ntrebare");
        assert!(!n.report().is_empty() && n.counts().len() >= 2);
        assert_eq!(normalize("clean").report(), "no changes");
    }

    #[test]
    fn hostile_input_never_panics() {
        let weird = "\u{0}\u{FFFD}Ã\u{80}Â\u{A0}ÅÅŸ\u{0301}\u{0326}ﬁ\u{200D}€Ã¢";
        let n = normalize(weird);
        assert_eq!(n.revert(), weird);
        let _ = detect(weird);
        let _ = detect_segments(weird);
        let _ = restore_ro(weird, RestoreMode::Guess);
        let _ = strip_diacritics(weird);
    }

    #[test]
    fn turkish_case_and_fold() {
        assert_eq!(to_lower_tr("ISPARTA İSTANBUL"), "ısparta istanbul");
        assert_eq!(to_upper_tr("istanbul ısparta"), "İSTANBUL ISPARTA");
        assert_eq!(to_upper_lang("title", Lang::En), "TITLE");
        assert!(eq_fold("DİYARBAKIR", "diyarbakır", Lang::Tr));
        assert!(!eq_fold("I", "i", Lang::Tr));
        assert_eq!(
            strip_diacritics("Școală țară ăâî Straße Łódź Ñandú Çà ő ű"),
            "Scoala tara aai Strasse Lodz Nandu Ca o u"
        );
        assert_eq!(strip_diacritics("e\u{0301}"), "e");
        assert_eq!(strip_diacritics("Привет 日本"), "Привет 日本");
    }

    // ------------------------------------------------------------------ identification
    fn is(text: &str, l: Lang) {
        let d = detect(text);
        assert_eq!(
            d.lang,
            Some(l),
            "{text:?} -> {:?} (conf {:.2}, {:?})",
            d.lang,
            d.confidence,
            d.scores
        );
    }

    #[test]
    fn product_languages_sentences() {
        is(
            "The quick brown fox jumps over the lazy dog because it was not tired.",
            Lang::En,
        );
        is(
            "What time is it and where are you going with that?",
            Lang::En,
        );
        is(
            "Aceasta este o propoziție scrisă în limba română, pentru a testa detectorul.",
            Lang::Ro,
        );
        is(
            "Mâine mergem la munte, dacă nu plouă și dacă avem timp.",
            Lang::Ro,
        );
        is(
            "Le chat est sur la table et il regarde par la fenêtre avec les enfants.",
            Lang::Fr,
        );
        is(
            "Il gatto è sul tavolo e guarda dalla finestra insieme ai bambini.",
            Lang::It,
        );
        is(
            "El gato está sobre la mesa y mira por la ventana con los niños.",
            Lang::Es,
        );
        is(
            "O gato está em cima da mesa e olha pela janela com as crianças.",
            Lang::Pt,
        );
        is(
            "Die Katze sitzt auf dem Tisch und schaut mit den Kindern aus dem Fenster.",
            Lang::De,
        );
        is(
            "A macska az asztalon ül, és a gyerekekkel együtt nézi az ablakot.",
            Lang::Hu,
        );
    }

    #[test]
    fn product_languages_short() {
        is("Buna ziua, ce mai faceti?", Lang::Ro);
        is("Bună ziua, ce mai faceți?", Lang::Ro);
        is("Multumesc foarte mult pentru ajutor", Lang::Ro);
        is("Bonjour, comment allez-vous aujourd'hui?", Lang::Fr);
        is("Buongiorno, come stai oggi?", Lang::It);
        is("Hola, ¿cómo estás hoy?", Lang::Es);
        is("Olá, como você está hoje? Obrigado!", Lang::Pt);
        is("Guten Tag, wie geht es Ihnen heute?", Lang::De);
        is("Szia, hogy vagy ma? Köszönöm szépen!", Lang::Hu);
        is("Thank you very much, please", Lang::En);
    }

    #[test]
    fn diacritic_free_romanian_versus_neighbours() {
        is(
            "Nu stiu daca pot sa vin maine la intalnire, dar o sa va anunt.",
            Lang::Ro,
        );
        is("Sunt foarte bine, multumesc! Tu ce faci azi?", Lang::Ro);
        is("Vreau sa merg la mare in vacanta aceasta", Lang::Ro);
    }

    #[test]
    fn other_languages() {
        is(
            "Het huis van mijn vader is groot en er staat een boom in de tuin.",
            Lang::Nl,
        );
        is(
            "To jest bardzo dobry dzień, ponieważ świeci słońce i jestem w domu.",
            Lang::Pl,
        );
        is(
            "Dnes je velmi hezký den a já jsem rád, že jsem doma s rodinou.",
            Lang::Cs,
        );
        is(
            "Dnes je veľmi pekný deň a ja som rád, že som doma s rodinou.",
            Lang::Sk,
        );
        is(
            "Jag har en katt och den sitter på bordet och tittar ut genom fönstret.",
            Lang::Sv,
        );
        is(
            "Jeg har en kat, og den sidder på bordet og kigger ud af vinduet med mig.",
            Lang::Da,
        );
        is(
            "Jeg har en katt, og den sitter på bordet og ser ut av vinduet med meg.",
            Lang::No,
        );
        is(
            "Minulla on kissa ja se istuu pöydällä ja katsoo ikkunasta ulos.",
            Lang::Fi,
        );
        is(
            "Bugün hava çok güzel ve ben evde değilim, çünkü arkadaşlarımla dışarıdayım.",
            Lang::Tr,
        );
        is(
            "Сегодня очень хорошая погода, и мы идём гулять в парк с детьми.",
            Lang::Ru,
        );
        is(
            "Сьогодні дуже гарна погода, і ми йдемо гуляти в парк з дітьми, якщо хочеш.",
            Lang::Uk,
        );
        is(
            "Днес времето е много хубаво и ние отиваме в парка със семейството си.",
            Lang::Bg,
        );
        is(
            "Σήμερα ο καιρός είναι πολύ καλός και πάμε βόλτα στο πάρκο.",
            Lang::El,
        );
        is(
            "اليوم الطقس جميل جدا ونحن ذاهبون إلى الحديقة مع العائلة",
            Lang::Ar,
        );
        is(
            "היום מזג האוויר יפה מאוד ואנחנו הולכים לפארק עם המשפחה",
            Lang::He,
        );
        is("今天天气很好，我们和家人一起去公园。", Lang::Zh);
        is(
            "今日はとても天気がいいので、家族と一緒に公園へ行きます。",
            Lang::Ja,
        );
        is(
            "오늘은 날씨가 아주 좋아서 가족과 함께 공원에 갑니다.",
            Lang::Ko,
        );
        is(
            "आज मौसम बहुत अच्छा है और हम परिवार के साथ पार्क जा रहे हैं।",
            Lang::Hi,
        );
    }

    #[test]
    fn short_or_empty_input_is_honest() {
        for s in ["", "   ", "12345", "!!!", "ok", "xyz", "Python", "🙂"] {
            let d = detect(s);
            assert!(!d.reliable, "{s:?} must not be reliable: {d:?}");
        }
        assert_eq!(detect("").lang, None);
        // a bare ambiguous greeting is a tie between German and Dutch, not a guess
        assert!(!detect("Hallo").reliable);
        assert_eq!(
            detect_lang("Salut, ça va? Merci beaucoup, et vous aussi!"),
            Some(Lang::Fr)
        );
        assert_eq!(detect_lang("hmm"), None);
    }

    #[test]
    fn detection_probabilities_sum_sensibly() {
        let d = detect("Aceasta este o propoziție scrisă în limba română.");
        assert!(d.reliable && d.confidence > 0.5);
        let sum: f32 = d.scores.iter().map(|x| x.1).sum();
        assert!(sum <= 1.0001 && sum > 0.5);
        assert!(d.prob(Lang::Ro) > d.prob(Lang::It));
        assert_eq!(d.script, Script::Latin);
    }

    #[test]
    fn segments_mixed_text() {
        let t = "Bună ziua, ce mai faceți astăzi? The weather is very nice and we are going to the park. Je suis très content de vous voir ici avec les enfants.";
        let segs = detect_segments(t);
        let langs: Vec<_> = segs.iter().map(|s| s.lang).collect();
        assert_eq!(
            langs,
            vec![Some(Lang::Ro), Some(Lang::En), Some(Lang::Fr)],
            "{segs:?}"
        );
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs.last().unwrap().end, t.len());
        for w in segs.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        // script switch inside one sentence
        let m = detect_segments(
            "Мне очень нравится это место и погода сегодня hello my friend how are you today",
        );
        assert_eq!(
            m.iter().map(|s| s.lang).collect::<Vec<_>>(),
            vec![Some(Lang::Ru), Some(Lang::En)],
            "{m:?}"
        );
        // a Latin name inside a Russian sentence does not split it
        let n = detect_segments("Я очень люблю программировать на Rust и сегодня я пишу новый код");
        assert_eq!(n.len(), 1, "{n:?}");
        assert!(detect_segments("").is_empty());
    }

    #[test]
    fn lang_codes_roundtrip() {
        for l in Lang::ALL {
            assert_eq!(Lang::from_code(l.code()), Some(l));
        }
        assert_eq!(Lang::from_code("RO-ro"), Some(Lang::Ro));
        assert_eq!(Lang::from_code("pt_PT"), Some(Lang::Pt));
        assert_eq!(Lang::from_code("nb"), Some(Lang::No));
        assert_eq!(Lang::from_code("xx"), None);
        assert!(Lang::Hu.is_product() && !Lang::Nl.is_product());
        assert_eq!(script_of('ș'), Script::Latin);
        assert_eq!(script_of('ж'), Script::Cyrillic);
        assert_eq!(script_of('中'), Script::Han);
        assert_eq!(script_of('1'), Script::Common);
    }

    // ------------------------------------------------------------------ answer-language lock
    fn toy_vocab() -> Vec<Vec<u8>> {
        let t = |s: &str| s.as_bytes().to_vec();
        vec![
            t("hello"),
            t(" și"),
            t("ț"),
            t("ţ"),
            t("привет"),
            t("中文"),
            t("مرحبا"),
            t("123"),
            t("."),
            vec![],
            vec![0xD0],
            vec![0xB0],
            vec![0xE4],
            vec![0xE4, 0xB8],
            t("ß"),
            t("é"),
            t("ă"),
            t(" cod"),
            vec![0xF0, 0x9F, 0x99],
            vec![0xEA, 0xB0],
        ]
    }

    #[test]
    fn lock_mask_for_romanian() {
        let lock = LangLock::from_token_bytes(&toy_vocab(), Lang::Ro);
        let m = lock.mask();
        assert_eq!(m.len(), 20);
        let allowed = |i: u32| m.allows(i);
        assert!(
            allowed(0)
                && allowed(1)
                && allowed(2)
                && allowed(3)
                && allowed(7)
                && allowed(8)
                && allowed(9)
        );
        assert!(!allowed(4) && !allowed(5) && !allowed(6));
        // partial UTF-8 of forbidden scripts is blocked, continuation bytes and emoji prefixes are neutral
        assert!(
            !allowed(10)
                && allowed(11)
                && !allowed(12)
                && !allowed(13)
                && allowed(18)
                && !allowed(19)
        );
        assert_eq!(lock.forbidden_count(), 7);
        assert!(allowed(14) && allowed(16));
        let strict = LangLock::from_token_bytes(&toy_vocab(), Lang::Ro).deny_ro_cedilla(true);
        assert!(!strict.mask().allows(3) && strict.mask().allows(2));
    }

    #[test]
    fn lock_bias_targets_diacritics() {
        let lock = LangLock::from_token_bytes(&toy_vocab(), Lang::Ro);
        let b: HashMap<u32, f32> = lock.bias_with(2.0, 1.0).into_iter().collect();
        assert_eq!(b.get(&2), Some(&2.0)); // ț
        assert_eq!(b.get(&16), Some(&2.0)); // ă
        assert_eq!(b.get(&3), Some(&-4.0)); // ţ cedilla: pushed away at twice the boost
        assert_eq!(b.get(&14), Some(&-1.0)); // ß is foreign to Romanian
        assert!(!b.contains_key(&0) && !b.contains_key(&4));
    }

    #[test]
    fn lock_for_non_latin_targets() {
        let ru = LangLock::from_token_bytes(&toy_vocab(), Lang::Ru);
        let m = ru.mask();
        assert!(m.allows(4) && m.allows(0) && !m.allows(5) && !m.allows(6));
        let strict = LangLock::from_token_bytes(&toy_vocab(), Lang::Ru).forbid_latin();
        assert!(!strict.mask().allows(0) && strict.mask().allows(4));
        let ja = LangLock::from_token_bytes(&toy_vocab(), Lang::Ja);
        assert!(ja.mask().allows(5) && !ja.mask().allows(4));
        let wide = LangLock::from_token_bytes(&toy_vocab(), Lang::En).allow(Script::Cyrillic);
        assert!(wide.mask().allows(4));
    }

    #[test]
    fn guard_detects_script_and_language_drift() {
        let mut g = LangGuard::new(Lang::Ro);
        let ro = "Aceasta este o propoziție scrisă în limba română, pentru a testa detectorul de abatere. ";
        for _ in 0..4 {
            assert!(g.feed(ro).is_none());
        }
        // Cyrillic leak fires immediately
        let d = g
            .feed("Теперь я пишу по-русски без причины, как будто так и надо.")
            .expect("script drift");
        assert!(matches!(d.kind, DriftKind::Script(Script::Cyrillic)));
        assert!(
            g.feed("ещё немного русского текста для проверки").is_none(),
            "only reported once"
        );

        // English leak needs two failed checks
        let mut g = LangGuard::new(Lang::Ro);
        let en = "The weather is very nice today and we are going to the park with all of the children. ";
        let mut hit = None;
        for _ in 0..6 {
            if let Some(d) = g.feed(en) {
                hit = Some(d);
                break;
            }
        }
        let d = hit.expect("language drift");
        assert_eq!(d.kind, DriftKind::Language(Lang::En));
        assert!(d.at_char > 40);
    }

    #[test]
    fn guard_ignores_code_and_quotes_and_handles_split_utf8() {
        let mut g = LangGuard::new(Lang::Ro);
        let ro = "Aceasta este o propoziție scrisă în limba română, pentru a testa detectorul de abatere. ";
        g.feed(ro);
        assert!(g.feed("```\nlet privet = \"привет мир, это просто код\";\nfor i in 0..10 { println!(\"{}\", i); }\n```").is_none());
        for _ in 0..3 {
            assert!(g.feed(ro).is_none());
        }
        // multi-byte letters arriving byte by byte
        let mut g = LangGuard::new(Lang::Ro);
        let mut fired = None;
        for b in "Теперь я пишу по-русски, как будто так и надо, правда?".bytes()
        {
            if let Some(d) = g.feed_bytes(&[b]) {
                fired = Some(d);
            }
        }
        assert!(matches!(
            fired.map(|d| d.kind),
            Some(DriftKind::Script(Script::Cyrillic))
        ));
        g.reset();
        assert_eq!(g.chars_seen(), 0);
    }

    // ------------------------------------------------------------------ Romanian formatting
    #[test]
    fn numbers_money_percent() {
        assert_eq!(format_int_ro(1234567), "1.234.567");
        assert_eq!(format_int_ro(-1234), "-1.234");
        assert_eq!(format_int_ro(12), "12");
        assert_eq!(format_decimal_ro(1234.56, 2), "1.234,56");
        assert_eq!(format_decimal_ro(0.5, 2), "0,50");
        assert_eq!(format_decimal_ro(-1234567.891, 1), "-1.234.567,9");
        assert_eq!(format_decimal_ro(-0.001, 2), "0,00");
        assert_eq!(format_decimal_ro(1000.0, 0), "1.000");
        assert_eq!(format_decimal_ro(f64::NAN, 2), "NaN");
        assert_eq!(format_percent_ro(12.5, 1), "12,5%");
        assert_eq!(format_money_ro(1234.56, "RON"), "1.234,56 lei");
        assert_eq!(format_money_ro(0.5, "lei"), "0,50 lei");
        assert_eq!(format_money_ro(1.0, "RON"), "1 leu");
        assert_eq!(format_money_ro(2.0, "RON"), "2 lei");
        assert_eq!(format_money_ro(19.0, "RON"), "19 lei");
        assert_eq!(format_money_ro(20.0, "RON"), "20 de lei");
        assert_eq!(format_money_ro(101.0, "RON"), "101 lei");
        assert_eq!(format_money_ro(100.0, "RON"), "100 de lei");
        assert_eq!(format_money_ro(1000.0, "RON"), "1.000 de lei");
        assert_eq!(format_money_ro(0.0, "RON"), "0 lei");
        assert_eq!(format_money_ro(-5.5, "RON"), "-5,50 lei");
        assert_eq!(format_money_ro(20.0, "EUR"), "20 de euro");
        assert_eq!(format_money_ro(3.0, "USD"), "3 dolari");
        assert_eq!(format_money_ro(12.34, "CHF"), "12,34 CHF");
        assert_eq!(format_money_minor_ro(123456, "ron"), "1.234,56 lei");
    }

    #[test]
    fn number_parsing() {
        assert_eq!(parse_number_ro("1.234,56"), Some(1234.56));
        assert_eq!(parse_number_ro("-12,5"), Some(-12.5));
        assert_eq!(parse_number_ro("1 234,5"), Some(1234.5));
        assert_eq!(parse_number_ro("1.234"), Some(1234.0));
        assert_eq!(parse_number_ro("1234"), Some(1234.0));
        assert_eq!(parse_number_ro("2,75"), Some(2.75));
        assert_eq!(parse_number_ro("1.23"), None);
        assert_eq!(parse_number_ro("1,2,3"), None);
        assert_eq!(parse_number_ro("12.3456"), None);
        assert_eq!(parse_number_ro(""), None);
        assert_eq!(parse_number_ro("abc"), None);
        assert_eq!(parse_number_ro("-"), None);
        for x in [0.0, 1.0, 999.99, 1234567.89, 0.01] {
            assert_eq!(parse_number_ro(&format_decimal_ro(x, 2)), Some(x));
        }
    }

    #[test]
    fn dates() {
        assert_eq!(
            format_date_ro(2026, 11, 24, false).as_deref(),
            Some("24 noiembrie 2026")
        );
        assert_eq!(
            format_date_ro(2026, 11, 24, true).as_deref(),
            Some("marți, 24 noiembrie 2026")
        );
        assert_eq!(
            format_date_ro(2024, 2, 29, true).as_deref(),
            Some("joi, 29 februarie 2024")
        );
        assert_eq!(format_date_ro(2025, 2, 29, false), None);
        assert_eq!(format_date_ro(2025, 13, 1, false), None);
        assert_eq!(
            format_date_ro(2000, 1, 1, true).as_deref(),
            Some("sâmbătă, 1 ianuarie 2000")
        );
        assert_eq!(weekday(1970, 1, 1), 4);
        assert_eq!(parse_date_ro("24 noiembrie 2026"), Some((2026, 11, 24)));
        assert_eq!(
            parse_date_ro("Marți, 24 NOIEMBRIE 2026"),
            Some((2026, 11, 24))
        );
        assert_eq!(
            parse_date_ro("marti, 24 noiembrie 2026"),
            Some((2026, 11, 24))
        );
        assert_eq!(parse_date_ro("1 mar. 2025"), Some((2025, 3, 1)));
        assert_eq!(parse_date_ro("24.11.2026"), Some((2026, 11, 24)));
        assert_eq!(parse_date_ro("24/11/2026"), Some((2026, 11, 24)));
        assert_eq!(parse_date_ro("2026-11-24"), Some((2026, 11, 24)));
        assert_eq!(parse_date_ro("31 februarie 2026"), None);
        assert_eq!(parse_date_ro("24.11.26"), None);
        assert_eq!(parse_date_ro("mâine"), None);
        for (y, m, d) in [(2026, 11, 24), (2024, 2, 29), (1999, 12, 31)] {
            assert_eq!(
                parse_date_ro(&format_date_ro(y, m, d, true).unwrap()),
                Some((y, m, d))
            );
        }
    }

    // ------------------------------------------------------------------ restoration
    fn fix(s: &str) -> String {
        restore_ro(s, RestoreMode::Conservative).text
    }

    #[test]
    fn lexicon_is_consistent() {
        assert!(ro_lexicon_size() > 700, "{}", ro_lexicon_size());
        for f in LEX_FORMS
            .split_whitespace()
            .chain(LEX_PAIRS_A.split_whitespace())
        {
            let k = strip_diacritics(&f.to_lowercase());
            assert!(k.is_ascii() && !k.is_empty(), "{f}");
        }
        for g in LEX_GROUPS.split_whitespace() {
            let forms: Vec<&str> = g.split('|').collect();
            let k = strip_diacritics(forms[0]);
            for f in &forms {
                assert_eq!(strip_diacritics(f), k, "{g}");
            }
            assert!(forms.len() >= 2);
        }
        // the plain list never collides with an accented word
        for w in LEX_PLAIN.split_whitespace() {
            assert_eq!(strip_diacritics(w), w, "{w}");
        }
    }

    #[test]
    fn restores_common_sentences() {
        assert_eq!(
            fix("Buna ziua, ce mai faceti?"),
            "Bună ziua, ce mai faceți?"
        );
        assert_eq!(
            fix("Multumesc foarte mult, mergem la scoala maine"),
            "Mulțumesc foarte mult, mergem la școala mâine"
        );
        assert_eq!(
            fix("Nu stiu daca pot sa vin la intalnire"),
            "Nu știu dacă pot să vin la intalnire"
        );
        assert_eq!(
            fix("Vreau sa invat sa programez"),
            "Vreau să învăț să programez"
        );
        assert_eq!(fix("ASTAZI SI MAINE"), "ASTĂZI ȘI MÂINE");
        assert_eq!(fix("Stefan a venit dupa mine"), "Ștefan a venit după mine");
        assert_eq!(fix("Bucuresti si Iasi"), "București și Iași");
    }

    #[test]
    fn context_rules_for_sa_ca_va() {
        assert_eq!(fix("Stiu ca vine, ca sa vezi"), "Știu că vine, ca să vezi");
        assert_eq!(
            fix("Pentru ca nu poate sa vina"),
            "Pentru că nu poate să vina"
        );
        assert_eq!(fix("Va rog sa ma ajutati"), "Vă rog să mă ajutați");
        assert_eq!(fix("Va multumesc"), "Vă mulțumesc");
        assert_eq!(fix("El va fi acolo"), "El va fi acolo");
        assert_eq!(fix("Am vrut sa nu"), "Am vrut să nu");
    }

    #[test]
    fn ambiguity_is_flagged_not_guessed() {
        let r = restore_ro("Fata si tara", RestoreMode::Conservative);
        assert!(r.text.starts_with("Fata"), "{}", r.text);
        assert!(
            r.text.contains("și") && r.text.contains("țara"),
            "{}",
            r.text
        );
        assert!(
            r.notes
                .iter()
                .any(|n| n.kind == NoteKind::Ambiguous && n.original == "Fata")
        );
        assert!(r.flagged().count() >= 1);
        let r = restore_ro("o scoala", RestoreMode::Conservative);
        assert_eq!(r.text, "o școală");
        let r = restore_ro("scoala mea", RestoreMode::Conservative);
        assert_eq!(r.text, "școala mea");
        assert!(r.notes.iter().any(|n| n.kind == NoteKind::Ambiguous));
        // guess mode takes the preferred reading
        assert_eq!(
            restore_ro("Alta idee", RestoreMode::Guess).text,
            "Altă idee"
        );
        assert_eq!(
            restore_ro("Alta idee", RestoreMode::Conservative).text,
            "Alta idee"
        );
        assert_eq!(
            restore_ro("Data nasterii", RestoreMode::Conservative).text,
            "Data nasterii"
        );
    }

    #[test]
    fn leaves_protected_text_alone() {
        let s = "Scrie la ion@exemplu.ro sau vezi https://exemplu.ro/stiri/maine si `let sa = 1;` si fisier_nou si H2O si /usr/lib/scoala";
        let out = fix(s);
        assert!(
            out.contains("ion@exemplu.ro") && out.contains("https://exemplu.ro/stiri/maine"),
            "{out}"
        );
        assert!(
            out.contains("`let sa = 1;`")
                && out.contains("fisier_nou")
                && out.contains("H2O")
                && out.contains("/usr/lib/scoala"),
            "{out}"
        );
        assert!(out.contains(" și "), "{out}");
        // already-accented words and other languages
        assert_eq!(fix("Școala și țara"), "Școala și țara");
        let en = restore_ro(
            "The tara of the scoala is in a data center",
            RestoreMode::Conservative,
        );
        assert!(en.low_confidence() || en.text.contains("tara") || en.text.contains("țara"));
    }

    #[test]
    fn heuristics_apply_only_in_romanian_text() {
        let r = restore_ro(
            "Aceasta federatie are multe plantatii si legatii bune, dar nu stiu daca functioneaza",
            RestoreMode::Conservative,
        );
        assert!(
            r.text.contains("federație")
                && r.text.contains("plantații")
                && r.text.contains("legații"),
            "{}",
            r.text
        );
        assert!(r.notes.iter().any(|n| n.kind == NoteKind::Heuristic));
        let en = restore_ro(
            "The organization has many relations and the situation is stable",
            RestoreMode::Conservative,
        );
        assert!(
            !en.text.contains('ț') && !en.text.contains('ș'),
            "{}",
            en.text
        );
    }

    #[test]
    fn restoration_reports_coverage_and_offsets() {
        let r = restore_ro("Buna ziua, kubernetes si docker", RestoreMode::Conservative);
        assert!(r.words == 5 && r.known >= 3 && r.changed >= 2);
        for n in &r.notes {
            assert_eq!(&r.text[n.start..n.end], n.restored);
        }
        let u = restore_ro("blorp zint quark", RestoreMode::Conservative);
        assert!(u.low_confidence() && u.changed == 0);
        assert_eq!(restore_ro("", RestoreMode::Guess).text, "");
        assert_eq!(restore_ro("123 !!!", RestoreMode::Guess).changed, 0);
    }

    #[test]
    fn restored_text_roundtrips_through_strip() {
        // stripping the restored text gives back the input for the sentences we handle
        for s in [
            "Buna ziua, ce mai faceti?",
            "Multumesc pentru raspuns, maine incepem lucrarea",
            "Stiu ca vrei sa mergi la scoala",
        ] {
            let r = restore_ro(s, RestoreMode::Guess);
            assert_eq!(strip_diacritics(&r.text), s);
        }
    }
}
