// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! Citation audit: a deterministic, model-free check of whether each citation in a draft (or each `[n]` tag in a
//! model's answer) points at a passage that contains what the sentence says, and whether quotations exist word for
//! word.
//!
//! # What this does and does not tell you
//!
//! It checks **traceability, not truth**. A `supported` verdict means that most of the sentence's terms occur in
//! the passage it cites, that every figure in the sentence also occurs there, and that the cited page (when there
//! is one) is plausible. It does not mean the sentence is correct, that the source is right, or that the sentence
//! means what the passage means:
//!
//! * **Paraphrases are a weak spot.** Term overlap does not see synonyms. TEZAUR's own test on six faithful
//!   paraphrases recognised none of them (0 of 6): they come out `weak` or `unsupported`.
//! * **Negation is a weak spot.** "The authority must not refuse" overlaps almost completely with "The authority
//!   must refuse"; only a reader (or the optional second opinion, [`semantic_pass`]) sees the difference.
//! * Overlap is also high for a sentence that merely reuses the topic words of a passage.
//! * Every other verdict (`weak`, `unsupported`, `wrong-page`, ...) is a place where a person must open the page.
//!   Nothing here removes that step.
//!
//! The optional second opinion takes a caller-supplied `judge` callback; there is no model in this crate and no
//! network access. A model can be wrong as well, so every verdict it changes says so in its note, and it can never
//! turn a `number-mismatch` into anything but `contradicted`.
//!
//! # Where the text comes from
//!
//! The library is anything that implements [`Passages`] (see [`MemPassages`] for an in-memory one). Citations are
//! pandoc style (`[@key, p. 3; @other]`, LaTeX and LyX sources are converted first) for [`audit`], and `[n]` /
//! `[n, m]` tags for [`audit_answer`].
//!
//! # Hostile input
//!
//! No input makes these functions panic or run for long: input size, number of citations, quotations and sentences,
//! and total work are capped by [`AuditOptions`]; when a cap is hit the report says so (`AuditReport::notes`)
//! instead of silently checking less. Text is compared after Unicode folding (accents, ligatures, case), so
//! look-alike letters from other scripts (a Cyrillic "а" for a Latin "a") do **not** match: a quotation altered that
//! way is reported as not found, which is the safe direction.

mod markup;
mod numbers;
mod text;

use crate::locator::{self, Locator, Val};
use crate::sanitize;
use crate::unitext::token_strings;
use std::collections::{HashMap, HashSet};

pub use markup::normalize_markup;
pub use numbers::{figures, missing_numbers, number_words};
pub use text::{norm, sentences};

/// Share of the sentence's terms that must be in the passage for `supported`.
pub const SUPPORTED: f32 = 0.6;
/// Share below which a sentence is `unsupported` rather than `weak`.
pub const WEAK: f32 = 0.35;

// ------------------------------------------------------------------ verdicts

/// The outcome for one citation (or one quotation, or one uncited statement).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    /// Enough of the sentence's terms and all its figures are in the cited passage (see the module docs).
    Supported,
    /// Some of the terms are in the source, not enough to call it supported: read the page.
    Weak,
    /// Hardly any of the terms occur anywhere in the cited source.
    Unsupported,
    /// The cited page does not cover the sentence, another page of the same source does.
    WrongPage,
    /// The sentence states a figure (digits or number words) that the best passage does not contain.
    NumberMismatch,
    /// No source in the library has the cited key (or the `[n]` excerpt does not exist).
    UnknownSource,
    /// The cited page is beyond the last page of the source.
    PageOutOfRange,
    /// A page interval that ends before it starts ("pp. 5-3"). Beyond the twelve verdicts of the original.
    BadRange,
    /// A long statement with no citation.
    Uncited,
    /// A quotation found word for word in the cited source.
    QuoteVerbatim,
    /// A quotation not in the cited source but found in another one.
    QuoteElsewhere,
    /// A quotation found nowhere in the library (or the search limit was reached; the note says which).
    QuoteNotFound,
    /// Set only by the optional second opinion: the passage seems to say something different.
    Contradicted,
}

impl Verdict {
    pub const ALL: [Verdict; 13] = [
        Verdict::Supported,
        Verdict::Weak,
        Verdict::Unsupported,
        Verdict::WrongPage,
        Verdict::NumberMismatch,
        Verdict::UnknownSource,
        Verdict::PageOutOfRange,
        Verdict::BadRange,
        Verdict::Uncited,
        Verdict::QuoteVerbatim,
        Verdict::QuoteElsewhere,
        Verdict::QuoteNotFound,
        Verdict::Contradicted,
    ];

    /// Stable machine name (the strings of the original).
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Supported => "supported",
            Verdict::Weak => "weak",
            Verdict::Unsupported => "unsupported",
            Verdict::WrongPage => "wrong-page",
            Verdict::NumberMismatch => "number-mismatch",
            Verdict::UnknownSource => "unknown-source",
            Verdict::PageOutOfRange => "page-out-of-range",
            Verdict::BadRange => "bad-range",
            Verdict::Uncited => "uncited",
            Verdict::QuoteVerbatim => "quote-verbatim",
            Verdict::QuoteElsewhere => "quote-elsewhere",
            Verdict::QuoteNotFound => "quote-not-found",
            Verdict::Contradicted => "contradicted",
        }
    }

    pub fn parse(s: &str) -> Option<Verdict> {
        Verdict::ALL.into_iter().find(|v| v.as_str() == s)
    }

    /// Everything except `supported` and `quote-verbatim` is a place to open the page.
    pub fn is_problem(self) -> bool {
        !matches!(self, Verdict::Supported | Verdict::QuoteVerbatim)
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug)]
pub struct Finding {
    /// The sentence (or the quotation), cut to 1000 characters for very long ones.
    pub sentence: String,
    /// The cited key (or `[n]`), when there is one.
    pub cite: Option<String>,
    /// First checkable page that was cited.
    pub page: Option<usize>,
    pub verdict: Verdict,
    /// Share of the sentence's terms found in the compared passage (0..1); 1 or 0 for quotations.
    pub support: f32,
    pub note: String,
    /// The passage the sentence was compared with (cut to 1500 characters); the input of the second opinion.
    pub passage: Option<String>,
}

impl Finding {
    pub fn is_problem(&self) -> bool {
        self.verdict.is_problem()
    }
}

// ------------------------------------------------------------------- library

/// What the audit needs to know about one source.
#[derive(Clone, Debug, Default)]
pub struct SourceInfo {
    /// The citation key (matched case-insensitively).
    pub key: String,
    /// Other names that resolve to the same source (file name without extension, a BibTeX alias...).
    pub aliases: Vec<String>,
    pub title: String,
    /// Publication year as written ("2019"); a bare number equal to it is not read as a page.
    pub year: String,
    /// Number of pages; 0 when unknown (then no page is ever out of range).
    pub pages: u32,
    /// Appended to the note of every finding that checked a page (an EPUB's declared print pagination, say).
    pub page_note: Option<String>,
}

/// One passage of a source. Page 0 means the source has no page numbers (text, EPUB, DOCX...).
#[derive(Clone, Copy, Debug)]
pub struct Passage<'a> {
    pub page: u32,
    pub text: &'a str,
}

/// The library the audit reads. Implement it over the real index; [`MemPassages`] is a small in-memory one.
pub trait Passages {
    fn source_count(&self) -> usize;
    fn source(&self, id: usize) -> Option<SourceInfo>;
    /// The passages of source `id`, in document order.
    fn passages(&self, id: usize) -> Vec<Passage<'_>>;
}

#[derive(Clone, Debug, Default)]
pub struct MemSource {
    pub info: SourceInfo,
    pub passages: Vec<(u32, String)>,
}

#[derive(Clone, Debug, Default)]
pub struct MemPassages {
    pub sources: Vec<MemSource>,
}

impl MemPassages {
    pub fn add(&mut self, info: SourceInfo, passages: Vec<(u32, String)>) {
        self.sources.push(MemSource { info, passages });
    }
}

impl Passages for MemPassages {
    fn source_count(&self) -> usize {
        self.sources.len()
    }
    fn source(&self, id: usize) -> Option<SourceInfo> {
        self.sources.get(id).map(|s| s.info.clone())
    }
    fn passages(&self, id: usize) -> Vec<Passage<'_>> {
        self.sources
            .get(id)
            .map(|s| {
                s.passages
                    .iter()
                    .map(|(page, text)| Passage {
                        page: *page,
                        text: text.as_str(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ------------------------------------------------------------------- options

#[derive(Clone, Debug)]
pub struct AuditOptions {
    /// Input beyond this many bytes is not read (cut on a character boundary).
    pub max_input_bytes: usize,
    /// Citation groups extracted from the text.
    pub max_groups: usize,
    /// Individual citations checked; the rest are counted in `AuditReport::skipped_citations`.
    pub max_citations: usize,
    /// Quotations checked.
    pub max_quotes: usize,
    pub max_sentences: usize,
    pub max_findings: usize,
    /// Longest sentence (in characters) that is analysed; the rest of a longer one is ignored.
    pub max_sentence_chars: usize,
    /// Most distinct terms of a sentence that are compared.
    pub max_terms: usize,
    /// Work allowed in total (about one unit per byte of library text read plus one per term comparison).
    pub work_budget: u64,
    /// Library tokens kept in memory between citations.
    pub cache_tokens: usize,
    /// Normalised library text kept in memory (bytes).
    pub cache_text_bytes: usize,
    /// Sources indexed by key.
    pub max_sources: usize,
}

impl Default for AuditOptions {
    fn default() -> Self {
        AuditOptions {
            max_input_bytes: 2 << 20,
            max_groups: 20_000,
            max_citations: 20_000,
            max_quotes: 2_000,
            max_sentences: 50_000,
            max_findings: 100_000,
            max_sentence_chars: 4_000,
            max_terms: 400,
            work_budget: 150_000_000,
            cache_tokens: 4_000_000,
            cache_text_bytes: 256 << 20,
            max_sources: 1_000_000,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AuditReport {
    pub findings: Vec<Finding>,
    /// Plain-language statements of every limit that was hit ("the input was cut at ..."). Empty when nothing was.
    pub notes: Vec<String>,
    /// Citations that were not checked because a limit was hit.
    pub skipped_citations: usize,
}

impl AuditReport {
    pub fn count(&self, v: Verdict) -> usize {
        self.findings.iter().filter(|f| f.verdict == v).count()
    }

    /// True when a cap was hit: the findings are then a prefix of the answer, not all of it.
    pub fn truncated(&self) -> bool {
        !self.notes.is_empty() || self.skipped_citations > 0
    }

    pub fn markdown(&self) -> String {
        let mut s = audit_markdown(&self.findings);
        if self.truncated() {
            s.push_str("\n**Incomplete.**");
            for n in &self.notes {
                s.push_str(&format!(" {n}"));
            }
            if self.skipped_citations > 0 {
                s.push_str(&format!(
                    " {} citation(s) were not checked.",
                    self.skipped_citations
                ));
            }
            s.push('\n');
        }
        s
    }
}

// ------------------------------------------------------------------ citations

#[derive(Clone, Debug, PartialEq)]
pub struct Cite {
    pub key: String,
    /// First checkable page, as written (a year-like bare number is sorted out later, against the cited work's
    /// year).
    pub page: Option<usize>,
    /// Everything written after the key, interpreted.
    pub locators: Vec<Locator>,
}

const MAX_KEY_CHARS: usize = 200;
const MAX_LOCATOR_CHARS: usize = 200;

/// Replaces every pandoc-style group `[@a, p. 3; @b]` with a marker character; returns the marked text and the
/// groups in marker order. (Bounded: a group longer than 600 characters is not a group.)
pub fn extract_cites(text: &str) -> (String, Vec<Vec<Cite>>) {
    let clean = text::strip_marks(text);
    let (t, g, _) = text::extract_groups(&clean, &parse_group, usize::MAX);
    (t, g)
}

fn parse_group(inner: &str) -> Vec<Cite> {
    let mut v = Vec::new();
    for item in inner.split(';') {
        let item = item.trim();
        // `@key`, `-@key` (suppress author), `see @key`; an `@` inside a word (an e-mail address) is not a citation
        let at = if item.starts_with('@') {
            Some(0)
        } else {
            item.char_indices()
                .find(|&(i, c)| {
                    c == '@' && item[..i].ends_with(|p: char| p.is_whitespace() || p == '-')
                })
                .map(|(i, _)| i)
        };
        let Some(at) = at else { continue };
        let rest = &item[at + 1..];
        let key: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || "_:.-/".contains(*c))
            .take(MAX_KEY_CHARS)
            .collect();
        let key = key.trim_end_matches(['.', ':', '-']).to_string();
        if key.is_empty() {
            continue;
        }
        let tail: String = rest[key.len()..].chars().take(MAX_LOCATOR_CHARS).collect();
        let locators = locator::parse(&tail);
        let page = locators.iter().find_map(|l| l.page_range()).map(|(a, _)| a);
        v.push(Cite {
            key,
            page,
            locators,
        });
    }
    v
}

/// `[n]`, `[n, m]` or `[n; m]` with digits only. A number that does not fit `usize` is `usize::MAX` (out of range).
fn numbered_group(inner: &str) -> Vec<Cite> {
    let mut v = Vec::new();
    for p in inner.split([',', ';']).map(str::trim) {
        if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) {
            return Vec::new();
        }
        let n = p.parse::<usize>().unwrap_or(usize::MAX);
        v.push(Cite {
            key: n.to_string(),
            page: None,
            locators: Vec::new(),
        });
    }
    v
}

// --------------------------------------------------------------------- engine

struct Cached<'a> {
    passages: Vec<Passage<'a>>,
    info: SourceInfo,
    sets: Option<Vec<HashSet<String>>>,
    norm_text: Option<String>,
}

struct Engine<'a> {
    src: &'a dyn Passages,
    opts: &'a AuditOptions,
    by_key: HashMap<String, usize>,
    cache: HashMap<usize, Cached<'a>>,
    tokens_cached: usize,
    text_cached: usize,
    work_left: u64,
    out_of_work: bool,
    notes: Vec<String>,
}

impl<'a> Engine<'a> {
    fn new(src: &'a dyn Passages, opts: &'a AuditOptions) -> Self {
        let mut by_key = HashMap::new();
        let n = src.source_count().min(opts.max_sources);
        for id in 0..n {
            if let Some(info) = src.source(id) {
                for k in std::iter::once(&info.key).chain(info.aliases.iter()) {
                    if !k.is_empty() {
                        by_key.entry(k.to_lowercase()).or_insert(id);
                    }
                }
            }
        }
        Engine {
            src,
            opts,
            by_key,
            cache: HashMap::new(),
            tokens_cached: 0,
            text_cached: 0,
            work_left: opts.work_budget,
            out_of_work: false,
            notes: Vec::new(),
        }
    }

    fn resolve(&self, key: &str) -> Option<usize> {
        self.by_key.get(&key.to_lowercase()).copied()
    }

    fn load(&mut self, id: usize) -> Option<&mut Cached<'a>> {
        if !self.cache.contains_key(&id) {
            let info = self.src.source(id)?;
            let passages = self.src.passages(id);
            self.cache.insert(
                id,
                Cached {
                    passages,
                    info,
                    sets: None,
                    norm_text: None,
                },
            );
        }
        self.cache.get_mut(&id)
    }

    /// Takes `n` units of work; false (and a note, once) when the budget is gone.
    fn spend(&mut self, n: u64) -> bool {
        if self.out_of_work {
            return false;
        }
        if n > self.work_left {
            self.out_of_work = true;
            self.work_left = 0;
            self.notes.push(
                "the work limit was reached (a very large library or text); the remaining checks were not run"
                    .into(),
            );
            return false;
        }
        self.work_left -= n;
        true
    }

    /// Coverage of the sentence's terms `q` in every passage of source `id`; `None` when out of work.
    fn scores(&mut self, id: usize, q: &[String]) -> Option<Vec<f32>> {
        let (npass, text_bytes, has_sets) = {
            let c = self.load(id)?;
            (
                c.passages.len() as u64,
                c.passages.iter().map(|p| p.text.len() as u64).sum::<u64>(),
                c.sets.is_some(),
            )
        };
        if q.is_empty() {
            return Some(vec![0.0; npass as usize]);
        }
        let cost = npass.saturating_mul(q.len() as u64);
        if !has_sets && !self.spend(text_bytes.saturating_add(cost)) {
            return None;
        }
        if has_sets && !self.spend(cost) {
            return None;
        }
        let budget_left = self.opts.cache_tokens.saturating_sub(self.tokens_cached);
        let c = self.cache.get_mut(&id)?;
        let frac = |hits: usize| hits as f32 / q.len() as f32;
        if let Some(sets) = &c.sets {
            return Some(
                sets.iter()
                    .map(|s| frac(q.iter().filter(|w| s.contains(*w)).count()))
                    .collect(),
            );
        }
        let sets: Vec<HashSet<String>> = c
            .passages
            .iter()
            .map(|p| token_strings(p.text).into_iter().collect())
            .collect();
        let out = sets
            .iter()
            .map(|s| frac(q.iter().filter(|w| s.contains(*w)).count()))
            .collect();
        let size: usize = sets.iter().map(HashSet::len).sum();
        if size <= budget_left {
            self.tokens_cached += size;
            c.sets = Some(sets);
        }
        Some(out)
    }

    /// Is the normalised quotation `nq` in source `id`? `None` when out of work.
    fn has_quote(&mut self, id: usize, nq: &str) -> Option<bool> {
        let text_budget_left = self.opts.cache_text_bytes.saturating_sub(self.text_cached);
        let have = self.load(id)?.norm_text.is_some();
        if !have {
            let bytes: u64 = self
                .cache
                .get(&id)?
                .passages
                .iter()
                .map(|p| p.text.len() as u64)
                .sum();
            if !self.spend(bytes.saturating_mul(2)) {
                return None;
            }
        } else if !self.spend(nq.len() as u64) {
            return None;
        }
        let c = self.cache.get_mut(&id)?;
        if let Some(t) = &c.norm_text {
            return Some(text::contains_words(t, nq));
        }
        let joined = c
            .passages
            .iter()
            .map(|p| p.text)
            .collect::<Vec<_>>()
            .join(" ");
        let t = norm(&joined);
        let found = text::contains_words(&t, nq);
        if t.len() <= text_budget_left {
            self.text_cached += t.len();
            c.norm_text = Some(t);
        }
        Some(found)
    }
}

fn cut(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut o: String = s.chars().take(max.saturating_sub(1)).collect();
        o.push('…');
        o
    }
}

fn distinct_terms(claim: &str, max: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut v = Vec::new();
    for t in token_strings(claim) {
        if v.len() >= max {
            break;
        }
        if seen.insert(t.clone()) {
            v.push(t);
        }
    }
    v
}

// ---------------------------------------------------------------------- audit

/// Audits a draft (Markdown, LaTeX or LyX) with pandoc-style citations against the library.
pub fn audit(lib: &dyn Passages, draft: &str, opts: &AuditOptions) -> AuditReport {
    run(lib, draft, opts, false)
}

/// One excerpt that was given to a model, numbered from 1 in the order given.
#[derive(Clone, Copy, Debug)]
pub struct Excerpt<'a> {
    /// Shown in findings (a reference label); may be empty.
    pub label: &'a str,
    /// Page of the excerpt, 0 when unknown.
    pub page: u32,
    pub text: &'a str,
}

/// Audits a model's answer that cites the excerpts it was given as `[1]`, `[2, 3]`. `[0]` and numbers beyond the
/// last excerpt are `unknown-source`; a quotation is looked for in the cited excerpts first, then in the others.
pub fn audit_answer(excerpts: &[Excerpt<'_>], answer: &str, opts: &AuditOptions) -> AuditReport {
    let mut lib = MemPassages::default();
    for (i, e) in excerpts.iter().enumerate() {
        lib.add(
            SourceInfo {
                key: (i + 1).to_string(),
                title: if e.label.is_empty() {
                    format!("[{}]", i + 1)
                } else {
                    sanitize::short(e.label, 120)
                },
                ..SourceInfo::default()
            },
            vec![(e.page, e.text.to_string())],
        );
    }
    run(&lib, answer, opts, true)
}

fn run(lib: &dyn Passages, input: &str, opts: &AuditOptions, numbered: bool) -> AuditReport {
    let mut rep = AuditReport::default();
    let mut eng = Engine::new(lib, opts);

    // 1. input: bounded, converted to Markdown, headings dropped
    let mut raw = input;
    if raw.len() > opts.max_input_bytes {
        let mut e = opts.max_input_bytes;
        while !raw.is_char_boundary(e) {
            e -= 1;
        }
        raw = &raw[..e];
        rep.notes.push(format!(
            "the text is longer than {} bytes; only the beginning was checked",
            opts.max_input_bytes
        ));
    }
    let md = if numbered {
        sanitize::text(raw)
    } else {
        normalize_markup(&sanitize::text(raw))
    };
    let md = text::strip_marks(&md);
    let body: String = md
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let parse: &dyn Fn(&str) -> Vec<Cite> = if numbered {
        &numbered_group
    } else {
        &parse_group
    };
    let (marked, groups, capped) = text::extract_groups(&body, parse, opts.max_groups);
    if capped {
        rep.notes.push(format!(
            "more than {} citation groups; the later ones were left unchecked",
            opts.max_groups
        ));
    }

    let (mut cites_done, mut quotes_done) = (0usize, 0usize);
    let mut sentence_limit_hit = false;
    let all_sentences = sentences(&marked);
    let total_sentences = all_sentences.len();
    'sentences: for (si, s) in all_sentences.into_iter().enumerate() {
        if si >= opts.max_sentences {
            sentence_limit_hit = true;
            break;
        }
        if rep.findings.len() >= opts.max_findings {
            rep.notes
                .push(format!("stopped after {} findings", opts.max_findings));
            break;
        }
        let mut cites: Vec<Cite> = Vec::new();
        for ch in s.chars() {
            if let Some(g) = text::mark_index(ch).and_then(|i| groups.get(i)) {
                cites.extend(g.iter().cloned());
            }
        }
        let plain_full: String = s
            .chars()
            .filter(|c| text::mark_index(*c).is_none())
            .collect::<String>()
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" ;", ";")
            .trim()
            .to_string();
        let plain: String = plain_full.chars().take(opts.max_sentence_chars).collect();
        let shown = cut(&plain_full, 1000);

        // 2. quotations must exist word for word
        for q in text::quotes(&plain, 20) {
            if quotes_done >= opts.max_quotes {
                if quotes_done == opts.max_quotes {
                    rep.notes.push(format!(
                        "only the first {} quotations were checked",
                        opts.max_quotes
                    ));
                    quotes_done += 1;
                }
                break;
            }
            quotes_done += 1;
            let nq = norm(&q);
            if nq.split(' ').count() < 3 {
                continue; // nothing but punctuation and one word: not a quotation we can look for
            }
            let cited_ids: Vec<usize> = cites.iter().filter_map(|c| eng.resolve(&c.key)).collect();
            let mut found_in = None;
            let mut limited = false;
            for &id in &cited_ids {
                match eng.has_quote(id, &nq) {
                    Some(true) => {
                        found_in = Some((id, true));
                        break;
                    }
                    Some(false) => {}
                    None => {
                        limited = true;
                        break;
                    }
                }
            }
            if found_in.is_none() && !limited {
                for id in 0..eng.src.source_count().min(opts.max_sources) {
                    if cited_ids.contains(&id) {
                        continue;
                    }
                    match eng.has_quote(id, &nq) {
                        Some(true) => {
                            found_in = Some((id, false));
                            break;
                        }
                        Some(false) => {}
                        None => {
                            limited = true;
                            break;
                        }
                    }
                }
            }
            let (verdict, note) = match found_in {
                Some((_, true)) => (Verdict::QuoteVerbatim, "found word-for-word in the cited source".to_string()),
                Some((id, false)) => {
                    let title = eng.load(id).map(|c| c.info.title.clone()).unwrap_or_default();
                    (
                        Verdict::QuoteElsewhere,
                        format!("not in the cited source, but found in {}", sanitize::short(&title, 120)),
                    )
                }
                None if limited => (
                    Verdict::QuoteNotFound,
                    "not found word-for-word in the part of the library that could be searched (the work limit was reached)"
                        .to_string(),
                ),
                None => (
                    Verdict::QuoteNotFound,
                    "not found word-for-word anywhere in the library".to_string(),
                ),
            };
            rep.findings.push(Finding {
                sentence: cut(&q, 1000),
                cite: cites.first().map(|c| display_cite(c, numbered)),
                page: cites.first().and_then(|c| c.page),
                verdict,
                support: if verdict == Verdict::QuoteVerbatim {
                    1.0
                } else {
                    0.0
                },
                note,
                passage: None,
            });
        }

        // 3. uncited statements
        if cites.is_empty() {
            if plain.split_whitespace().count() >= 14 && !plain.contains('?') {
                rep.findings.push(Finding {
                    sentence: shown,
                    cite: None,
                    page: None,
                    verdict: Verdict::Uncited,
                    support: 0.0,
                    note: "a long statement with no citation".into(),
                    passage: None,
                });
            }
            continue;
        }

        // 4. each citation against its source
        let terms = distinct_terms(&plain, opts.max_terms);
        let mut scored: HashMap<usize, Option<Vec<f32>>> = HashMap::new();
        for c in &cites {
            if cites_done >= opts.max_citations {
                rep.skipped_citations += 1;
                continue;
            }
            cites_done += 1;
            let label = display_cite(c, numbered);
            let base = |verdict: Verdict, support: f32, note: String| Finding {
                sentence: shown.clone(),
                cite: Some(label.clone()),
                page: c.page,
                verdict,
                support,
                note,
                passage: None,
            };
            let Some(id) = eng.resolve(&c.key) else {
                let note = if numbered {
                    let n = if c.key.len() > 9 {
                        "a number far beyond the last one".to_string()
                    } else {
                        c.key.clone()
                    };
                    format!(
                        "excerpt {n} does not exist (there are {})",
                        lib.source_count()
                    )
                } else {
                    format!(
                        "no document in the library matches the key '{}'",
                        sanitize::short(&c.key, 80)
                    )
                };
                rep.findings.push(base(Verdict::UnknownSource, 0.0, note));
                continue;
            };
            let Some(info) = eng.load(id).map(|x| x.info.clone()) else {
                rep.findings.push(base(
                    Verdict::UnknownSource,
                    0.0,
                    "the source could not be read".into(),
                ));
                continue;
            };
            // Locators: a bare number equal to the cited work's year is not a page.
            let locs: Vec<&Locator> = c
                .locators
                .iter()
                .filter(|l| {
                    !(l.label == "page"
                        && !l.explicit_label
                        && matches!(&l.from, Val::Num(n) if n.to_string() == info.year))
                })
                .collect();
            let ranges: Vec<(usize, usize)> = locs.iter().filter_map(|l| l.page_range()).collect();
            let other: Vec<String> = locs
                .iter()
                .filter(|l| l.page_range().is_none())
                .map(|l| l.describe())
                .collect();
            let page = ranges.first().map(|r| r.0);
            let f = |verdict: Verdict, support: f32, note: String| Finding {
                page,
                note: if other.is_empty() {
                    note.clone()
                } else {
                    format!("{note} [locator: {}]", other.join("; "))
                },
                ..base(verdict, support, note)
            };
            if let Some(bad) = locs.iter().find(|l| l.is_inverted()) {
                rep.findings.push(f(
                    Verdict::BadRange,
                    0.0,
                    format!("the interval {} ends before it starts", bad.describe()),
                ));
                continue;
            }
            if let Some(end) = ranges.iter().map(|r| r.1).max()
                && info.pages > 0
                && end > info.pages as usize
            {
                rep.findings.push(f(
                    Verdict::PageOutOfRange,
                    0.0,
                    format!(
                        "{} has {} pages",
                        sanitize::short(&info.title, 120),
                        info.pages
                    ),
                ));
                continue;
            }
            scored.entry(id).or_insert_with(|| eng.scores(id, &terms));
            let Some(Some(scores)) = scored.get(&id) else {
                rep.skipped_citations += 1;
                continue;
            };
            let Some(passages) = eng.cache.get(&id).map(|x| &x.passages) else {
                rep.skipped_citations += 1;
                continue;
            };
            // Text, EPUB and DOCX sources have no page numbers: a cited page cannot be checked there.
            let paged = passages.iter().any(|p| p.page > 0);
            let want_pages = !ranges.is_empty();
            let restrict = paged
                && ranges
                    .iter()
                    .any(|&(a, b)| b.min(a.saturating_add(200)) >= 1);
            let in_cited = |p: u32| {
                p > 0
                    && ranges.iter().any(|&(a, b)| {
                        (p as usize) >= a && (p as usize) <= b.min(a.saturating_add(200))
                    })
            };
            let (mut sup, mut sup_i) = (0.0f32, None::<usize>);
            let (mut any, mut anyp) = (0.0f32, 0u32);
            for (i, (p, &sc)) in passages.iter().zip(scores.iter()).enumerate() {
                if sc > any {
                    any = sc;
                    anyp = p.page;
                }
                if (!restrict || in_cited(p.page)) && (sup_i.is_none() || sc > sup) {
                    sup = sc;
                    sup_i = Some(i);
                }
            }
            let too_few = terms.len() < 3;
            let (verdict, mut note) = if passages.is_empty() {
                (
                    Verdict::Unsupported,
                    "this source has no readable text".to_string(),
                )
            } else if sup >= SUPPORTED && !too_few {
                (
                    Verdict::Supported,
                    format!(
                        "{:.0}% of the sentence's terms appear in the cited passage",
                        sup * 100.0
                    ),
                )
            } else if restrict && any >= SUPPORTED && !too_few && !in_cited(anyp) {
                (
                    Verdict::WrongPage,
                    format!(
                        "the cited page covers {:.0}%, but p. {} in the same source covers {:.0}% — probably p. {}",
                        sup * 100.0,
                        anyp,
                        any * 100.0,
                        anyp
                    ),
                )
            } else if too_few && sup >= WEAK {
                (
                    Verdict::Weak,
                    "the sentence has too few distinct terms to judge: read the passage"
                        .to_string(),
                )
            } else if sup >= WEAK {
                (
                    Verdict::Weak,
                    format!(
                        "only {:.0}% of the sentence's terms appear in the cited passage",
                        sup * 100.0
                    ),
                )
            } else if any >= WEAK {
                (
                    Verdict::Weak,
                    format!(
                        "the cited passage covers {:.0}%; the best match in the source is {:.0}% (p. {})",
                        sup * 100.0,
                        any * 100.0,
                        anyp
                    ),
                )
            } else {
                (
                    Verdict::Unsupported,
                    format!(
                        "only {:.0}% of the sentence's terms appear anywhere in this source",
                        any * 100.0
                    ),
                )
            };
            let top: Option<&str> = sup_i.and_then(|i| passages.get(i)).map(|p| p.text);
            if matches!(verdict, Verdict::Supported | Verdict::Weak)
                && let Some(t) = top
            {
                let t: String = t.chars().take(20_000).collect();
                let m = missing_numbers(&plain, &t);
                if !m.is_empty() {
                    let mut fin = f(
                        Verdict::NumberMismatch,
                        sup,
                        format!(
                            "the sentence states {} but the cited passage does not contain {}",
                            m.join(", "),
                            if m.len() == 1 { "it" } else { "them" }
                        ),
                    );
                    fin.passage = Some(t.chars().take(1500).collect());
                    rep.findings.push(fin);
                    continue;
                }
            }
            if want_pages && !paged {
                note.push_str(
                    " (this source has no page numbers: the cited page could not be checked)",
                );
            } else if want_pages
                && restrict
                && let Some(pn) = &info.page_note
            {
                note.push_str(&format!(" ({})", sanitize::short(pn, 200)));
            }
            let mut fin = f(verdict, sup, note);
            fin.passage = top.map(|t| t.chars().take(1500).collect());
            rep.findings.push(fin);
        }
        if eng.out_of_work {
            // nothing more can be checked; say how much was left
            let rest = total_sentences.saturating_sub(si + 1);
            if rest > 0 {
                rep.notes
                    .push(format!("{rest} later sentence(s) were not checked"));
            }
            break 'sentences;
        }
    }
    if sentence_limit_hit {
        rep.notes.push(format!(
            "only the first {} sentences were checked",
            opts.max_sentences
        ));
    }
    rep.notes.append(&mut eng.notes);
    rep.notes.dedup();
    rep
}

fn display_cite(c: &Cite, numbered: bool) -> String {
    if numbered {
        format!("[{}]", c.key)
    } else {
        sanitize::short(&c.key, 120)
    }
}

// ------------------------------------------------------------ semantic pass

/// What a local model says about one sentence and its passage.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Sem {
    Supports,
    Partial,
    Contradicts,
    Unrelated,
    Unclear,
}

pub const SYSTEM_SEMANTIC: &str = "You are a strict citation checker. You get ONE passage from a source and ONE sentence that cites it. \
Judge only from the passage. Answer with exactly one word: SUPPORTS (the passage says what the sentence says, even in other words), \
PARTIAL (only part of the sentence is in the passage), CONTRADICTS (the passage says the opposite or different facts) or UNRELATED.";

/// Reads the model's reply: the first word decides; anything else is `Unclear`.
pub fn parse_sem(text: &str) -> Sem {
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
    match first.as_str() {
        "SUPPORTS" | "SUPPORTED" => Sem::Supports,
        "PARTIAL" | "PARTIALLY" => Sem::Partial,
        "CONTRADICTS" | "CONTRADICTED" | "CONTRADICTION" => Sem::Contradicts,
        "UNRELATED" | "NOT" | "UNSUPPORTED" => Sem::Unrelated,
        _ => Sem::Unclear,
    }
}

/// Second opinion from a model on every cited sentence that has a passage, at most `max_calls` calls. It can rescue
/// a correct paraphrase (weak/unsupported becomes supported), catch a negation that term overlap accepts (supported
/// becomes contradicted) and flag overlap without meaning (supported becomes weak).
///
/// Quotations, unknown sources and page errors are left alone: they are decided by exact checks. A
/// `number-mismatch` is also exact, so a model's "supports" cannot lift it (the original let it); only
/// "contradicts" changes it. The model can be wrong: every changed finding says so in its note.
/// `judge(sentence, passage)` returns `None` when the model fails (that finding stays as it was).
/// Returns (calls answered, findings changed).
pub fn semantic_pass(
    findings: &mut [Finding],
    max_calls: usize,
    judge: &mut dyn FnMut(&str, &str) -> Option<Sem>,
) -> (usize, usize) {
    let (mut checked, mut changed, mut calls) = (0, 0, 0);
    for f in findings.iter_mut() {
        if !matches!(
            f.verdict,
            Verdict::Supported | Verdict::Weak | Verdict::Unsupported | Verdict::NumberMismatch
        ) {
            continue;
        }
        let Some(p) = f.passage.clone() else { continue };
        if calls >= max_calls {
            break;
        }
        calls += 1;
        let Some(sem) = judge(&f.sentence, &p) else {
            continue;
        };
        checked += 1;
        let before = f.verdict;
        let pct = f.support * 100.0;
        match sem {
            Sem::Supports => {
                if before == Verdict::NumberMismatch {
                    f.note.push_str(
                        " · [model] finds the wording supported, but the figure check stands",
                    );
                } else if before != Verdict::Supported {
                    f.verdict = Verdict::Supported;
                    f.note = format!(
                        "[model] paraphrase confirmed by the local model (term overlap was {pct:.0}%) — verify by reading"
                    );
                } else {
                    f.note.push_str(" · [model] agrees");
                }
            }
            Sem::Contradicts => {
                f.verdict = Verdict::Contradicted;
                f.note = format!(
                    "[model] the passage seems to say something different or the opposite (term overlap was {pct:.0}%)"
                );
            }
            Sem::Partial | Sem::Unrelated => {
                if before == Verdict::Supported {
                    f.verdict = Verdict::Weak;
                    f.note = format!(
                        "[model] words overlap ({pct:.0}%) but the model finds the passage {} the sentence",
                        if sem == Sem::Partial {
                            "only partly supporting"
                        } else {
                            "unrelated to"
                        }
                    );
                } else {
                    f.note.push_str(if sem == Sem::Partial {
                        " · [model] partial support"
                    } else {
                        " · [model] finds the passage unrelated"
                    });
                }
            }
            Sem::Unclear => {}
        }
        if f.verdict != before {
            changed += 1;
        }
    }
    (checked, changed)
}

// -------------------------------------------------------------------- report

/// A readable summary: counts, then every finding that needs a person.
pub fn audit_markdown(f: &[Finding]) -> String {
    let n = |v: Verdict| f.iter().filter(|x| x.verdict == v).count();
    let mut s = format!(
        "# Citation audit\n\n{} checks — supported {}, weak {}, contradicted {}, unsupported {}, wrong page {}, figure mismatch {}, unknown source {}, page out of range {}, bad page interval {}, uncited {}; quotations: verbatim {}, elsewhere {}, not found {}.\n\nThis checks that each citation can be traced to words in the source, not that the sentence is true. Paraphrases and negations are weak spots of a term-overlap check. Open the page for everything below.\n\n",
        f.len(),
        n(Verdict::Supported),
        n(Verdict::Weak),
        n(Verdict::Contradicted),
        n(Verdict::Unsupported),
        n(Verdict::WrongPage),
        n(Verdict::NumberMismatch),
        n(Verdict::UnknownSource),
        n(Verdict::PageOutOfRange),
        n(Verdict::BadRange),
        n(Verdict::Uncited),
        n(Verdict::QuoteVerbatim),
        n(Verdict::QuoteElsewhere),
        n(Verdict::QuoteNotFound)
    );
    for x in f.iter().filter(|x| x.is_problem()) {
        let c = x
            .cite
            .as_deref()
            .map(|c| {
                format!(
                    " [@{}{}]",
                    c,
                    x.page.map(|p| format!(", p. {p}")).unwrap_or_default()
                )
            })
            .unwrap_or_default();
        s.push_str(&format!(
            "- **{}**{} — {}\n  > {}\n",
            x.verdict,
            c,
            sanitize::line(&x.note),
            sanitize::line(&x.sentence)
        ));
    }
    if !f.iter().any(|x| x.is_problem()) {
        s.push_str("No problems found by this check.\n");
    }
    s
}

#[cfg(test)]
mod tests;
