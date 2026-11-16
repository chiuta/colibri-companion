//! Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf) for the idea; the generators are new.
//! Hostile-input tests for every bibliography reader: random and mutated inputs must never panic
//! and must finish in bounded time. Deterministic (seeded xorshift), so a failure reproduces.

use crate::{bib, bibtex, csljson, dedup, ids, names, ris};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

const BIB: &str = r#"@string{tl = "Law Review"}
@article{smith2019, author = {Smith, John and {von Beethoven}, Ludwig and Doe, Jr., Jane},
  title = {Recognition of {Foreign} Qualifications: {\"u}ber}, journal = tl # " II",
  year = 2019, doi = {10.1000/xyz}, pages = {1--5}, month = jul, isbn = {978-92-95055-12-4}}
@book{b2, author = "Council of Europe", title = {Lisbon}, date = {1997-04-11}, crossref = {smith2019}}
@comment{x}
"#;
const RIS: &str = "TY  - JOUR\nAU  - Smith, Jane\nAU  - Doe, John\nTI  - On proof\nT2  - Law Rev\nPY  - 2019\nDO  - 10.1000/x\nSN  - 0317-8471\nER  - \n";
const CSL: &str = r#"[{"id":"a","type":"book","title":"T","author":[{"family":"X","given":"Y"},{"literal":"Org"}],"issued":{"date-parts":[[2001,2]]},"DOI":"10.1/x","ISBN":"9789295055124"}]"#;

fn mutate(rng: &mut Rng, seed: &str) -> String {
    let mut c: Vec<char> = seed.chars().collect();
    let junk: Vec<char> = "{}()\"\\#=,@%$&~^'`\n\r\t ab1Z\u{0}\u{202e}\u{feff}\u{7f}é東😀"
        .chars()
        .collect();
    for _ in 0..1 + rng.below(12) {
        if c.is_empty() {
            c.push('@');
        }
        let i = rng.below(c.len());
        match rng.below(5) {
            0 => c[i] = junk[rng.below(junk.len())],
            1 => c.insert(i, junk[rng.below(junk.len())]),
            2 => {
                c.remove(i);
            }
            3 => {
                let j = (i + rng.below(40)).min(c.len());
                let dup: Vec<char> = c[i..j].to_vec();
                let at = rng.below(c.len());
                for (k, x) in dup.into_iter().enumerate() {
                    c.insert(at + k, x);
                }
            }
            _ => c.truncate(i),
        }
    }
    c.into_iter().collect()
}

fn every_reader(t: &str) {
    let _ = bibtex::read(t);
    let (es, _) = bib::parse_with_warnings(t);
    let _ = bib::parse_any("x.bib", t);
    let _ = ris::parse(t);
    let _ = csljson::parse(t);
    let _ = names::parse_names(t);
    let _ = ids::extract(t);
    let _ = ids::normalize_doi(t);
    let _ = ids::check_isbn(t);
    let _ = ids::check_issn(t);
    let _ = ids::check_orcid(t);
    for e in es.iter().take(50) {
        let _ = (
            e.cite(),
            e.reference(),
            e.to_ris(),
            e.to_csl(),
            e.to_bibtex(),
        );
    }
    let _ = dedup::analyze(&es);
}

#[test]
fn mutated_inputs_never_panic() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for round in 0..1500 {
        let seed = [BIB, RIS, CSL][round % 3];
        every_reader(&mutate(&mut rng, seed));
    }
}

#[test]
fn random_characters_never_panic() {
    let mut rng = Rng(42);
    let alphabet: Vec<char> = "@{}()\"\\#=, \n.-:/0123456789xXabc\u{0}é東\u{202e}"
        .chars()
        .collect();
    for _ in 0..800 {
        let n = rng.below(400);
        let t: String = (0..n)
            .map(|_| alphabet[rng.below(alphabet.len())])
            .collect();
        every_reader(&t);
    }
}

#[test]
fn degenerate_shapes_finish_quickly() {
    let t0 = Instant::now();
    let big = 300_000;
    let shapes = [
        "@a{".repeat(big / 3), // thousands of unclosed entries
        "{".repeat(big),       // brace storm
        "}".repeat(big),
        format!("@article{{k, title = {}}}", "{".repeat(big)),
        format!("@article{{k, title = \"{}\"}}", "\\".repeat(big)),
        "TY  - JOUR\n".repeat(big / 12), // one record per line, no ER
        "AU  - ".repeat(big / 6),
        "[".repeat(big), // JSON depth bomb
        format!("[{}]", "{\"id\":\"a\"},".repeat(big / 14)),
        "10.1000/".repeat(big / 8), // DOI-looking soup
        "and ".repeat(big / 4),
        ", ".repeat(big / 2),
        "\u{0301}".repeat(big), // combining marks
    ];
    for s in &shapes {
        let start = Instant::now();
        every_reader(s);
        if start.elapsed().as_millis() > 500 {
            eprintln!(
                "slow shape {:?}: {:?}",
                s.chars().take(12).collect::<String>(),
                start.elapsed()
            );
        }
        assert!(
            start.elapsed().as_secs() < 20,
            "too slow ({:?}) on a shape starting {:?}",
            start.elapsed(),
            s.chars().take(20).collect::<String>()
        );
    }
    eprintln!("degenerate shapes: {:?}", t0.elapsed());
}

#[test]
fn abbreviation_amplification_is_cut() {
    let big = "x".repeat(200_000);
    let uses = "a # ".repeat(2000);
    let t = format!("@string{{a = \"{big}\"}}\n@misc{{k, title = {uses} a}}");
    let f = bibtex::read(&t);
    let title = f.entries[0].get("title").unwrap();
    assert!(
        title.len() <= bibtex::MAX_VALUE_CHARS + big.len(),
        "{}",
        title.len()
    );
    assert!(f.warnings.iter().any(|w| w.contains("cut at")));
}

#[test]
fn warnings_are_bounded() {
    let t: String = (0..50_000)
        .map(|i| format!("@misc{{k{i}, title = undefined{i}}}\n"))
        .collect();
    let f = bibtex::read(&t);
    assert!(f.warnings.len() <= bibtex::MAX_WARNINGS + 1);
    assert!(f.warnings.last().unwrap().contains("more warnings"));
}

#[test]
fn many_entries_stay_fast() {
    let t: String = (0..40_000)
        .map(|i| format!("@article{{k{i}, author={{A{i}, B}}, title={{Title number {i} about things}}, year={{{}}}, doi={{10.1000/{i}}}}}\n", 1990 + i % 30))
        .collect();
    let start = Instant::now();
    let (es, _) = bib::parse_with_warnings(&t);
    assert_eq!(es.len(), 40_000);
    let rep = dedup::analyze(&es);
    assert!(rep.groups.is_empty());
    assert!(start.elapsed().as_secs() < 30, "{:?}", start.elapsed());
}
