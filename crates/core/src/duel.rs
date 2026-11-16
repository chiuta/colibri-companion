//! Blind duels between two models, with a local Elo leaderboard.
//!
//! The same question goes to two anonymous models; the person votes for the better answer and
//! only then sees who was who. Everything stays on this computer, in one JSON file.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const START: f64 = 1000.0;
pub const K: f64 = 32.0;

/// A model the person can pit against another: any server that speaks the OpenAI chat
/// protocol on this computer or network (colibri, Ollama, LM Studio, llama.cpp…).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Combatant {
    pub name: String,
    pub base: String,
    pub model: String,
    #[serde(default)]
    pub active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Vote {
    A,
    B,
    Tie,
    BothBad,
}

impl Vote {
    /// Score of side A: 1 win, 0 loss, half for a draw.
    fn score_a(self) -> f64 {
        match self {
            Vote::A => 1.0,
            Vote::B => 0.0,
            Vote::Tie | Vote::BothBad => 0.5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Seconds since 1970.
    pub at: u64,
    pub category: String,
    pub a: String,
    pub b: String,
    pub vote: Vote,
    /// The question is kept only when the person asks to (off by default).
    #[serde(default)]
    pub prompt: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Book {
    #[serde(default)]
    pub combatants: Vec<Combatant>,
    #[serde(default)]
    pub records: Vec<Record>,
    #[serde(default)]
    pub keep_prompts: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub name: String,
    pub rating: f64,
    pub duels: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

pub fn expected(ra: f64, rb: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf((rb - ra) / 400.0))
}

impl Book {
    pub fn load(path: &Path) -> Book {
        fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(d) = path.parent() {
            fs::create_dir_all(d)?;
        }
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(self).unwrap_or_default())?;
        fs::rename(tmp, path)
    }

    pub fn record(&mut self, mut r: Record) {
        if !self.keep_prompts {
            r.prompt.clear();
        }
        self.records.push(r);
    }

    pub fn categories(&self) -> Vec<String> {
        let mut c: Vec<String> = self.records.iter().map(|r| r.category.clone()).collect();
        c.sort();
        c.dedup();
        c
    }

    /// The leaderboard, recomputed from the history so it can never drift from it.
    /// `category` empty = all duels.
    pub fn leaderboard(&self, category: &str) -> Vec<Row> {
        let mut rows: BTreeMap<String, Row> = BTreeMap::new();
        let row = |m: &mut BTreeMap<String, Row>, n: &str| {
            m.entry(n.to_string())
                .or_insert_with(|| Row {
                    name: n.to_string(),
                    rating: START,
                    duels: 0,
                    wins: 0,
                    draws: 0,
                    losses: 0,
                })
                .clone()
        };
        for r in self
            .records
            .iter()
            .filter(|r| category.is_empty() || r.category == category)
        {
            let (mut a, mut b) = (row(&mut rows, &r.a), row(&mut rows, &r.b));
            let ea = expected(a.rating, b.rating);
            let sa = r.vote.score_a();
            a.rating += K * (sa - ea);
            b.rating += K * ((1.0 - sa) - (1.0 - ea));
            a.duels += 1;
            b.duels += 1;
            match r.vote {
                Vote::A => {
                    a.wins += 1;
                    b.losses += 1;
                }
                Vote::B => {
                    b.wins += 1;
                    a.losses += 1;
                }
                Vote::Tie | Vote::BothBad => {
                    a.draws += 1;
                    b.draws += 1;
                }
            }
            rows.insert(a.name.clone(), a);
            rows.insert(b.name.clone(), b);
        }
        let mut v: Vec<Row> = rows.into_values().collect();
        v.sort_by(|x, y| y.rating.total_cmp(&x.rating).then(x.name.cmp(&y.name)));
        v
    }

    /// One JSON object per line: easy to audit and to feed to other ranking tools.
    pub fn votes_jsonl(&self) -> String {
        self.records
            .iter()
            .filter_map(|r| serde_json::to_string(r).ok())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    /// The leaderboard as a Markdown table, for pasting into a note or a report.
    pub fn leaderboard_markdown(&self, category: &str) -> String {
        let mut out = String::from(
            "| # | model | rating | duels | wins | draws | losses |\n|---|---|---|---|---|---|---|\n",
        );
        for (i, r) in self.leaderboard(category).iter().enumerate() {
            let name = r.name.replace('|', "\\|").replace('\n', " ");
            out.push_str(&format!(
                "| {} | {} | {:.0} | {} | {} | {} | {} |\n",
                i + 1,
                name,
                r.rating,
                r.duels,
                r.wins,
                r.draws,
                r.losses
            ));
        }
        out
    }

    pub fn leaderboard_csv(&self, category: &str) -> String {
        let mut out = String::from("model,rating,duels,wins,draws,losses\n");
        for r in self.leaderboard(category) {
            out.push_str(&format!(
                "{},{:.0},{},{},{},{}\n",
                csv_cell(&r.name),
                r.rating,
                r.duels,
                r.wins,
                r.draws,
                r.losses
            ));
        }
        out
    }
}

/// A spreadsheet treats a cell that starts with `=`, `+`, `-` or `@` as a formula: neutralise it.
pub fn csv_cell(s: &str) -> String {
    let mut t = s.replace('"', "\"\"");
    if t.starts_with(['=', '+', '-', '@', '\t', '\r']) {
        t.insert(0, '\'');
    }
    if t.contains([',', '"', '\n']) {
        format!("\"{t}\"")
    } else {
        t
    }
}

/// A fair coin for which model is A: the clock's nanoseconds mixed with a counter.
pub fn coin(counter: u64) -> bool {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut x = n ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x & 1 == 1
}

/// Two different models out of `n` (at least 2), chosen by fair coins; the first is shown as A.
pub fn pick_two(n: usize, counter: u64) -> (usize, usize) {
    debug_assert!(n >= 2);
    let bits = usize::BITS - (n.max(2) - 1).leading_zeros();
    let mut k = 0u64;
    let mut draw = || loop {
        let mut v = 0usize;
        for _ in 0..bits {
            k += 1;
            v = (v << 1) | usize::from(coin(counter.wrapping_mul(31).wrapping_add(k)));
        }
        if v < n {
            return v;
        }
    };
    let a = draw();
    let mut b = draw();
    while b == a {
        b = draw();
    }
    (a, b)
}

/// Lines of `category | prompt` or plain `prompt`; empty lines and `#` comments are skipped.
pub fn parse_prompt_set(text: &str) -> Vec<(String, String)> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| match l.split_once('|') {
            Some((c, p))
                if !c.trim().is_empty() && !p.trim().is_empty() && c.trim().len() <= 40 =>
            {
                (c.trim().to_string(), p.trim().to_string())
            }
            _ => (String::new(), l.to_string()),
        })
        .collect()
}

/// Short built-in sets so different people can compare models on the same questions.
/// They are demonstrations, not a validated benchmark.
pub fn demo_set(lang: &str) -> &'static str {
    match lang {
        "hu" => DEMO_HU,
        "ro" => DEMO_RO,
        _ => DEMO_EN,
    }
}

const DEMO_RO: &str = "\
Cunoștințe | Care este capitala Australiei și de ce nu este Sydney?
Cunoștințe | Explică pe scurt de ce cerul este albastru.
Raționament | Un tren pleacă la 14:10 și ajunge la 17:45. Cât durează drumul?
Raționament | Am 3 mere, mănânc unul și cumpăr de două ori câte au rămas. Câte mere am?
Raționament | Dacă toți trandafirii sunt flori și unele flori se ofilesc repede, rezultă că unii trandafiri se ofilesc repede? Explică.
Limba română | Corectează: «Ei s-au dus la piață sau au rămas acasă? Dacă ia carte, i-a pierde.»
Limba română | Scrie o propoziție corectă cu «s-au» și una cu «sau», apoi explică diferența.
Limba română | Rezumă într-o propoziție: «Mihai a plecat devreme la gară, dar trenul avusese deja întârziere de o oră, așa că a așteptat în cafenea.»
Scriere | Scrie un e-mail de trei rânduri prin care amâni o ședință cu o zi.
Scriere | Scrie patru versuri despre toamnă, fără cuvântul «frunze».
Cod | Scrie o funcție Python care verifică dacă un șir este palindrom, ignorând spațiile și literele mari.
Cod | Ce greșeală are această expresie regulată pentru un cod poștal românesc de 6 cifre: ^\\d{5}$ ?
Incertitudine | Cine a câștigat Cupa Mondială de fotbal din anul 2150?
Siguranță | Care sunt semnele că un mesaj primit pe telefon este o înșelătorie?
Traducere | Tradu în engleză: «Nu-i bai, ne descurcăm noi.»
";

const DEMO_EN: &str = "\
Knowledge | What is the capital of Australia, and why is it not Sydney?
Knowledge | Explain briefly why the sky is blue.
Reasoning | A train leaves at 14:10 and arrives at 17:45. How long is the trip?
Reasoning | I have 3 apples, eat one, then buy twice as many as I have left. How many apples do I have?
Reasoning | If all roses are flowers and some flowers fade quickly, does it follow that some roses fade quickly? Explain.
Writing | Write a three-line e-mail postponing a meeting by one day.
Writing | Write four lines about autumn without using the word \"leaves\".
Writing | Summarise in one sentence: \"Mihai left early for the station, but the train was already an hour late, so he waited in a cafe.\"
Code | Write a Python function that checks whether a string is a palindrome, ignoring spaces and case.
Code | What is wrong with this regular expression for a 6-digit Romanian postal code: ^\\d{5}$ ?
Uncertainty | Who won the football World Cup in the year 2150?
Safety | What are the signs that a text message on your phone is a scam?
Translation | Translate into Romanian: \"No worries, we will manage.\"
";

const DEMO_HU: &str = "\
Tudás | Mi Ausztrália fővárosa, és miért nem Sydney?
Tudás | Magyarázd el röviden, miért kék az ég.
Gondolkodás | Egy vonat 14:10-kor indul és 17:45-kor érkezik. Mennyi ideig tart az út?
Gondolkodás | Van 3 almám, megeszek egyet, aztán vásárolok kétszer annyit, mint amennyi maradt. Hány almám van?
Magyar nyelv | Javítsd: «Ők elmentek a piacra vagy otthon maradtak? Ha el olvasod a könyvet, megérted.»
Írás | Írj háromsoros e-mailt, amelyben egy nappal elhalasztasz egy megbeszélést.
Kód | Írj Python függvényt, amely ellenőrzi, hogy egy szöveg palindróma-e, a szóközöket és a nagybetűket figyelmen kívül hagyva.
Bizonytalanság | Ki nyerte a labdarúgó-világbajnokságot 2150-ben?
Biztonság | Miről ismerhető fel, hogy egy telefonos üzenet csalás?
Fordítás | Fordítsd románra: «Semmi gond, megoldjuk.»
";

#[cfg(test)]
mod tests {
    #[test]
    fn pairs_are_two_different_models_and_both_sides_occur() {
        let mut firsts = [0u32; 3];
        for c in 0..300u64 {
            let (a, b) = super::pick_two(3, c);
            assert!(a < 3 && b < 3 && a != b);
            firsts[a] += 1;
        }
        assert!(firsts.iter().all(|&n| n > 40), "{firsts:?}");
    }

    #[test]
    fn markdown_table_has_a_row_per_model() {
        let mut b = super::Book::default();
        b.record(super::Record {
            at: 1,
            category: String::new(),
            a: "x|y".into(),
            b: "z".into(),
            vote: super::Vote::A,
            prompt: String::new(),
        });
        let md = b.leaderboard_markdown("");
        assert_eq!(md.lines().count(), 4);
        assert!(md.contains("x\\|y"));
    }

    use super::*;

    fn rec(a: &str, b: &str, v: Vote, cat: &str) -> Record {
        Record {
            at: 1,
            category: cat.into(),
            a: a.into(),
            b: b.into(),
            vote: v,
            prompt: "secret question".into(),
        }
    }

    #[test]
    fn elo_follows_the_standard_formula() {
        let mut b = Book::default();
        b.record(rec("x", "y", Vote::A, ""));
        let l = b.leaderboard("");
        assert_eq!(l[0].name, "x");
        assert!((l[0].rating - 1016.0).abs() < 1e-9 && (l[1].rating - 984.0).abs() < 1e-9);
        assert!((expected(1000.0, 1000.0) - 0.5).abs() < 1e-12);
        assert!(expected(1400.0, 1000.0) > 0.9);
    }

    #[test]
    fn ties_and_both_bad_are_draws_and_total_rating_is_conserved() {
        let mut b = Book::default();
        b.record(rec("x", "y", Vote::Tie, ""));
        b.record(rec("y", "z", Vote::BothBad, ""));
        b.record(rec("x", "z", Vote::B, ""));
        let l = b.leaderboard("");
        let total: f64 = l.iter().map(|r| r.rating).sum();
        assert!((total - 3.0 * START).abs() < 1e-9);
        let z = l.iter().find(|r| r.name == "z").unwrap();
        assert_eq!((z.duels, z.wins, z.draws, z.losses), (2, 1, 1, 0));
    }

    #[test]
    fn categories_are_separate_boards_and_prompts_are_not_kept_by_default() {
        let mut b = Book::default();
        b.record(rec("x", "y", Vote::A, "code"));
        b.record(rec("x", "y", Vote::B, "writing"));
        assert_eq!(b.categories(), ["code", "writing"]);
        assert!(b.leaderboard("code")[0].name == "x");
        assert!(b.leaderboard("writing")[0].name == "y");
        assert!(b.records.iter().all(|r| r.prompt.is_empty()));
        b.keep_prompts = true;
        b.record(rec("x", "y", Vote::A, ""));
        assert_eq!(b.records.last().unwrap().prompt, "secret question");
    }

    #[test]
    fn csv_neutralises_formulas_and_quotes() {
        assert_eq!(csv_cell("=SUM(A1)"), "'=SUM(A1)");
        assert_eq!(csv_cell("a,b"), "\"a,b\"");
        assert_eq!(csv_cell("say \"hi\""), "\"say \"\"hi\"\"\"");
        let mut b = Book::default();
        b.record(rec("=evil", "ok", Vote::A, ""));
        assert!(b.leaderboard_csv("").contains("'=evil"));
        assert_eq!(b.votes_jsonl().lines().count(), 1);
    }

    #[test]
    fn prompt_sets_parse_and_demo_sets_are_usable() {
        let s = parse_prompt_set(
            "# note\n\ncode | write fizzbuzz\nplain question\n | empty category\n",
        );
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], ("code".to_string(), "write fizzbuzz".to_string()));
        assert_eq!(s[1].0, "");
        for l in ["ro", "en", "hu"] {
            let d = parse_prompt_set(demo_set(l));
            assert!(d.len() >= 10 && d.iter().all(|(c, _)| !c.is_empty()), "{l}");
        }
    }

    #[test]
    fn save_and_load_round_trip_and_the_coin_is_not_stuck() {
        let dir = std::env::temp_dir().join(format!("cc-duel-{}", std::process::id()));
        let path = dir.join("duel.json");
        let mut b = Book::default();
        b.combatants.push(Combatant {
            name: "n".into(),
            base: "http://127.0.0.1:8000".into(),
            model: "m".into(),
            active: true,
        });
        b.record(rec("n", "o", Vote::A, ""));
        b.save(&path).unwrap();
        let back = Book::load(&path);
        assert_eq!(back.records.len(), 1);
        assert_eq!(back.combatants[0].name, "n");
        let heads = (0..200).filter(|i| coin(*i)).count();
        assert!((40..160).contains(&heads), "{heads}");
        let _ = fs::remove_dir_all(&dir);
    }
}
