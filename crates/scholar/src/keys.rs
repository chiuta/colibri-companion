//! Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf).
//! Citation keys: `surname` + `year`, then `a`, `b`, ... `z`, `aa` ... when that key is taken.
//! Collision-safe: a key is never handed out twice, comparing without regard to case (BibTeX
//! treats `Smith2019` and `smith2019` as one key). Every call is O(1) amortised, however many
//! entries share the same base (one author with hundreds of papers in a year, or thousands of
//! entries without author or year).

use crate::bib::Entry;
use std::collections::{HashMap, HashSet};

/// Longest key made by this module, in bytes.
pub const MAX_KEY_LEN: usize = 64;

/// The base of a generated key: ASCII-folded lower-case surname of the first author (`ref` when
/// there is none) followed by the year's alphanumerics. Never empty, never contains characters
/// that break `\cite{}` or a BibTeX key.
pub fn key_base(e: &Entry) -> String {
    let fam: String = e
        .people_for_key()
        .map(|f| {
            f.chars()
                .flat_map(|c| c.to_lowercase())
                .map(crate::bibtext::fold)
                .filter(|c| c.is_ascii_alphanumeric())
                .take(40)
                .collect()
        })
        .unwrap_or_default();
    let fam = if fam.is_empty() {
        "ref".to_string()
    } else {
        fam
    };
    let year: String = e
        .year
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(8)
        .collect();
    format!("{fam}{year}")
}

/// 0 -> a, 25 -> z, 26 -> aa, 27 -> ab ...
pub fn letter_suffix(mut n: usize) -> String {
    let mut s = Vec::new();
    loop {
        s.push((b'a' + (n % 26) as u8) as char);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    s.iter().rev().collect()
}

/// The keys already taken, and for each base the next suffix to try.
#[derive(Default, Debug, Clone)]
pub struct KeySet {
    used: HashSet<String>,
    next: HashMap<String, usize>,
}

impl KeySet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserves keys that exist already (the library being merged into).
    pub fn with_existing<'a>(keys: impl IntoIterator<Item = &'a str>) -> Self {
        let mut s = Self::new();
        for k in keys {
            s.used.insert(k.to_lowercase());
        }
        s
    }

    pub fn contains(&self, key: &str) -> bool {
        self.used.contains(&key.to_lowercase())
    }

    pub fn len(&self) -> usize {
        self.used.len()
    }

    pub fn is_empty(&self) -> bool {
        self.used.is_empty()
    }

    /// `explicit` (cleaned of characters that break a key) when it is non-empty and free, else the
    /// generated base with the first free letter suffix. The chosen key is reserved.
    pub fn unique(&mut self, e: &Entry, explicit: &str) -> String {
        let explicit = crate::bibtext::key(explicit);
        let key =
            if !explicit.is_empty() && explicit.len() <= MAX_KEY_LEN && !self.contains(&explicit) {
                explicit
            } else {
                let base = key_base(e);
                if !self.contains(&base) {
                    base
                } else {
                    let n = self.next.entry(base.to_lowercase()).or_insert(0);
                    loop {
                        let k = format!("{base}{}", letter_suffix(*n));
                        *n += 1;
                        if !self.used.contains(&k.to_lowercase()) {
                            break k;
                        }
                    }
                }
            };
        self.used.insert(key.to_lowercase());
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(author: &str, year: &str) -> Entry {
        Entry {
            author: author.into(),
            year: year.into(),
            ..Default::default()
        }
    }

    #[test]
    fn suffixes_run_a_to_z_then_aa() {
        assert_eq!(letter_suffix(0), "a");
        assert_eq!(letter_suffix(25), "z");
        assert_eq!(letter_suffix(26), "aa");
        assert_eq!(letter_suffix(27), "ab");
        assert_eq!(letter_suffix(26 + 26 * 26 - 1), "zz");
        assert_eq!(letter_suffix(26 + 26 * 26), "aaa");
    }

    #[test]
    fn keys_are_surname_year_then_letters_and_case_insensitive() {
        let mut ks = KeySet::new();
        let x = e("Smith, Jane", "2019");
        assert_eq!(ks.unique(&x, ""), "smith2019");
        assert_eq!(ks.unique(&x, ""), "smith2019a");
        assert_eq!(ks.unique(&x, ""), "smith2019b");
        assert_eq!(ks.unique(&x, "Smith2019"), "smith2019c"); // taken, any case
        assert_eq!(ks.unique(&x, "mine"), "mine");
        assert_eq!(ks.unique(&e("Ștefan, Ion", "2001"), ""), "stefan2001");
        assert_eq!(ks.unique(&e("", ""), ""), "ref");
        assert_eq!(ks.unique(&e("", ""), ""), "refa");
    }

    #[test]
    fn existing_keys_are_respected_and_hostile_parts_are_dropped() {
        let mut ks = KeySet::with_existing(["Smith2019"]);
        assert_eq!(ks.unique(&e("Smith, J", "2019"), ""), "smith2019a");
        let k = ks.unique(&e("Doe, J", "20{19}\n,x"), "bad key{,}");
        assert_eq!(k, "badkey");
        assert_eq!(key_base(&e("Doe, J", "20{19}\n,x")), "doe2019x");
        let long = "x".repeat(500);
        assert!(ks.unique(&e("Doe, J", "2000"), &long).len() <= MAX_KEY_LEN);
    }

    #[test]
    fn a_hundred_thousand_collisions_stay_fast_and_unique() {
        let x = e("Smith, Jane", "2019");
        let run = |n: usize| {
            let mut ks = KeySet::new();
            let mut seen = HashSet::new();
            let t = std::time::Instant::now();
            for _ in 0..n {
                assert!(seen.insert(ks.unique(&x, "")));
            }
            t.elapsed()
        };
        // Doubling the work must not much more than double the time (a quadratic search for a
        // free suffix would quadruple it); a ratio does not depend on how slow the machine is.
        let (small, big) = (run(50_000), run(100_000));
        assert!(
            big.as_secs_f64() < small.as_secs_f64() * 3.2 + 0.05,
            "not linear: {small:?} then {big:?}"
        );
    }
}
