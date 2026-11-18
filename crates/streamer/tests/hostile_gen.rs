//! Adversarial inputs: random text, mutated valid seeds, pathological shapes. Every parser must
//! return (an error or an empty result is fine), never panic and never take long.
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

pub fn iterations() -> usize {
    std::env::var("ADVERSARIAL_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500)
}

const NASTY: &[&str] = &[
    "\0",
    "\u{feff}",
    "\u{202e}",
    "\u{200b}",
    "é",
    "e\u{301}",
    "😀",
    "𝕏",
    "\u{10ffff}",
    "\r\n",
    "\\",
    "\"",
    "'",
    "{",
    "}",
    "[",
    "]",
    "(",
    ")",
    "<",
    ">",
    "&",
    "&#x110000;",
    "&#0;",
    "%",
    "$",
    "@",
    "#",
    "..",
    "/",
    "-",
    "--",
    "---",
    ":",
    ";",
    ",",
    "\t",
    "\u{1b}[31m",
    "[@",
    "[1]",
    "[0]",
    "[99999999999999999999]",
    "p. ",
    "pp. 3-",
    "9999999999999999999999",
    "-1",
    "1e999",
    "NaN",
    "null",
    "true",
    "@article{",
    "TY  - ",
    "ER  - ",
    "\\cite{",
    "\\begin{",
    "<?xml",
    "<![CDATA[",
    "<!--",
    "-->",
    "<a href=\"",
    "]]>",
];

pub fn random_text(r: &mut Rng) -> String {
    let n = r.below(400);
    let mut s = String::new();
    for _ in 0..n {
        match r.below(6) {
            0 => s.push_str(NASTY[r.below(NASTY.len())]),
            1 => s.push(char::from_u32(r.next_u64() as u32 % 0x30000).unwrap_or('x')),
            2 => s.push((b' ' + (r.next_u64() % 95) as u8) as char),
            3 => s.push('\n'),
            _ => s.push((b'a' + (r.next_u64() % 26) as u8) as char),
        }
    }
    s
}

pub fn mutate(seed: &str, r: &mut Rng) -> String {
    let chars: Vec<char> = seed.chars().collect();
    let mut v = chars.clone();
    for _ in 0..1 + r.below(8) {
        let at = r.below(v.len() + 1);
        match r.below(5) {
            0 if !v.is_empty() => {
                v.remove(at.min(v.len() - 1));
            }
            1 => {
                for c in NASTY[r.below(NASTY.len())].chars().rev() {
                    v.insert(at, c);
                }
            }
            2 if !v.is_empty() => {
                let k = r.below(v.len());
                let c = v[k];
                for _ in 0..1 + r.below(300) {
                    v.insert(at, c);
                }
            }
            3 => v.truncate(at),
            _ if v.len() > 2 => {
                let a = r.below(v.len());
                let b = r.below(v.len());
                v.swap(a, b);
            }
            _ => {}
        }
    }
    v.into_iter().collect()
}

pub fn deep(open: &str, close: &str, n: usize) -> String {
    format!("{}{}", open.repeat(n), close.repeat(n))
}

/// Runs `f` on many hostile inputs derived from `seeds`; fails on a panic or a slow call.
pub fn hammer(name: &str, seeds: &[&str], f: impl Fn(&str)) {
    let n = iterations();
    let mut r = Rng::new(0xC011B21 ^ name.len() as u64);
    let mut inputs: Vec<String> = vec![
        String::new(),
        "\0".repeat(1000),
        "a".repeat(200_000),
        deep("(", ")", 20_000),
        deep("[", "]", 20_000),
        deep("{", "}", 20_000),
        deep("<a>", "</a>", 10_000),
        "\n".repeat(100_000),
        "😀".repeat(50_000),
        "@a{".repeat(20_000),
        "[@x, p. 1] ".repeat(20_000),
        "1.".repeat(100_000),
    ];
    for s in seeds {
        inputs.push((*s).to_string());
    }
    for i in 0..n {
        let base = if seeds.is_empty() || i % 3 == 0 {
            random_text(&mut r)
        } else {
            mutate(seeds[r.below(seeds.len())], &mut r)
        };
        inputs.push(base);
    }
    let mut worst = Duration::ZERO;
    for (i, inp) in inputs.iter().enumerate() {
        let t = Instant::now();
        let res = catch_unwind(AssertUnwindSafe(|| f(inp)));
        let d = t.elapsed();
        worst = worst.max(d);
        if res.is_err() {
            let shown: String = inp.chars().take(300).collect();
            panic!(
                "{name}: panic on input #{i} ({} chars): {shown:?}",
                inp.chars().count()
            );
        }
        assert!(
            d < Duration::from_secs(5),
            "{name}: input #{i} ({} chars) took {d:?}",
            inp.chars().count()
        );
    }
    eprintln!("{name}: {} inputs, slowest {worst:?}", inputs.len());
}
