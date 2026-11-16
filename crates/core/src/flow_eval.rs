//! Testing an automation: run it on sample inputs several times and report how often the output
//! meets simple expectations, with an honest interval instead of a bare percentage.
//! Nothing is saved or sent during a test.

use crate::flows::{Env, Flow, Outcome, run};

/// One sample: what to feed the automation, and what its output must (or must not) contain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    pub input: String,
    pub must: Vec<String>,
    pub must_not: Vec<String>,
}

pub const MAX_CASES: usize = 20;
pub const MAX_REPEATS: usize = 10;

/// One case per line: `input ::: word ; other word ; !forbidden`. Lines starting with `#` and
/// empty lines are skipped. Without `:::` the line is only an input: it passes when the flow ends well.
pub fn parse_cases(text: &str) -> Vec<Case> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .take(MAX_CASES)
        .map(|l| {
            let (input, rest) = l.split_once(":::").unwrap_or((l, ""));
            let mut must = Vec::new();
            let mut must_not = Vec::new();
            for part in rest.split(';').map(str::trim).filter(|p| !p.is_empty()) {
                match part.strip_prefix('!') {
                    Some(w) if !w.trim().is_empty() => must_not.push(w.trim().to_lowercase()),
                    Some(_) => {}
                    None => must.push(part.to_lowercase()),
                }
            }
            Case {
                input: input.trim().to_string(),
                must,
                must_not,
            }
        })
        .collect()
}

pub fn passes(case: &Case, o: &Outcome) -> bool {
    if !o.ok {
        return false;
    }
    let out = o.output.to_lowercase();
    case.must.iter().all(|w| out.contains(w)) && !case.must_not.iter().any(|w| out.contains(w))
}

/// 95% Wilson score interval for `pass` successes in `n` tries, as fractions in 0..=1.
pub fn wilson(pass: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.96_f64;
    let n = n as f64;
    let p = pass as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub tries: usize,
    pub passed: usize,
    /// Per case: (passed, tries).
    pub per_case: Vec<(usize, usize)>,
    pub low: f64,
    pub high: f64,
    pub stopped: bool,
}

/// An environment for tests: reading and asking go through, writing and sending do not.
pub struct Dry<'a> {
    pub inner: &'a mut dyn Env,
}

impl Env for Dry<'_> {
    fn fetch(&mut self, url: &str) -> Result<String, String> {
        self.inner.fetch(url)
    }
    fn read_folder(&mut self, path: &str) -> Result<String, String> {
        self.inner.read_folder(path)
    }
    fn read_file(&mut self, path: &str) -> Result<String, String> {
        self.inner.read_file(path)
    }
    fn repo_file(&mut self, service: &str, repo: &str, path: &str) -> Result<String, String> {
        self.inner.repo_file(service, repo, path)
    }
    fn ask(&mut self, prompt: &str) -> Result<String, String> {
        self.inner.ask(prompt)
    }
    fn save(&mut self, name: &str, _text: &str) -> Result<String, String> {
        Ok(format!("(test: {name} not saved)"))
    }
    fn webhook(&mut self, _url: &str, _text: &str) -> Result<(), String> {
        Ok(())
    }
    fn notify(&mut self, _text: &str) {}
    fn approve(&mut self, _what: &str) -> bool {
        false
    }
    fn log(&mut self, _line: &str) {}
    fn cancelled(&self) -> bool {
        self.inner.cancelled()
    }
}

/// Runs every case `repeats` times. `progress(done, total)` is called after each run.
pub fn evaluate(
    flow: &Flow,
    cases: &[Case],
    repeats: usize,
    env: &mut dyn Env,
    mut progress: impl FnMut(usize, usize),
) -> Report {
    let repeats = repeats.clamp(1, MAX_REPEATS);
    let total = cases.len() * repeats;
    let mut per_case = vec![(0usize, 0usize); cases.len()];
    let (mut done, mut passed, mut stopped) = (0, 0, false);
    'all: for (i, case) in cases.iter().enumerate() {
        for _ in 0..repeats {
            if env.cancelled() {
                stopped = true;
                break 'all;
            }
            let mut dry = Dry { inner: env };
            let o = run(flow, &case.input, false, &mut dry);
            per_case[i].1 += 1;
            if passes(case, &o) {
                per_case[i].0 += 1;
                passed += 1;
            }
            done += 1;
            progress(done, total);
        }
    }
    let (low, high) = wilson(passed, done);
    Report {
        tries: done,
        passed,
        per_case,
        low,
        high,
        stopped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flows::Step;

    struct Mock {
        calls: usize,
        saved: bool,
    }
    impl Env for Mock {
        fn fetch(&mut self, _: &str) -> Result<String, String> {
            Ok(String::new())
        }
        fn read_folder(&mut self, _: &str) -> Result<String, String> {
            Ok(String::new())
        }
        fn read_file(&mut self, _: &str) -> Result<String, String> {
            Ok(String::new())
        }
        fn repo_file(&mut self, _: &str, _: &str, _: &str) -> Result<String, String> {
            Ok(String::new())
        }
        fn ask(&mut self, p: &str) -> Result<String, String> {
            self.calls += 1;
            // every second answer is wrong
            Ok(if self.calls % 2 == 1 {
                format!("Hello {p}")
            } else {
                "nope".into()
            })
        }
        fn save(&mut self, _: &str, _: &str) -> Result<String, String> {
            self.saved = true;
            Ok("x".into())
        }
        fn webhook(&mut self, _: &str, _: &str) -> Result<(), String> {
            Ok(())
        }
        fn notify(&mut self, _: &str) {}
        fn approve(&mut self, _: &str) -> bool {
            true
        }
        fn log(&mut self, _: &str) {}
        fn cancelled(&self) -> bool {
            false
        }
    }

    fn flow() -> Flow {
        Flow {
            id: "t".into(),
            name: "t".into(),
            steps: vec![
                Step::Ask {
                    prompt: "{{input}}".into(),
                },
                Step::Save {
                    name: "o.md".into(),
                },
            ],
            trigger: Default::default(),
            enabled: false,
            input: String::new(),
            last_run: 0,
            builtin: false,
        }
    }

    #[test]
    fn cases_are_parsed_with_expectations() {
        let c = parse_cases("# c\nworld ::: Hello ; !nope\n\nplain");
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].input, "world");
        assert_eq!(c[0].must, vec!["hello"]);
        assert_eq!(c[0].must_not, vec!["nope"]);
        assert!(c[1].must.is_empty());
    }

    #[test]
    fn wilson_is_honest_about_small_samples() {
        let (lo, hi) = wilson(5, 5);
        assert!(lo > 0.5 && lo < 0.6, "{lo}");
        assert!((hi - 1.0).abs() < 1e-9);
        let (lo, hi) = wilson(50, 100);
        assert!(lo > 0.39 && hi < 0.61);
        assert_eq!(wilson(0, 0), (0.0, 1.0));
    }

    #[test]
    fn a_test_run_counts_passes_and_never_saves() {
        let mut env = Mock {
            calls: 0,
            saved: false,
        };
        let cases = parse_cases("world ::: hello");
        let r = evaluate(&flow(), &cases, 4, &mut env, |_, _| {});
        assert_eq!(r.tries, 4);
        assert_eq!(r.passed, 2);
        assert_eq!(r.per_case, vec![(2, 4)]);
        assert!(!env.saved, "tests must not write files");
        assert!(r.low < 0.5 && r.high > 0.5);
    }
}
