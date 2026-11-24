//! Tiny command-line parser: `--name value`, `--name=value`, bare flags and positionals.

/// Options that never take a value.
pub const FLAGS: &[&str] = &[
    "raw",
    "no-mmap",
    "quiet",
    "help",
    "log-prompts",
    "no-log",
    "reproducible",
    "stream-layers",
    "calibrate",
    "json",
    "no-governor",
    "governor",
    "discover",
    "verbose",
    "no-lock",
    "keep-steps",
    "leaves",
    "compress",
    "wire-f16",
    "lookahead",
    "self-spec",
    "all",
    "dry-run",
    "explain",
    "no-attention",
    "no-dedup",
    "no-discovery",
    "discovery",
    "overwrite",
    "allow-admin",
];

pub struct Args {
    pub pos: Vec<String>,
    pub opts: Vec<(String, Option<String>)>,
}

impl Args {
    pub fn parse(v: &[String]) -> Args {
        let (mut pos, mut opts) = (vec![], vec![]);
        let mut i = 0;
        while i < v.len() {
            if let Some(name) = v[i].strip_prefix("--") {
                if let Some((k, val)) = name.split_once('=') {
                    opts.push((k.to_string(), Some(val.to_string())));
                } else if FLAGS.contains(&name) {
                    opts.push((name.to_string(), None));
                } else if i + 1 < v.len() {
                    opts.push((name.to_string(), Some(v[i + 1].clone())));
                    i += 1;
                } else {
                    opts.push((name.to_string(), None));
                }
            } else {
                pos.push(v[i].clone());
            }
            i += 1;
        }
        Args { pos, opts }
    }
    pub fn flag(&self, k: &str) -> bool {
        self.opts.iter().any(|(n, _)| n == k)
    }
    pub fn all(&self, k: &str) -> Vec<&str> {
        self.opts
            .iter()
            .filter(|(n, _)| n == k)
            .filter_map(|(_, v)| v.as_deref())
            .collect()
    }
    pub fn get(&self, k: &str) -> Option<&str> {
        self.opts
            .iter()
            .rev()
            .find(|(n, _)| n == k)
            .and_then(|(_, v)| v.as_deref())
    }
    /// A secret from `--NAME VALUE`, `--FILE PATH` (first line of the file; keeps it out of `ps` and shell history)
    /// or the environment variable `env`, in that order of preference. Giving both the value and the file is an error.
    pub fn secret(
        &self,
        name: &str,
        file: &str,
        env: Option<&str>,
    ) -> Result<Option<String>, String> {
        match (self.get(name), self.get(file)) {
            (Some(_), Some(_)) => Err(format!("give --{name} or --{file}, not both")),
            (_, Some(f)) => {
                let t = std::fs::read_to_string(f)
                    .map_err(|e| format!("--{file}: cannot read `{f}`: {e}"))?;
                let v = t
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim_end_matches('\r')
                    .to_string();
                if v.is_empty() {
                    Err(format!("--{file}: `{f}` is empty"))
                } else {
                    Ok(Some(v))
                }
            }
            (Some(v), None) => Ok(Some(v.to_string())),
            (None, None) => Ok(env
                .and_then(|e| std::env::var(e).ok())
                .filter(|v| !v.is_empty())),
        }
    }
    pub fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>, String> {
        match self.get(k) {
            None => Ok(None),
            Some(s) => s
                .parse()
                .map(Some)
                .map_err(|_| format!("--{k}: `{s}` is not a valid number")),
        }
    }
}

/// `4`, `4g`, `4GiB`, `512m`, `1.5t` → bytes (powers of 1024).
pub fn parse_size(s: &str) -> Result<u64, String> {
    let t = s.trim().to_ascii_lowercase();
    let t = t.trim_end_matches('b').trim_end_matches('i');
    let (num, mult) = match t.chars().last() {
        Some('k') => (&t[..t.len() - 1], 1u64 << 10),
        Some('m') => (&t[..t.len() - 1], 1 << 20),
        Some('g') => (&t[..t.len() - 1], 1 << 30),
        Some('t') => (&t[..t.len() - 1], 1 << 40),
        _ => (t, 1),
    };
    let n: f64 = num
        .trim()
        .parse()
        .map_err(|_| format!("`{s}` is not a size (try 8g, 512m or a byte count)"))?;
    if !n.is_finite() || n < 0.0 {
        return Err(format!("`{s}` is not a size"));
    }
    let v = n * mult as f64;
    if v >= u64::MAX as f64 {
        return Err(format!("`{s}` is too large"));
    }
    Ok(v as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn a(s: &str) -> Args {
        Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    #[test]
    fn options_and_positionals() {
        let x = a("model --max-tokens 5 --reproducible --seed=7 --lora a:0.5 --lora b");
        assert_eq!(x.pos, vec!["model"]);
        assert!(x.flag("reproducible"));
        assert_eq!(x.num::<usize>("max-tokens").unwrap(), Some(5));
        assert_eq!(x.num::<u64>("seed").unwrap(), Some(7));
        assert_eq!(x.all("lora"), vec!["a:0.5", "b"]);
        assert!(a("--max-tokens x").num::<usize>("max-tokens").is_err());
    }

    #[test]
    fn every_boolean_flag_used_in_the_source_is_registered() {
        // a boolean `--x` missing from FLAGS would swallow the next argument as its value
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut missing = vec![];
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().and_then(|x| x.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&p).unwrap();
            let mut rest = text.as_str();
            while let Some(i) = rest.find(".flag(\"") {
                rest = &rest[i + 7..];
                let name = &rest[..rest.find('"').unwrap()];
                if !FLAGS.contains(&name) {
                    missing.push(format!("{name} ({})", p.display()));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "boolean flags missing from FLAGS: {missing:?}"
        );
    }

    #[test]
    fn a_boolean_flag_does_not_swallow_the_next_argument() {
        for f in [
            "all",
            "dry-run",
            "explain",
            "no-attention",
            "no-dedup",
            "no-discovery",
            "overwrite",
            "allow-admin",
        ] {
            let x = a(&format!("--{f} model"));
            assert!(
                x.flag(f) && x.pos == vec!["model"],
                "--{f} swallowed its neighbour"
            );
        }
    }

    #[test]
    fn secrets_come_from_a_file_an_option_or_the_environment() {
        let p = std::env::temp_dir().join(format!("corbul-secret-{}", std::process::id()));
        std::fs::write(&p, "from-the-file\r\nignored second line\n").unwrap();
        let f = p.to_str().unwrap();
        assert_eq!(
            a(&format!("--token-file {f}"))
                .secret("token", "token-file", None)
                .unwrap()
                .as_deref(),
            Some("from-the-file")
        );
        assert_eq!(
            a("--token abc")
                .secret("token", "token-file", None)
                .unwrap()
                .as_deref(),
            Some("abc")
        );
        assert!(
            a(&format!("--token abc --token-file {f}"))
                .secret("token", "token-file", None)
                .is_err()
        );
        assert!(
            a("--token-file /nonexistent/zzz")
                .secret("token", "token-file", None)
                .is_err()
        );
        std::fs::write(&p, "\n").unwrap();
        assert!(
            a(&format!("--token-file {f}"))
                .secret("token", "token-file", None)
                .is_err()
        );
        assert_eq!(a("").secret("token", "token-file", None).unwrap(), None);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert_eq!(parse_size("8g").unwrap(), 8 << 30);
        assert_eq!(parse_size("8GiB").unwrap(), 8 << 30);
        assert_eq!(parse_size("1.5m").unwrap(), 1_572_864);
        assert!(parse_size("lots").is_err());
        assert!(parse_size("-3g").is_err());
        assert!(parse_size("1e30t").is_err());
    }
}
