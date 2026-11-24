//! Tiny command-line parser: `--name value`, `--name=value`, bare flags and positionals.

/// Options that never take a value.
pub const FLAGS: &[&str] = &[
    "raw", "no-mmap", "quiet", "help", "log-prompts", "no-log", "reproducible", "stream-layers", "calibrate", "json", "no-governor",
    "governor", "discover", "verbose", "no-lock", "keep-steps", "leaves", "compress", "wire-f16", "lookahead", "self-spec",
];

pub struct Args { pub pos: Vec<String>, pub opts: Vec<(String, Option<String>)> }

impl Args {
    pub fn parse(v: &[String]) -> Args {
        let (mut pos, mut opts) = (vec![], vec![]);
        let mut i = 0;
        while i < v.len() {
            if let Some(name) = v[i].strip_prefix("--") {
                if let Some((k, val)) = name.split_once('=') { opts.push((k.to_string(), Some(val.to_string()))); }
                else if FLAGS.contains(&name) { opts.push((name.to_string(), None)); }
                else if i + 1 < v.len() { opts.push((name.to_string(), Some(v[i + 1].clone()))); i += 1; }
                else { opts.push((name.to_string(), None)); }
            } else { pos.push(v[i].clone()); }
            i += 1;
        }
        Args { pos, opts }
    }
    pub fn flag(&self, k: &str) -> bool { self.opts.iter().any(|(n, _)| n == k) }
    pub fn all(&self, k: &str) -> Vec<&str> { self.opts.iter().filter(|(n, _)| n == k).filter_map(|(_, v)| v.as_deref()).collect() }
    pub fn get(&self, k: &str) -> Option<&str> { self.opts.iter().rev().find(|(n, _)| n == k).and_then(|(_, v)| v.as_deref()) }
    pub fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>, String> {
        match self.get(k) { None => Ok(None), Some(s) => s.parse().map(Some).map_err(|_| format!("--{k}: `{s}` is not a valid number")) }
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
    let n: f64 = num.trim().parse().map_err(|_| format!("`{s}` is not a size (try 8g, 512m or a byte count)"))?;
    if !n.is_finite() || n < 0.0 { return Err(format!("`{s}` is not a size")); }
    let v = n * mult as f64;
    if v >= u64::MAX as f64 { return Err(format!("`{s}` is too large")); }
    Ok(v as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn a(s: &str) -> Args { Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>()) }

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
