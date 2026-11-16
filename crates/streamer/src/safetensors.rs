//! The safetensors container: an 8-byte little-endian length N, then N bytes of JSON that map each
//! tensor name to its dtype, shape and byte range inside the data that follows.

use serde_json::Value;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Headers larger than this are refused (the format's own reference limit is 100 MB).
pub const MAX_HEADER: u64 = 100 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute byte range in the file: header included.
    pub start: u64,
    pub end: u64,
}

impl Tensor {
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Bytes per element for the dtypes whose size is a whole number of bytes; `None` for the rest
/// (sub-byte or unknown), which are not size-checked.
fn dtype_bytes(d: &str) -> Option<u64> {
    Some(match d {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" => 1,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => return None,
    })
}

/// The first object key that appears twice in the same JSON object, at any depth (compared after
/// unescaping, so `"a"` and `"\u0061"` are the same key). `serde_json` keeps only the last of two
/// equal keys without a word, which would let a header hide one tensor behind another; this is a
/// single pass over the bytes that tracks only strings and the nesting of `{}`/`[]`. Malformed
/// JSON never panics here (the real parser reports it).
pub fn duplicate_key(json: &[u8]) -> Option<String> {
    // one entry per open container: (keys seen, a key is expected next); `None` marks an array
    let mut stack: Vec<(Option<HashSet<String>>, bool)> = Vec::new();
    let n = json.len();
    let mut i = 0;
    while i < n {
        match json[i] {
            b'{' => stack.push((Some(HashSet::new()), true)),
            b'[' => stack.push((None, false)),
            b'}' | b']' => {
                stack.pop();
            }
            b',' => {
                if let Some((Some(_), expect)) = stack.last_mut() {
                    *expect = true;
                }
            }
            b'"' => {
                let mut j = i + 1;
                while j < n && json[j] != b'"' {
                    j += if json[j] == b'\\' { 2 } else { 1 };
                }
                if j >= n {
                    return None; // unterminated string: not our error to report
                }
                if let Some((Some(keys), expect)) = stack.last_mut() {
                    if *expect {
                        *expect = false;
                        if let Ok(k) = serde_json::from_slice::<String>(&json[i..=j]) {
                            if !keys.insert(k.clone()) {
                                return Some(k);
                            }
                        }
                    }
                }
                i = j;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Parses a header already in memory. `file_len` bounds every range. Hostile input gets an error:
/// offsets are added with overflow checks, the declared size of every tensor must match its
/// dtype and shape, and no two tensors may share bytes.
pub fn parse_header(bytes: &[u8], file_len: u64) -> Result<Vec<Tensor>, String> {
    if bytes.len() < 8 {
        return Err("shorter than the 8-byte length prefix".into());
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    if n > MAX_HEADER {
        return Err(format!(
            "header of {n} bytes is over the {MAX_HEADER} limit"
        ));
    }
    let base = 8 + n; // n <= MAX_HEADER, cannot overflow
    if base > file_len {
        return Err("header runs past the end of the file".into());
    }
    let end = usize::try_from(base)
        .ok()
        .filter(|e| *e <= bytes.len())
        .ok_or("header runs past the data given")?;
    if let Some(k) = duplicate_key(&bytes[8..end]) {
        return Err(format!("duplicate key {k:?} in the header"));
    }
    let json: Value = serde_json::from_slice(&bytes[8..end]).map_err(|e| e.to_string())?;
    let obj = json.as_object().ok_or("header is not a JSON object")?;
    let mut out = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"]
            .as_str()
            .ok_or("tensor without dtype")?
            .to_string();
        let shape = v["shape"]
            .as_array()
            .ok_or("tensor without shape")?
            .iter()
            .map(|d| d.as_u64().ok_or("bad dimension"))
            .collect::<Result<Vec<_>, _>>()?;
        let off = v["data_offsets"]
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or_else(|| format!("{name}: data_offsets must be two numbers"))?;
        let (b, e) = match (off[0].as_u64(), off[1].as_u64()) {
            (Some(b), Some(e)) if b <= e => (b, e),
            _ => return Err(format!("bad data_offsets for {name}")),
        };
        let (start, end) = match (base.checked_add(b), base.checked_add(e)) {
            (Some(s), Some(e)) => (s, e),
            _ => return Err(format!("{name}: offsets overflow")),
        };
        if end > file_len {
            return Err(format!("{name} reaches past the end of the file"));
        }
        if let Some(size) = dtype_bytes(&dtype) {
            let elems = shape
                .iter()
                .try_fold(1u64, |acc, d| acc.checked_mul(*d))
                .ok_or_else(|| format!("{name}: shape overflows"))?;
            if elems.checked_mul(size) != Some(e - b) {
                return Err(format!(
                    "{name}: {dtype} {shape:?} does not match its byte range"
                ));
            }
        }
        out.push(Tensor {
            name: name.clone(),
            dtype,
            shape,
            start,
            end,
        });
    }
    out.sort_by(|a, b| (a.start, a.end, &a.name).cmp(&(b.start, b.end, &b.name)));
    let mut last_end = 0u64;
    for t in out.iter().filter(|t| t.start < t.end) {
        if t.start < last_end {
            return Err(format!("{} overlaps another tensor", t.name));
        }
        last_end = t.end;
    }
    Ok(out)
}

/// Reads only the header of a file: a few kilobytes to a few megabytes, never the weights.
pub fn read_header(path: &Path) -> Result<Vec<Tensor>, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let file_len = f.metadata().map_err(|e| e.to_string())?.len();
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(|e| e.to_string())?;
    let n = u64::from_le_bytes(len);
    if n > MAX_HEADER {
        return Err(format!(
            "header of {n} bytes is over the {MAX_HEADER} limit"
        ));
    }
    // refuse before allocating: a 20-byte file must not make us reserve 100 MB
    if 8 + n > file_len {
        return Err("header runs past the end of the file".into());
    }
    let mut buf = vec![0u8; 8 + n as usize];
    buf[..8].copy_from_slice(&len);
    f.read_exact(&mut buf[8..]).map_err(|e| e.to_string())?;
    parse_header(&buf, file_len)
}

#[cfg(test)]
pub(crate) fn build(tensors: &[(&str, &str, &[u64], u64, u64)], data_len: u64) -> Vec<u8> {
    let mut m = serde_json::Map::new();
    for (n, d, s, b, e) in tensors {
        m.insert(
            n.to_string(),
            serde_json::json!({"dtype": d, "shape": s, "data_offsets": [b, e]}),
        );
    }
    let j = serde_json::to_vec(&Value::Object(m)).unwrap();
    let mut out = (j.len() as u64).to_le_bytes().to_vec();
    out.extend(j);
    out.extend(vec![0u8; data_len as usize]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_ranges_as_absolute_offsets() {
        let f = build(&[("a", "F16", &[2, 2], 0, 8), ("b", "U8", &[4], 8, 12)], 12);
        let t = parse_header(&f, f.len() as u64).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].name, "a");
        assert_eq!(t[1].end - t[1].start, 4);
        assert_eq!(t[1].end, f.len() as u64);
    }

    #[test]
    fn refuses_ranges_past_the_file_and_huge_headers() {
        let f = build(&[("a", "U8", &[4], 0, 400)], 4);
        assert!(parse_header(&f, f.len() as u64).is_err());
        let mut huge = (MAX_HEADER + 1).to_le_bytes().to_vec();
        huge.extend([0u8; 16]);
        assert!(parse_header(&huge, 1 << 40).is_err());
        assert!(parse_header(&[1, 2, 3], 3).is_err());
    }

    fn raw(json: &str, data: usize) -> Vec<u8> {
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend(json.as_bytes());
        out.extend(vec![0u8; data]);
        out
    }

    fn err(f: &[u8]) -> String {
        parse_header(f, f.len() as u64).unwrap_err()
    }

    #[test]
    fn offsets_near_u64_max_are_an_error_not_an_overflow() {
        let j = format!(
            r#"{{"a":{{"dtype":"U8","shape":[1],"data_offsets":[{},{}]}}}}"#,
            u64::MAX - 1,
            u64::MAX
        );
        assert!(err(&raw(&j, 8)).contains("overflow"));
    }

    #[test]
    fn a_shape_that_does_not_fit_its_bytes_is_refused() {
        let f = build(&[("a", "F16", &[2, 2], 0, 4)], 4);
        assert!(err(&f).contains("does not match"));
        let f = build(&[("a", "U8", &[u64::MAX, 2], 0, 4)], 4);
        assert!(err(&f).contains("overflows"));
    }

    #[test]
    fn overlapping_tensors_are_refused_but_empty_ones_may_sit_anywhere() {
        let f = build(&[("a", "U8", &[8], 0, 8), ("b", "U8", &[8], 4, 12)], 12);
        assert!(err(&f).contains("overlaps"));
        let f = build(&[("a", "U8", &[8], 0, 8), ("e", "U8", &[0], 4, 4)], 8);
        assert!(parse_header(&f, f.len() as u64).is_ok());
    }

    #[test]
    fn data_offsets_need_exactly_two_numbers() {
        let j = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1,2]}}"#;
        assert!(err(&raw(j, 1)).contains("two numbers"));
        let j = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0]}}"#;
        assert!(err(&raw(j, 1)).contains("two numbers"));
    }

    #[test]
    fn a_header_longer_than_the_file_is_refused_before_allocating() {
        let dir = std::env::temp_dir().join(format!("st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("liar.safetensors");
        let mut f = (90u64 * 1024 * 1024).to_le_bytes().to_vec(); // claims 90 MB
        f.extend([b'{'; 12]);
        std::fs::write(&p, f).unwrap();
        let e = read_header(&p).unwrap_err();
        assert!(e.contains("past the end"), "{e}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn junk_headers_are_errors() {
        assert!(parse_header(&[], 0).is_err());
        assert!(parse_header(&0u64.to_le_bytes(), 8).is_err()); // empty JSON
        assert!(err(&raw("[1,2]", 0)).contains("not a JSON object"));
        assert!(err(&raw(r#"{"a":5}"#, 0)).contains("dtype"));
        let mut bad = raw(r#"{"a":"xx"}"#, 0);
        bad[14] = 0xff; // invalid UTF-8 inside a string value
        assert!(parse_header(&bad, bad.len() as u64).is_err());
        let f = build(&[("a", "U8", &[8], 0, 8)], 3); // header fine, file truncated
        assert!(err(&f).contains("past the end"));
    }

    #[test]
    fn unknown_and_sub_byte_dtypes_are_not_size_checked() {
        let f = build(&[("a", "F4", &[3], 0, 2)], 2);
        assert!(parse_header(&f, f.len() as u64).is_ok());
    }

    #[test]
    fn a_key_given_twice_is_an_error_not_a_silent_last_wins() {
        // the second "a" would overwrite the first in a plain parse, hiding a 4-byte tensor
        let j = r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"a":{"dtype":"U8","shape":[1],"data_offsets":[4,5]}}"#;
        assert!(err(&raw(j, 5)).contains("duplicate key \"a\""));
        // inside one tensor entry as well
        let j = r#"{"a":{"dtype":"U8","dtype":"F32","shape":[1],"data_offsets":[0,1]}}"#;
        assert!(err(&raw(j, 1)).contains("duplicate key \"dtype\""));
        // equal after unescaping
        let j = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"\u0061":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        assert!(err(&raw(j, 2)).contains("duplicate"));
        // the same name in different objects, in arrays, or inside a string value is fine
        let ok = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]},"__metadata__":{"x":"\",\"a\":","y":["a","a",{"k":1},{"k":2}]}}"#;
        let f = raw(ok, 2);
        assert_eq!(parse_header(&f, f.len() as u64).unwrap().len(), 2);
    }

    #[test]
    fn the_duplicate_scan_agrees_with_a_reference_on_random_json() {
        // reference: serialise with the keys collected by hand, count duplicates the slow way
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let keys = [
            "a",
            "b",
            "dtype",
            "x y",
            "q\"uote",
            "back\\slash",
            "é",
            "\u{1F600}",
            "",
        ];
        // returns (json text, whether some object has the same key twice)
        fn random_json(r: &mut impl FnMut() -> u64, keys: &[&str], depth: u32) -> (String, bool) {
            match r() % if depth > 3 { 3 } else { 6 } {
                0 => ("1".into(), false),
                1 => (
                    serde_json::to_string(keys[(r() % keys.len() as u64) as usize]).unwrap(),
                    false,
                ),
                2 => ("null".into(), false),
                3 | 4 => {
                    let n = r() % 5;
                    let (mut parts, mut seen, mut dup) = (Vec::new(), Vec::new(), false);
                    for _ in 0..n {
                        let k = keys[(r() % keys.len() as u64) as usize];
                        if seen.contains(&k) {
                            dup = true;
                        }
                        seen.push(k);
                        let (v, d) = random_json(r, keys, depth + 1);
                        dup |= d;
                        parts.push(format!("{}:{v}", serde_json::to_string(k).unwrap()));
                    }
                    (format!("{{{}}}", parts.join(",")), dup)
                }
                _ => {
                    let n = r() % 4;
                    let (mut parts, mut dup) = (Vec::new(), false);
                    for _ in 0..n {
                        let (v, d) = random_json(r, keys, depth + 1);
                        dup |= d;
                        parts.push(v);
                    }
                    (format!("[{}]", parts.join(",")), dup)
                }
            }
        }
        let (mut flagged, mut clean) = (0, 0);
        for _ in 0..4000 {
            let (text, has_dup) = random_json(&mut r, &keys, 0);
            assert_eq!(duplicate_key(text.as_bytes()).is_some(), has_dup, "{text}");
            if has_dup {
                flagged += 1;
            } else {
                clean += 1;
            }
        }
        assert!(flagged > 200 && clean > 200, "{flagged} {clean}");
    }

    #[test]
    fn the_duplicate_scan_never_panics_on_garbage() {
        let mut s = 77u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let alphabet = b"{}[],:\"\\ab1 \n";
        for _ in 0..20000 {
            let len = (r() % 40) as usize;
            let g: Vec<u8> = (0..len)
                .map(|_| alphabet[(r() % alphabet.len() as u64) as usize])
                .collect();
            let _ = duplicate_key(&g);
        }
        assert_eq!(duplicate_key(b"\"unterminated"), None);
        assert_eq!(duplicate_key(b"{\"a\\"), None);
        assert_eq!(duplicate_key(b""), None);
        assert_eq!(duplicate_key(b"}}}]]]"), None);
    }

    #[test]
    fn random_bytes_and_mutated_valid_headers_are_errors_or_ok_never_panics() {
        let mut s = 123u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let base = build(
            &[
                ("a.layers.0.mlp.experts.1.w", "F16", &[2, 3], 0, 12),
                ("b", "U8", &[4], 12, 16),
            ],
            16,
        );
        let mut ok_after = 0;
        for _ in 0..5000 {
            let mut f = base.clone();
            for _ in 0..(1 + r() % 4) {
                let at = (r() % f.len() as u64) as usize;
                f[at] = (r() & 0xff) as u8;
            }
            let cut = (r() % (f.len() as u64 + 1)) as usize;
            let len = if r() % 3 == 0 { cut } else { f.len() };
            if parse_header(&f[..len], f.len() as u64).is_ok() {
                ok_after += 1;
            }
        }
        // a header that parses must also respect its own invariants
        let f = base.clone();
        let t = parse_header(&f, f.len() as u64).unwrap();
        assert!(
            t.iter()
                .all(|t| t.start <= t.end && t.end <= f.len() as u64)
        );
        assert!(ok_after < 5000);
    }
}
