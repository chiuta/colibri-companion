//! The safetensors container: an 8-byte little-endian length N, then N bytes of JSON that map each
//! tensor name to its dtype, shape and byte range inside the data that follows.

use serde_json::Value;
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
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// Parses a header already in memory. `file_len` bounds every range.
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
    let end = 8usize
        .checked_add(n as usize)
        .filter(|e| *e <= bytes.len())
        .ok_or("header runs past the data given")?;
    let json: Value = serde_json::from_slice(&bytes[8..end]).map_err(|e| e.to_string())?;
    let obj = json.as_object().ok_or("header is not a JSON object")?;
    let base = 8 + n;
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
            .ok_or("tensor without data_offsets")?;
        let (b, e) = match (
            off.first().and_then(Value::as_u64),
            off.get(1).and_then(Value::as_u64),
        ) {
            (Some(b), Some(e)) if b <= e => (b, e),
            _ => return Err(format!("bad data_offsets for {name}")),
        };
        let (start, end) = (base + b, base + e);
        if end > file_len {
            return Err(format!("{name} reaches past the end of the file"));
        }
        out.push(Tensor {
            name: name.clone(),
            dtype,
            shape,
            start,
            end,
        });
    }
    out.sort_by_key(|t| t.start);
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
}
