//! GGUF reader: header, metadata and tensor table only. Weights are never read here, so a
//! 400 GB file is inspected in milliseconds; the data region is addressed by offset.
use crate::quant::DType;
use std::collections::HashMap;
use std::io::Read;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U64(u64), I64(i64), F64(f64), Bool(bool), Str(String), Array(Vec<Value>),
}
impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self { Value::U64(v) => Some(*v), Value::I64(v) if *v >= 0 => Some(*v as u64), _ => None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self { Value::F64(v) => Some(*v), Value::U64(v) => Some(*v as f64), Value::I64(v) => Some(*v as f64), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: DType,
    /// Offset from the start of the data region.
    pub offset: u64,
}
impl TensorInfo {
    pub fn elements(&self) -> Option<u64> { self.dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)) }
    pub fn bytes(&self) -> Option<u64> { self.dtype.bytes_for(self.elements()?) }
}

#[derive(Debug)]
pub enum GgufError { Io(std::io::Error), BadMagic, Version(u32), Truncated, TooLarge(&'static str), Invalid(String) }
impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "{e}"),
            GgufError::BadMagic => write!(f, "not a GGUF file"),
            GgufError::Version(v) => write!(f, "GGUF version {v} is not supported"),
            GgufError::Truncated => write!(f, "the file ends inside its header"),
            GgufError::TooLarge(w) => write!(f, "implausibly large {w} in the header"),
            GgufError::Invalid(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for GgufError {}
impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::UnexpectedEof { GgufError::Truncated } else { GgufError::Io(e) }
    }
}

#[derive(Debug, Default)]
pub struct Gguf {
    pub version: u32,
    pub meta: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where tensor data begins.
    pub data_start: u64,
}

const MAX_STR: u64 = 1 << 24;
const MAX_ARR: u64 = 1 << 28;
const MAX_COUNT: u64 = 1 << 24;
const MAX_DEPTH: u32 = 4;

struct R<T: Read> { r: T, pos: u64 }
impl<T: Read> R<T> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        self.pos += N as u64;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, GgufError> { Ok(u32::from_le_bytes(self.bytes()?)) }
    fn u64(&mut self) -> Result<u64, GgufError> { Ok(u64::from_le_bytes(self.bytes()?)) }
    fn string(&mut self) -> Result<String, GgufError> {
        let n = self.u64()?;
        if n > MAX_STR { return Err(GgufError::TooLarge("string")); }
        let mut v = vec![0u8; n as usize];
        self.r.read_exact(&mut v)?;
        self.pos += n;
        Ok(String::from_utf8_lossy(&v).into_owned())
    }
    fn skip(&mut self, n: u64) -> Result<(), GgufError> {
        let got = std::io::copy(&mut (&mut self.r).take(n), &mut std::io::sink())?;
        self.pos += got;
        if got < n { Err(GgufError::Truncated) } else { Ok(()) }
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, GgufError> {
        Ok(match ty {
            0 => Value::U64(self.bytes::<1>()?[0] as u64),
            1 => Value::I64(self.bytes::<1>()?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.bytes()?) as u64),
            3 => Value::I64(i16::from_le_bytes(self.bytes()?) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(i32::from_le_bytes(self.bytes()?) as i64),
            6 => Value::F64(f32::from_le_bytes(self.bytes()?) as f64),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth >= MAX_DEPTH { return Err(GgufError::Invalid("arrays nested too deeply".into())); }
                let et = self.u32()?;
                let n = self.u64()?;
                if n > MAX_ARR { return Err(GgufError::TooLarge("array")); }
                // Big arrays (token lists) are kept: callers need vocab size. Cap what we hold.
                let keep = n.min(1 << 20) as usize;
                let mut v = Vec::with_capacity(keep.min(4096));
                for i in 0..n {
                    let item = self.value(et, depth + 1)?;
                    if (i as usize) < keep { v.push(item); }
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            t => return Err(GgufError::Invalid(format!("unknown metadata type {t}"))),
        })
    }
}

impl Gguf {
    pub fn parse<T: Read>(r: T) -> Result<Gguf, GgufError> {
        let mut r = R { r, pos: 0 };
        if &r.bytes::<4>()? != b"GGUF" { return Err(GgufError::BadMagic); }
        let version = r.u32()?;
        if !(2..=3).contains(&version) { return Err(GgufError::Version(version)); }
        let (n_tensors, n_kv) = (r.u64()?, r.u64()?);
        if n_tensors > MAX_COUNT || n_kv > MAX_COUNT { return Err(GgufError::TooLarge("count")); }
        let mut g = Gguf { version, ..Default::default() };
        for _ in 0..n_kv {
            let k = r.string()?;
            let t = r.u32()?;
            let v = r.value(t, 0)?;
            g.meta.insert(k, v);
        }
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()?;
            if nd > 8 { return Err(GgufError::Invalid(format!("tensor {name} has {nd} dimensions"))); }
            let dims = (0..nd).map(|_| r.u64()).collect::<Result<Vec<_>, _>>()?;
            let dtype = DType::from_id(r.u32()?);
            let offset = r.u64()?;
            g.tensors.push(TensorInfo { name, dims, dtype, offset });
        }
        let align = g.meta.get("general.alignment").and_then(Value::as_u64).filter(|a| *a > 0 && a.is_power_of_two()).unwrap_or(32);
        g.data_start = r.pos.div_ceil(align) * align;
        Ok(g)
    }
    pub fn architecture(&self) -> Option<&str> { self.meta.get("general.architecture")?.as_str() }
    /// Architecture-scoped metadata, e.g. `get_u64("block_count")` → `llama.block_count`.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.meta.get(&format!("{}.{key}", self.architecture()?))?.as_u64()
    }
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.meta.get(&format!("{}.{key}", self.architecture()?))?.as_f64()
    }
    pub fn total_tensor_bytes(&self) -> u64 { self.tensors.iter().filter_map(TensorInfo::bytes).sum() }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn put_str(v: &mut Vec<u8>, s: &str) { v.extend((s.len() as u64).to_le_bytes()); v.extend(s.as_bytes()); }

    pub fn sample() -> Vec<u8> {
        let mut v = b"GGUF".to_vec();
        v.extend(3u32.to_le_bytes());
        v.extend(2u64.to_le_bytes()); // tensors
        v.extend(3u64.to_le_bytes()); // kv
        put_str(&mut v, "general.architecture"); v.extend(8u32.to_le_bytes()); put_str(&mut v, "llama");
        put_str(&mut v, "llama.block_count"); v.extend(4u32.to_le_bytes()); v.extend(32u32.to_le_bytes());
        put_str(&mut v, "tokenizer.ggml.tokens"); v.extend(9u32.to_le_bytes()); v.extend(8u32.to_le_bytes());
        v.extend(2u64.to_le_bytes()); put_str(&mut v, "a"); put_str(&mut v, "b");
        put_str(&mut v, "token_embd.weight"); v.extend(2u32.to_le_bytes()); v.extend(256u64.to_le_bytes()); v.extend(4u64.to_le_bytes());
        v.extend(12u32.to_le_bytes()); v.extend(0u64.to_le_bytes());
        put_str(&mut v, "output_norm.weight"); v.extend(1u32.to_le_bytes()); v.extend(256u64.to_le_bytes());
        v.extend(0u32.to_le_bytes()); v.extend(1152u64.to_le_bytes());
        v
    }

    #[test]
    fn reads_header_metadata_and_tensors() {
        let g = Gguf::parse(&sample()[..]).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.get_u64("block_count"), Some(32));
        assert_eq!(g.tensors.len(), 2);
        assert_eq!(g.tensors[0].dtype, DType::Q4K);
        assert_eq!(g.tensors[0].bytes(), Some(4 * 144));
        assert_eq!(g.data_start % 32, 0);
        assert!(g.data_start >= sample().len() as u64);
    }

    #[test]
    fn every_truncation_is_an_error_never_a_panic() {
        let s = sample();
        for n in 0..s.len() { assert!(Gguf::parse(&s[..n]).is_err(), "prefix {n}"); }
    }

    #[test]
    fn hostile_headers_are_refused() {
        assert!(matches!(Gguf::parse(&b"GGUX\0\0\0\0"[..]), Err(GgufError::BadMagic)));
        let mut v = b"GGUF".to_vec();
        v.extend(9u32.to_le_bytes());
        v.extend([0u8; 16]);
        assert!(matches!(Gguf::parse(&v[..]), Err(GgufError::Version(9))));
        let mut v = b"GGUF".to_vec();
        v.extend(3u32.to_le_bytes());
        v.extend(u64::MAX.to_le_bytes());
        v.extend(0u64.to_le_bytes());
        assert!(matches!(Gguf::parse(&v[..]), Err(GgufError::TooLarge(_))));
    }
}
