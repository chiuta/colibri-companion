//! Streaming GGUF v3 writer.
//!
//! The header (metadata and the tensor table) is written first, which is possible because every
//! tensor's size follows from its shape and type. The tensor data then arrives one tensor at a time
//! through [`GgufWriter::write_tensor`], so a model far larger than RAM can be written.
//!
//! Layout rules followed here (from the public GGUF description):
//! * little-endian; `GGUF`, version 3, tensor count, metadata count, then the metadata, then the table;
//! * strings are a `u64` length plus UTF-8 bytes; arrays are an element type, a `u64` count and the items;
//! * a tensor's dimensions are stored fastest-varying first, i.e. the **reverse** of the row-major shape;
//! * every tensor starts at a multiple of `general.alignment` (default 32) from the start of the data
//!   region, which itself starts at an aligned file offset.
use crate::quant::DType;
use std::collections::HashSet;
use std::io::Write;

/// llama.cpp refuses tensor names longer than this many bytes.
pub const MAX_NAME: usize = 64;

/// One metadata value. Every GGUF scalar type is represented so files stay readable by other tools,
/// which compare the stored type exactly (a `u32` key read as `u64` is an error there).
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    pub fn str(s: impl Into<String>) -> MetaValue {
        MetaValue::Str(s.into())
    }
    pub fn strings<I: IntoIterator<Item = S>, S: Into<String>>(it: I) -> MetaValue {
        MetaValue::Array(it.into_iter().map(|s| MetaValue::Str(s.into())).collect())
    }
    fn type_id(&self) -> u32 {
        match self {
            MetaValue::U8(_) => 0,
            MetaValue::I8(_) => 1,
            MetaValue::U16(_) => 2,
            MetaValue::I16(_) => 3,
            MetaValue::U32(_) => 4,
            MetaValue::I32(_) => 5,
            MetaValue::F32(_) => 6,
            MetaValue::Bool(_) => 7,
            MetaValue::Str(_) => 8,
            MetaValue::Array(_) => 9,
            MetaValue::U64(_) => 10,
            MetaValue::I64(_) => 11,
            MetaValue::F64(_) => 12,
        }
    }
    /// Convert a value read by [`crate::gguf::Gguf`] (which widens integers to 64 bits) back to the
    /// narrowest type that holds it, the way llama.cpp writes counts and ids: unsigned values that fit
    /// become `u32`, signed ones `i32`, floats `f32`.
    pub fn from_read(v: &crate::gguf::Value) -> MetaValue {
        use crate::gguf::Value as V;
        match v {
            V::U64(x) => u32::try_from(*x).map_or(MetaValue::U64(*x), MetaValue::U32),
            V::I64(x) => i32::try_from(*x).map_or(MetaValue::I64(*x), MetaValue::I32),
            V::F64(x) => MetaValue::F32(*x as f32),
            V::Bool(b) => MetaValue::Bool(*b),
            V::Str(s) => MetaValue::Str(s.clone()),
            V::Array(a) => MetaValue::Array(a.iter().map(MetaValue::from_read).collect()),
        }
    }
}

/// A tensor to be written: name, **row-major** shape and stored type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDesc {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
}

impl TensorDesc {
    pub fn new(name: impl Into<String>, shape: &[u64], dtype: DType) -> TensorDesc {
        TensorDesc {
            name: name.into(),
            shape: shape.to_vec(),
            dtype,
        }
    }
    pub fn elements(&self) -> Option<u64> {
        self.shape.iter().try_fold(1u64, |a, &d| a.checked_mul(d))
    }
    /// Bytes the tensor's data occupies, if the shape is valid for the type.
    pub fn bytes(&self) -> Option<u64> {
        self.dtype.bytes_for(self.elements()?)
    }
}

#[derive(Debug)]
pub enum WriteError {
    Io(std::io::Error),
    /// A problem with what the caller supplied (names, shapes, sizes, order).
    Invalid(String),
}
impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::Io(e) => write!(f, "{e}"),
            WriteError::Invalid(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for WriteError {}
impl From<std::io::Error> for WriteError {
    fn from(e: std::io::Error) -> Self {
        WriteError::Io(e)
    }
}
fn invalid<T>(s: impl Into<String>) -> Result<T, WriteError> {
    Err(WriteError::Invalid(s.into()))
}

/// Counts bytes written so alignment padding can be computed without seeking.
struct Counting<W: Write> {
    w: W,
    pos: u64,
}
impl<W: Write> Counting<W> {
    fn put(&mut self, b: &[u8]) -> Result<(), WriteError> {
        self.w.write_all(b)?;
        self.pos += b.len() as u64;
        Ok(())
    }
    fn u32(&mut self, v: u32) -> Result<(), WriteError> {
        self.put(&v.to_le_bytes())
    }
    fn u64(&mut self, v: u64) -> Result<(), WriteError> {
        self.put(&v.to_le_bytes())
    }
    fn string(&mut self, s: &str) -> Result<(), WriteError> {
        self.u64(s.len() as u64)?;
        self.put(s.as_bytes())
    }
    fn pad_to(&mut self, align: u64) -> Result<(), WriteError> {
        let rem = self.pos % align;
        if rem != 0 {
            self.put(&vec![0u8; (align - rem) as usize])?;
        }
        Ok(())
    }
}

fn write_scalar<W: Write>(w: &mut Counting<W>, v: &MetaValue) -> Result<(), WriteError> {
    match v {
        MetaValue::U8(x) => w.put(&[*x]),
        MetaValue::I8(x) => w.put(&x.to_le_bytes()),
        MetaValue::U16(x) => w.put(&x.to_le_bytes()),
        MetaValue::I16(x) => w.put(&x.to_le_bytes()),
        MetaValue::U32(x) => w.put(&x.to_le_bytes()),
        MetaValue::I32(x) => w.put(&x.to_le_bytes()),
        MetaValue::F32(x) => w.put(&x.to_le_bytes()),
        MetaValue::Bool(b) => w.put(&[*b as u8]),
        MetaValue::Str(s) => w.string(s),
        MetaValue::U64(x) => w.put(&x.to_le_bytes()),
        MetaValue::I64(x) => w.put(&x.to_le_bytes()),
        MetaValue::F64(x) => w.put(&x.to_le_bytes()),
        MetaValue::Array(_) => invalid("internal: arrays are written by write_payload"),
    }
}

/// `key, value-type, value` — arrays carry their element type and count after the value type.
fn write_kv<W: Write>(w: &mut Counting<W>, key: &str, v: &MetaValue) -> Result<(), WriteError> {
    w.string(key)?;
    w.u32(v.type_id())?;
    write_payload(w, v, 0)
}

fn write_payload<W: Write>(
    w: &mut Counting<W>,
    v: &MetaValue,
    depth: u32,
) -> Result<(), WriteError> {
    match v {
        MetaValue::Array(items) => {
            if depth >= 3 {
                return invalid("metadata arrays nested too deeply");
            }
            let ty = items.first().map_or(4, MetaValue::type_id); // an empty array is typed u32
            if items.iter().any(|i| i.type_id() != ty) {
                return invalid("an array mixes element types");
            }
            w.u32(ty)?;
            w.u64(items.len() as u64)?;
            for i in items {
                write_payload(w, i, depth + 1)?;
            }
            Ok(())
        }
        other => write_scalar(w, other),
    }
}

/// Streaming writer. Create it with the metadata and the full tensor list, then supply each
/// tensor's bytes in the same order.
pub struct GgufWriter<W: Write> {
    w: Counting<W>,
    descs: Vec<TensorDesc>,
    expected: Vec<u64>,
    next: usize,
    align: u64,
    data_start: u64,
}

impl<W: Write> GgufWriter<W> {
    /// Write the header. `general.alignment`, when present in `meta`, sets the alignment (a power of two).
    pub fn new(
        out: W,
        meta: &[(String, MetaValue)],
        tensors: Vec<TensorDesc>,
    ) -> Result<Self, WriteError> {
        let mut seen = HashSet::new();
        for (k, _) in meta {
            if k.is_empty() {
                return invalid("an empty metadata key");
            }
            if !seen.insert(k.as_str()) {
                return invalid(format!("duplicate metadata key `{k}`"));
            }
        }
        let align = match meta.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, MetaValue::U32(a))) => u64::from(*a),
            Some((_, MetaValue::U64(a))) => *a,
            Some(_) => return invalid("general.alignment must be an unsigned integer"),
            None => 32,
        };
        if align == 0 || !align.is_power_of_two() || align > (1 << 20) {
            return invalid("general.alignment must be a power of two up to 1 MiB");
        }
        let mut names = HashSet::new();
        let mut expected = Vec::with_capacity(tensors.len());
        for t in &tensors {
            if t.name.is_empty() || t.name.len() > MAX_NAME {
                return invalid(format!(
                    "tensor name `{}` must be 1..={MAX_NAME} bytes",
                    t.name
                ));
            }
            if !names.insert(t.name.as_str()) {
                return invalid(format!("duplicate tensor `{}`", t.name));
            }
            if t.shape.is_empty() || t.shape.len() > 4 {
                return invalid(format!("tensor `{}` needs 1 to 4 dimensions", t.name));
            }
            if t.shape.contains(&0) {
                return invalid(format!("tensor `{}` has a zero dimension", t.name));
            }
            let Some(b) = t.bytes() else {
                return invalid(format!(
                    "tensor `{}`: shape {:?} is not valid for {:?} (unknown type, overflow, or the last dimension is not a multiple of the block size)",
                    t.name, t.shape, t.dtype
                ));
            };
            expected.push(b);
        }
        let mut w = Counting { w: out, pos: 0 };
        w.put(b"GGUF")?;
        w.u32(3)?;
        w.u64(tensors.len() as u64)?;
        w.u64(meta.len() as u64)?;
        for (k, v) in meta {
            write_kv(&mut w, k, v)?;
        }
        // tensor table: offsets are relative to the data region, each aligned
        let mut off = 0u64;
        for (t, &b) in tensors.iter().zip(&expected) {
            w.string(&t.name)?;
            w.u32(t.shape.len() as u32)?;
            for &d in t.shape.iter().rev() {
                w.u64(d)?;
            }
            w.u32(dtype_id(t.dtype)?)?;
            w.u64(off)?;
            off = match off
                .checked_add(b)
                .and_then(|o| o.checked_add(align - 1))
                .map(|o| o / align * align)
            {
                Some(o) => o,
                None => return invalid("the tensors together exceed the addressable size"),
            };
        }
        w.pad_to(align)?;
        let data_start = w.pos;
        Ok(GgufWriter {
            w,
            descs: tensors,
            expected,
            next: 0,
            align,
            data_start,
        })
    }

    /// Where the tensor data begins in the file.
    pub fn data_start(&self) -> u64 {
        self.data_start
    }
    /// The tensor the next call to [`write_tensor`](Self::write_tensor) must supply.
    pub fn next_tensor(&self) -> Option<&TensorDesc> {
        self.descs.get(self.next)
    }

    /// Supply the bytes of the next tensor; its length must match exactly.
    pub fn write_tensor(&mut self, data: &[u8]) -> Result<(), WriteError> {
        let Some(d) = self.descs.get(self.next) else {
            return invalid("all tensors have already been written");
        };
        let want = self.expected[self.next];
        if data.len() as u64 != want {
            return invalid(format!(
                "tensor `{}`: {} bytes given, {want} expected",
                d.name,
                data.len()
            ));
        }
        self.w.pad_to(self.align)?;
        self.w.put(data)?;
        self.next += 1;
        Ok(())
    }

    /// Check that every tensor was supplied, flush, and return the underlying writer.
    pub fn finish(mut self) -> Result<W, WriteError> {
        if self.next != self.descs.len() {
            return invalid(format!(
                "{} of {} tensors were written",
                self.next,
                self.descs.len()
            ));
        }
        self.w.pad_to(self.align)?;
        self.w.w.flush()?;
        Ok(self.w.w)
    }

    /// Write a whole file: `provide(index, desc)` returns each tensor's bytes in turn.
    pub fn write_all(
        out: W,
        meta: &[(String, MetaValue)],
        tensors: Vec<TensorDesc>,
        mut provide: impl FnMut(usize, &TensorDesc) -> Result<Vec<u8>, WriteError>,
    ) -> Result<W, WriteError> {
        let mut g = GgufWriter::new(out, meta, tensors)?;
        for i in 0..g.descs.len() {
            let bytes = provide(i, &g.descs[i])?;
            g.write_tensor(&bytes)?;
        }
        g.finish()
    }
}

/// The numeric id GGUF uses for a stored type (the inverse of [`DType::from_id`]).
pub fn dtype_id(t: DType) -> Result<u32, WriteError> {
    Ok(match t {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::Q4_0 => 2,
        DType::Q4_1 => 3,
        DType::Q5_0 => 6,
        DType::Q5_1 => 7,
        DType::Q8_0 => 8,
        DType::Q2K => 10,
        DType::Q3K => 11,
        DType::Q4K => 12,
        DType::Q5K => 13,
        DType::Q6K => 14,
        DType::Q8K => 15,
        DType::BF16 => 30,
        DType::Other(n) => return invalid(format!("cannot write unknown tensor type {n}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{Gguf, Value};
    use crate::loader::{GgufFile, TensorSource};

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn dtype_ids_are_the_inverse_of_from_id() {
        for t in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Q8_0,
            DType::Q2K,
            DType::Q3K,
            DType::Q4K,
            DType::Q5K,
            DType::Q6K,
            DType::Q8K,
        ] {
            assert_eq!(DType::from_id(dtype_id(t).unwrap()), t);
        }
        assert!(dtype_id(DType::Other(99)).is_err());
    }

    #[test]
    fn round_trip_through_the_reader_with_every_metadata_type() {
        let meta = vec![
            ("general.architecture".to_string(), MetaValue::str("llama")),
            ("llama.block_count".to_string(), MetaValue::U32(4)),
            ("a.u8".to_string(), MetaValue::U8(200)),
            ("a.i8".to_string(), MetaValue::I8(-5)),
            ("a.u16".to_string(), MetaValue::U16(65000)),
            ("a.i16".to_string(), MetaValue::I16(-300)),
            ("a.i32".to_string(), MetaValue::I32(-70000)),
            ("a.f32".to_string(), MetaValue::F32(0.5)),
            ("a.bool".to_string(), MetaValue::Bool(true)),
            ("a.u64".to_string(), MetaValue::U64(1 << 40)),
            ("a.i64".to_string(), MetaValue::I64(-(1 << 40))),
            ("a.f64".to_string(), MetaValue::F64(1.25)),
            (
                "tok.list".to_string(),
                MetaValue::strings(["a", "bb", "ccc"]),
            ),
            (
                "nums".to_string(),
                MetaValue::Array(vec![MetaValue::I32(1), MetaValue::I32(-2)]),
            ),
            ("empty".to_string(), MetaValue::Array(vec![])),
        ];
        let w0: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let w1: Vec<f32> = (0..5).map(|i| -(i as f32)).collect();
        let descs = vec![
            TensorDesc::new("w0", &[3, 4], DType::F32),
            TensorDesc::new("w1", &[5], DType::F32),
        ];
        let out = GgufWriter::write_all(Vec::new(), &meta, descs, |i, _| {
            Ok(f32s(if i == 0 { &w0 } else { &w1 }))
        })
        .unwrap();
        let g = Gguf::parse(&out[..]).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.get_u64("block_count"), Some(4));
        assert_eq!(g.meta["a.u8"], Value::U64(200));
        assert_eq!(g.meta["a.i8"], Value::I64(-5));
        assert_eq!(g.meta["a.f64"], Value::F64(1.25));
        assert_eq!(g.meta["a.bool"], Value::Bool(true));
        assert_eq!(
            g.meta["tok.list"],
            Value::Array(vec![
                Value::Str("a".into()),
                Value::Str("bb".into()),
                Value::Str("ccc".into())
            ])
        );
        // dims are stored fastest-first: shape [3, 4] -> dims [4, 3]
        assert_eq!(g.tensors[0].dims, vec![4, 3]);
        assert_eq!(g.tensors[1].dims, vec![5]);
        assert_eq!(g.data_start % 32, 0);
        assert_eq!(g.tensors[0].offset, 0);
        assert_eq!(g.tensors[1].offset % 32, 0);
        let b = g.data_start as usize + g.tensors[1].offset as usize;
        assert_eq!(&out[b..b + 20], &f32s(&w1)[..]);
        assert_eq!(out.len() % 32, 0);
    }

    #[test]
    fn tensors_read_back_through_the_loader_including_quantised_types() {
        let dir = std::env::temp_dir().join(format!("gguf-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.gguf");
        let q8: Vec<u8> = {
            let mut v = vec![];
            for b in 0..2 {
                v.extend(0x3c00u16.to_le_bytes());
                v.extend((0..32).map(|i| (i as i8 - 16 + b) as u8));
            }
            v
        };
        let descs = vec![
            TensorDesc::new("a.weight", &[2, 32], DType::Q8_0),
            TensorDesc::new("b", &[4], DType::F32),
        ];
        let meta = vec![
            ("general.architecture".to_string(), MetaValue::str("llama")),
            ("general.alignment".to_string(), MetaValue::U32(64)),
        ];
        let f = std::fs::File::create(&p).unwrap();
        GgufWriter::write_all(std::io::BufWriter::new(f), &meta, descs, |i, _| {
            Ok(if i == 0 {
                q8.clone()
            } else {
                f32s(&[1.0, 2.0, 3.0, 4.0])
            })
        })
        .unwrap();
        let g = GgufFile::open(&p).unwrap();
        assert_eq!(g.gguf.data_start % 64, 0);
        let a = g.read("a.weight").unwrap();
        assert_eq!(a.len(), 64);
        assert_eq!(a[0], -16.0);
        assert_eq!(a[32], -15.0);
        assert_eq!(g.read("b").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_read_narrows_integers_like_llama_cpp() {
        assert_eq!(MetaValue::from_read(&Value::U64(7)), MetaValue::U32(7));
        assert_eq!(
            MetaValue::from_read(&Value::U64(1 << 40)),
            MetaValue::U64(1 << 40)
        );
        assert_eq!(MetaValue::from_read(&Value::I64(-3)), MetaValue::I32(-3));
        assert_eq!(MetaValue::from_read(&Value::F64(0.5)), MetaValue::F32(0.5));
    }

    #[test]
    fn wrong_sizes_order_and_names_are_errors() {
        let d = |n: &str, s: &[u64], t| TensorDesc::new(n, s, t);
        let meta: Vec<(String, MetaValue)> = vec![];
        // wrong byte count
        let mut w = GgufWriter::new(Vec::new(), &meta, vec![d("a", &[4], DType::F32)]).unwrap();
        assert!(w.write_tensor(&[0; 15]).is_err());
        w.write_tensor(&[0; 16]).unwrap();
        assert!(w.write_tensor(&[0; 16]).is_err()); // one tensor too many
        // too few tensors
        let w = GgufWriter::new(
            Vec::new(),
            &meta,
            vec![d("a", &[4], DType::F32), d("b", &[4], DType::F32)],
        )
        .unwrap();
        assert!(w.finish().is_err());
        // names and shapes
        assert!(
            GgufWriter::new(
                Vec::new(),
                &meta,
                vec![d("a", &[4], DType::F32), d("a", &[4], DType::F32)]
            )
            .is_err()
        );
        assert!(GgufWriter::new(Vec::new(), &meta, vec![d("", &[4], DType::F32)]).is_err());
        assert!(
            GgufWriter::new(
                Vec::new(),
                &meta,
                vec![d(&"x".repeat(65), &[4], DType::F32)]
            )
            .is_err()
        );
        assert!(GgufWriter::new(Vec::new(), &meta, vec![d("a", &[], DType::F32)]).is_err());
        assert!(GgufWriter::new(Vec::new(), &meta, vec![d("a", &[0, 4], DType::F32)]).is_err());
        assert!(
            GgufWriter::new(
                Vec::new(),
                &meta,
                vec![d("a", &[1, 1, 1, 1, 1], DType::F32)]
            )
            .is_err()
        );
        // a row that is not a whole number of blocks, an overflowing shape, an unknown type
        assert!(GgufWriter::new(Vec::new(), &meta, vec![d("a", &[2, 33], DType::Q8_0)]).is_err());
        assert!(
            GgufWriter::new(Vec::new(), &meta, vec![d("a", &[u64::MAX, 2], DType::F32)]).is_err()
        );
        assert!(GgufWriter::new(Vec::new(), &meta, vec![d("a", &[32], DType::Other(77))]).is_err());
        // sizes that overflow the address space only when summed
        let big = u64::MAX / 8;
        assert!(
            GgufWriter::new(
                Vec::new(),
                &meta,
                vec![d("a", &[big], DType::F32), d("b", &[big], DType::F32)]
            )
            .is_err()
        );
        // metadata problems
        let dup = vec![
            ("k".to_string(), MetaValue::U32(1)),
            ("k".to_string(), MetaValue::U32(2)),
        ];
        assert!(GgufWriter::new(Vec::new(), &dup, vec![]).is_err());
        let mixed = vec![(
            "k".to_string(),
            MetaValue::Array(vec![MetaValue::U32(1), MetaValue::Bool(true)]),
        )];
        assert!(GgufWriter::new(Vec::new(), &mixed, vec![]).is_err());
        let deep = (0..6).fold(MetaValue::U8(1), |v, _| MetaValue::Array(vec![v]));
        assert!(GgufWriter::new(Vec::new(), &[("k".to_string(), deep)], vec![]).is_err());
        for bad in [0u32, 3, 48] {
            let m = vec![("general.alignment".to_string(), MetaValue::U32(bad))];
            assert!(
                GgufWriter::new(Vec::new(), &m, vec![]).is_err(),
                "alignment {bad}"
            );
        }
        assert!(
            GgufWriter::new(
                Vec::new(),
                &[("general.alignment".to_string(), MetaValue::Str("x".into()))],
                vec![]
            )
            .is_err()
        );
    }

    #[test]
    fn nested_arrays_of_arrays_round_trip_up_to_the_depth_limit() {
        // GGUF allows arrays of arrays; the reader keeps them, so the writer must produce them correctly
        let v = MetaValue::Array(vec![
            MetaValue::Array(vec![MetaValue::U32(1), MetaValue::U32(2)]),
            MetaValue::Array(vec![MetaValue::U32(3)]),
        ]);
        let out = GgufWriter::write_all(
            Vec::new(),
            &[("m".to_string(), v)],
            vec![],
            |_, _| unreachable!(),
        )
        .unwrap();
        let g = Gguf::parse(&out[..]).unwrap();
        assert_eq!(
            g.meta["m"],
            Value::Array(vec![
                Value::Array(vec![Value::U64(1), Value::U64(2)]),
                Value::Array(vec![Value::U64(3)])
            ])
        );
    }

    #[test]
    fn a_failing_provider_stops_the_write() {
        let r = GgufWriter::write_all(
            Vec::new(),
            &[],
            vec![TensorDesc::new("a", &[1], DType::F32)],
            |_, _| invalid("boom"),
        );
        assert!(matches!(r, Err(WriteError::Invalid(_))));
    }
}
