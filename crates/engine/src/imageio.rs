//! Image input for the vision encoder: decoders for PNG, JPEG (baseline and progressive), BMP and
//! the PNM family, a deflate decoder of its own, resampling (nearest, bilinear, bicubic, area,
//! Lanczos), cropping/padding and conversion to the planar float layout encoders expect.
//!
//! Written from the public format specifications (RFC 1951/1950, the PNG specification, ITU T.81
//! for JPEG, the Microsoft BMP documentation, the Netpbm descriptions); no existing decoder was
//! read or copied. Pure std, no dependencies.
//!
//! **Hostile input is the normal case.** Every size taken from a file is checked against
//! [`Limits`] before anything is allocated, a decompressed stream may never grow past what the
//! header promises (so a deflate bomb is an error, not an out-of-memory), loops are bounded by
//! the input length, and nothing in here panics on any byte sequence (the tests mutate valid
//! files thousands of times to check that).
//!
//! Decoded images are always 8-bit RGB. Alpha is composited over a background colour (white by
//! default); 16-bit samples are reduced to 8 bits; greyscale and palette images are expanded.
//! Not supported (clear [`ImageError::Unsupported`] errors): JPEG with arithmetic coding, 12-bit
//! samples, lossless or hierarchical modes, CMYK/YCCK; 2-component JPEG; PNG outside the
//! standard colour-type/depth combinations; BMP with JPEG/PNG payloads.

use std::fmt;

/// One decoded picture: `rgb` holds `width * height * 3` bytes, row by row, top row first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image { pub width: usize, pub height: usize, pub rgb: Vec<u8> }

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageError {
    /// The bytes are not a valid file of the format they claim.
    Corrupt(String),
    /// The file ends before the picture does.
    Truncated(String),
    /// A valid file this decoder does not handle.
    Unsupported(String),
    /// The picture is larger than the configured [`Limits`].
    TooLarge(String),
}
impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageError::Corrupt(s) => write!(f, "corrupt image: {s}"),
            ImageError::Truncated(s) => write!(f, "truncated image: {s}"),
            ImageError::Unsupported(s) => write!(f, "unsupported image: {s}"),
            ImageError::TooLarge(s) => write!(f, "image too large: {s}"),
        }
    }
}
impl std::error::Error for ImageError {}
type Res<T> = Result<T, ImageError>;
fn corrupt<T>(s: impl Into<String>) -> Res<T> { Err(ImageError::Corrupt(s.into())) }
fn trunc<T>(s: impl Into<String>) -> Res<T> { Err(ImageError::Truncated(s.into())) }
fn unsupported<T>(s: impl Into<String>) -> Res<T> { Err(ImageError::Unsupported(s.into())) }

/// Upper bounds applied before any allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Most pixels a picture may have (width × height), also for the padded planes of a JPEG.
    pub max_pixels: u64,
    /// Longest side.
    pub max_dim: usize,
    /// Largest input accepted, in bytes.
    pub max_input: usize,
}
impl Default for Limits {
    /// 32 megapixels, 16 384 on a side, 256 MiB of input.
    fn default() -> Self { Limits { max_pixels: 1 << 25, max_dim: 1 << 14, max_input: 256 << 20 } }
}
impl Limits {
    fn check(&self, w: usize, h: usize) -> Res<()> {
        if w == 0 || h == 0 { return corrupt("the picture has a zero side"); }
        if w > self.max_dim || h > self.max_dim { return Err(ImageError::TooLarge(format!("{w}x{h} exceeds the {} pixel side limit", self.max_dim))); }
        if (w as u64).checked_mul(h as u64).is_none_or(|p| p > self.max_pixels) {
            return Err(ImageError::TooLarge(format!("{w}x{h} exceeds the {} pixel limit", self.max_pixels)));
        }
        Ok(())
    }
}

/// The container formats [`sniff`] recognises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format { Png, Jpeg, Bmp, Pnm }

/// Which format the first bytes announce, if any.
pub fn sniff(b: &[u8]) -> Option<Format> {
    if b.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) { Some(Format::Png) }
    else if b.starts_with(&[0xff, 0xd8, 0xff]) { Some(Format::Jpeg) }
    else if b.starts_with(b"BM") { Some(Format::Bmp) }
    else if b.len() >= 2 && b[0] == b'P' && (b'1'..=b'6').contains(&b[1]) { Some(Format::Pnm) }
    else { None }
}

/// Decode with the default limits, compositing transparency over white.
pub fn decode(bytes: &[u8]) -> Res<Image> { decode_with(bytes, &Limits::default(), [255, 255, 255]) }

/// Decode any supported format; `background` is what transparent pixels are blended with.
pub fn decode_with(bytes: &[u8], lim: &Limits, background: [u8; 3]) -> Res<Image> {
    if bytes.len() > lim.max_input { return Err(ImageError::TooLarge(format!("{} input bytes exceed the {} byte limit", bytes.len(), lim.max_input))); }
    match sniff(bytes) {
        Some(Format::Png) => decode_png(bytes, lim, background),
        Some(Format::Jpeg) => decode_jpeg(bytes, lim),
        Some(Format::Bmp) => decode_bmp(bytes, lim, background),
        Some(Format::Pnm) => decode_pnm(bytes, lim),
        None => unsupported("not a PNG, JPEG, BMP or PNM file"),
    }
}

fn alloc_rgb(w: usize, h: usize, lim: &Limits) -> Res<Vec<u8>> {
    lim.check(w, h)?;
    Ok(vec![0u8; w * h * 3])
}
fn blend(c: u8, a: u8, bg: u8) -> u8 { ((c as u32 * a as u32 + bg as u32 * (255 - a as u32) + 127) / 255) as u8 }

// ------------------------------------------------------------------------------------ checksums

const fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 { c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 }; k += 1; }
        t[n] = c;
        n += 1;
    }
    t
}
static CRC: [u32; 256] = crc_table();
/// CRC-32 (the PNG/zlib polynomial) of `data`.
pub fn crc32(data: &[u8]) -> u32 { crc32_update(0, data) }
fn crc32_update(prev: u32, data: &[u8]) -> u32 {
    let mut c = !prev;
    for &b in data { c = CRC[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8); }
    !c
}
/// Adler-32 of `data`.
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk { a += x as u32; b += a; }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

// -------------------------------------------------------------------------------------- inflate

struct Bits<'a> { d: &'a [u8], pos: usize, buf: u64, cnt: u32 }
impl<'a> Bits<'a> {
    fn new(d: &'a [u8]) -> Self { Bits { d, pos: 0, buf: 0, cnt: 0 } }
    fn bits(&mut self, n: u32) -> Res<u32> {
        debug_assert!(n <= 24);
        while self.cnt < n {
            let Some(&b) = self.d.get(self.pos) else { return trunc("deflate stream ends early"); };
            self.pos += 1;
            self.buf |= (b as u64) << self.cnt;
            self.cnt += 8;
        }
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.cnt -= n;
        Ok(v)
    }
    /// Drop the bits of the current byte (stored blocks start on a byte boundary).
    fn align(&mut self) {
        let drop = self.cnt % 8;
        self.buf >>= drop;
        self.cnt -= drop;
    }
    /// Bytes consumed so far, counting a partly used byte as consumed.
    fn consumed(&self) -> usize { self.pos - (self.cnt / 8) as usize }
}

struct Huff { count: [u16; 16], symbol: Vec<u16> }
impl Huff {
    fn new(lengths: &[u8]) -> Res<Huff> {
        let mut count = [0u16; 16];
        for &l in lengths { count[l as usize] += 1; }
        let mut left: i32 = 1;
        for &c in &count[1..] {
            left <<= 1;
            left -= c as i32;
            if left < 0 { return corrupt("over-subscribed Huffman code in a deflate stream"); }
        }
        let mut offs = [0u16; 16];
        for l in 1..15 { offs[l + 1] = offs[l] + count[l]; }
        let mut symbol = vec![0u16; lengths.len()];
        for (s, &l) in lengths.iter().enumerate() {
            if l != 0 { symbol[offs[l as usize] as usize] = s as u16; offs[l as usize] += 1; }
        }
        Ok(Huff { count, symbol })
    }
    fn decode(&self, b: &mut Bits) -> Res<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= b.bits(1)? as i32;
            let count = self.count[len] as i32;
            if code - count < first { return Ok(self.symbol[(index + (code - first)) as usize]); }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        corrupt("invalid Huffman code in a deflate stream")
    }
}

const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEXT: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DBASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DEXT: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// Raw deflate (RFC 1951). The output may never exceed `max_out` bytes; returns the data and the
/// number of input bytes used.
pub fn inflate(data: &[u8], max_out: usize) -> Res<(Vec<u8>, usize)> {
    let mut b = Bits::new(data);
    let mut out: Vec<u8> = Vec::with_capacity(max_out.min(1 << 20));
    loop {
        let last = b.bits(1)?;
        match b.bits(2)? {
            0 => {
                b.align();
                let len = b.bits(16)? as usize;
                let nlen = b.bits(16)? as usize;
                if len != (!nlen & 0xffff) { return corrupt("stored deflate block with a bad length check"); }
                if out.len() + len > max_out { return Err(ImageError::TooLarge("the compressed stream expands past the size its header promises".into())); }
                for _ in 0..len { out.push(b.bits(8)? as u8); }
            }
            1 => {
                let mut l = [0u8; 288];
                l[..144].fill(8); l[144..256].fill(9); l[256..280].fill(7); l[280..].fill(8);
                let lit = Huff::new(&l)?;
                let dist = Huff::new(&[5u8; 30])?;
                codes(&mut b, &mut out, &lit, &dist, max_out)?;
            }
            2 => {
                let nlen = b.bits(5)? as usize + 257;
                let ndist = b.bits(5)? as usize + 1;
                let ncode = b.bits(4)? as usize + 4;
                if nlen > 286 || ndist > 30 { return corrupt("too many deflate length or distance codes"); }
                const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                let mut cl = [0u8; 19];
                for &o in ORDER.iter().take(ncode) { cl[o] = b.bits(3)? as u8; }
                let clh = Huff::new(&cl)?;
                let mut lengths = vec![0u8; nlen + ndist];
                let mut i = 0;
                while i < nlen + ndist {
                    let sym = clh.decode(&mut b)?;
                    if sym < 16 { lengths[i] = sym as u8; i += 1; continue; }
                    let (val, rep) = match sym {
                        16 => { if i == 0 { return corrupt("deflate repeats a length before any"); } (lengths[i - 1], 3 + b.bits(2)? as usize) }
                        17 => (0, 3 + b.bits(3)? as usize),
                        _ => (0, 11 + b.bits(7)? as usize),
                    };
                    if i + rep > nlen + ndist { return corrupt("deflate length repeat runs past the table"); }
                    lengths[i..i + rep].fill(val);
                    i += rep;
                }
                if lengths[256] == 0 { return corrupt("deflate block without an end-of-block code"); }
                let lit = Huff::new(&lengths[..nlen])?;
                let dist = Huff::new(&lengths[nlen..])?;
                codes(&mut b, &mut out, &lit, &dist, max_out)?;
            }
            _ => return corrupt("deflate block of reserved type"),
        }
        if last == 1 { break; }
    }
    let used = b.consumed();
    Ok((out, used))
}

fn codes(b: &mut Bits, out: &mut Vec<u8>, lit: &Huff, dist: &Huff, max_out: usize) -> Res<()> {
    loop {
        let sym = lit.decode(b)? as usize;
        match sym {
            0..=255 => {
                if out.len() >= max_out { return Err(ImageError::TooLarge("the compressed stream expands past the size its header promises".into())); }
                out.push(sym as u8);
            }
            256 => return Ok(()),
            257..=285 => {
                let s = sym - 257;
                let len = LBASE[s] as usize + b.bits(LEXT[s] as u32)? as usize;
                let ds = dist.decode(b)? as usize;
                if ds >= 30 { return corrupt("deflate distance code out of range"); }
                let d = DBASE[ds] as usize + b.bits(DEXT[ds] as u32)? as usize;
                if d > out.len() { return corrupt("deflate distance reaches before the start of the data"); }
                if out.len() + len > max_out { return Err(ImageError::TooLarge("the compressed stream expands past the size its header promises".into())); }
                let start = out.len() - d;
                for k in 0..len { let v = out[start + k]; out.push(v); }
            }
            _ => return corrupt("deflate literal/length code out of range"),
        }
    }
}

/// zlib container (RFC 1950): header, deflate, Adler-32.
pub fn zlib_decompress(data: &[u8], max_out: usize) -> Res<Vec<u8>> {
    if data.len() < 6 { return trunc("zlib stream is shorter than its header"); }
    let (cmf, flg) = (data[0], data[1]);
    if cmf & 0x0f != 8 || (cmf >> 4) > 7 || !((cmf as u32) << 8 | flg as u32).is_multiple_of(31) { return corrupt("bad zlib header"); }
    if flg & 0x20 != 0 { return unsupported("zlib stream with a preset dictionary"); }
    let (out, used) = inflate(&data[2..], max_out)?;
    let tail = data.get(2 + used..2 + used + 4).ok_or_else(|| ImageError::Truncated("zlib stream misses its checksum".into()))?;
    if u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]) != adler32(&out) { return corrupt("zlib checksum mismatch"); }
    Ok(out)
}

// ------------------------------------------------------------------------------------------ PNG

fn be32(b: &[u8]) -> u32 { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) }

struct PngHeader { w: usize, h: usize, depth: u8, ctype: u8, interlace: bool }
impl PngHeader {
    fn channels(&self) -> usize { match self.ctype { 0 | 3 => 1, 2 => 3, 4 => 2, _ => 4 } }
    fn bits_pp(&self) -> usize { self.channels() * self.depth as usize }
    fn stride(&self, w: usize) -> usize { (w * self.bits_pp()).div_ceil(8) }
}
const ADAM7: [(usize, usize, usize, usize); 7] = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];

/// PNG: all colour types and bit depths, Adam7 interlacing, palette and `tRNS` transparency.
/// Chunk CRCs and the zlib checksum are verified.
pub fn decode_png(b: &[u8], lim: &Limits, bg: [u8; 3]) -> Res<Image> {
    if sniff(b) != Some(Format::Png) { return corrupt("missing PNG signature"); }
    let mut pos = 8;
    let (mut hdr, mut idat, mut plte, mut trns, mut seen_end) = (None::<PngHeader>, Vec::<u8>::new(), Vec::<u8>::new(), Vec::<u8>::new(), false);
    let mut first = true;
    while pos + 8 <= b.len() && !seen_end {
        let len = be32(&b[pos..]) as usize;
        let ty = &b[pos + 4..pos + 8];
        let body_end = pos.checked_add(8).and_then(|p| p.checked_add(len));
        let Some(body_end) = body_end.filter(|&e| e.checked_add(4).is_some_and(|c| c <= b.len())) else { return trunc("a PNG chunk runs past the end of the file"); };
        let body = &b[pos + 8..body_end];
        if crc32_update(crc32(ty), body) != be32(&b[body_end..]) {
            // Ancillary chunks with a bad CRC are skipped; critical ones are fatal.
            if ty[0] & 0x20 == 0 { return corrupt(format!("CRC mismatch in the {} chunk", String::from_utf8_lossy(ty))); }
            pos = body_end + 4;
            continue;
        }
        if first && ty != b"IHDR" { return corrupt("the first PNG chunk is not IHDR"); }
        first = false;
        match ty {
            b"IHDR" => {
                if hdr.is_some() || len != 13 { return corrupt("bad IHDR"); }
                let (w, h) = (be32(body) as usize, be32(&body[4..]) as usize);
                let (depth, ctype, comp, filt, il) = (body[8], body[9], body[10], body[11], body[12]);
                let ok = match ctype { 0 => matches!(depth, 1 | 2 | 4 | 8 | 16), 3 => matches!(depth, 1 | 2 | 4 | 8), 2 | 4 | 6 => matches!(depth, 8 | 16), _ => false };
                if !ok { return unsupported(format!("PNG colour type {ctype} with bit depth {depth}")); }
                if comp != 0 || filt != 0 || il > 1 { return unsupported("PNG with an unknown compression, filter or interlace method"); }
                lim.check(w, h)?;
                hdr = Some(PngHeader { w, h, depth, ctype, interlace: il == 1 });
            }
            b"PLTE" => {
                if len == 0 || !len.is_multiple_of(3) || len > 768 { return corrupt("bad PLTE size"); }
                plte = body.to_vec();
            }
            b"tRNS" => trns = body.to_vec(),
            b"IDAT" => {
                if idat.len() + body.len() > b.len() { return corrupt("IDAT data larger than the file"); }
                idat.extend_from_slice(body);
            }
            b"IEND" => seen_end = true,
            _ => if ty[0] & 0x20 == 0 { return unsupported(format!("critical PNG chunk {}", String::from_utf8_lossy(ty))); },
        }
        pos = body_end + 4;
    }
    let Some(h) = hdr else { return corrupt("PNG without IHDR"); };
    if idat.is_empty() { return corrupt("PNG without image data"); }
    if h.ctype == 3 && plte.is_empty() { return corrupt("palette PNG without PLTE"); }

    // Expected size of the filtered data.
    let passes: Vec<(usize, usize, usize, usize)> = if h.interlace { ADAM7.to_vec() } else { vec![(0, 0, 1, 1)] };
    let mut dims = vec![];
    let mut expect = 0usize;
    for &(x0, y0, dx, dy) in &passes {
        let pw = if h.w > x0 { (h.w - x0).div_ceil(dx) } else { 0 };
        let ph = if h.h > y0 { (h.h - y0).div_ceil(dy) } else { 0 };
        dims.push((pw, ph));
        if pw > 0 && ph > 0 { expect += ph * (1 + h.stride(pw)); }
    }
    let raw = zlib_decompress(&idat, expect)?;
    if raw.len() != expect { return trunc("the PNG data stream is shorter than the picture needs"); }

    let mut out = alloc_rgb(h.w, h.h, lim)?;
    let bpp = h.bits_pp().div_ceil(8).max(1);
    let mut off = 0;
    for (&(x0, y0, dx, dy), &(pw, ph)) in passes.iter().zip(&dims) {
        if pw == 0 || ph == 0 { continue; }
        let stride = h.stride(pw);
        let mut prev = vec![0u8; stride];
        let mut cur = vec![0u8; stride];
        for y in 0..ph {
            let ft = raw[off];
            cur.copy_from_slice(&raw[off + 1..off + 1 + stride]);
            off += 1 + stride;
            unfilter(ft, &mut cur, &prev, bpp)?;
            for x in 0..pw {
                let (px, py) = (x0 + x * dx, y0 + y * dy);
                let rgb = png_pixel(&h, &cur, x, &plte, &trns, bg);
                let o = (py * h.w + px) * 3;
                out[o..o + 3].copy_from_slice(&rgb);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
    }
    Ok(Image { width: h.w, height: h.h, rgb: out })
}

fn unfilter(ft: u8, cur: &mut [u8], prev: &[u8], bpp: usize) -> Res<()> {
    match ft {
        0 => {}
        1 => for i in bpp..cur.len() { cur[i] = cur[i].wrapping_add(cur[i - bpp]); },
        2 => for i in 0..cur.len() { cur[i] = cur[i].wrapping_add(prev[i]); },
        3 => for i in 0..cur.len() {
            let left = if i >= bpp { cur[i - bpp] as u16 } else { 0 };
            cur[i] = cur[i].wrapping_add(((left + prev[i] as u16) / 2) as u8);
        },
        4 => for i in 0..cur.len() {
            let a = if i >= bpp { cur[i - bpp] as i32 } else { 0 };
            let bb = prev[i] as i32;
            let c = if i >= bpp { prev[i - bpp] as i32 } else { 0 };
            let p = a + bb - c;
            let (pa, pb, pc) = ((p - a).abs(), (p - bb).abs(), (p - c).abs());
            let pred = if pa <= pb && pa <= pc { a } else if pb <= pc { bb } else { c };
            cur[i] = cur[i].wrapping_add(pred as u8);
        },
        _ => return corrupt(format!("unknown PNG filter type {ft}")),
    }
    Ok(())
}

/// Sample `i` (counting samples across the row) of a row stored at `depth` bits.
fn sample(row: &[u8], i: usize, depth: u8) -> u16 {
    match depth {
        8 => row[i] as u16,
        16 => u16::from_be_bytes([row[2 * i], row[2 * i + 1]]),
        d => {
            let per = 8 / d as usize;
            let byte = row[i / per];
            let shift = 8 - d as usize * (i % per + 1);
            ((byte >> shift) & ((1u8 << d) - 1)) as u16
        }
    }
}

fn png_pixel(h: &PngHeader, row: &[u8], x: usize, plte: &[u8], trns: &[u8], bg: [u8; 3]) -> [u8; 3] {
    let maxv = (1u32 << h.depth) - 1;
    let to8 = |v: u16| -> u8 { if h.depth == 8 { v as u8 } else if h.depth == 16 { (v >> 8) as u8 } else { (v as u32 * 255 / maxv) as u8 } };
    match h.ctype {
        0 => {
            let v = sample(row, x, h.depth);
            let g = to8(v);
            let key = (trns.len() >= 2).then(|| u16::from_be_bytes([trns[0], trns[1]]));
            if key == Some(v) { return bg; }
            [g, g, g]
        }
        2 => {
            let s = [sample(row, 3 * x, h.depth), sample(row, 3 * x + 1, h.depth), sample(row, 3 * x + 2, h.depth)];
            if trns.len() >= 6 && (0..3).all(|c| s[c] == u16::from_be_bytes([trns[2 * c], trns[2 * c + 1]])) { return bg; }
            [to8(s[0]), to8(s[1]), to8(s[2])]
        }
        3 => {
            let idx = sample(row, x, h.depth) as usize;
            let Some(p) = plte.get(3 * idx..3 * idx + 3) else { return [0, 0, 0]; };
            let a = trns.get(idx).copied().unwrap_or(255);
            [blend(p[0], a, bg[0]), blend(p[1], a, bg[1]), blend(p[2], a, bg[2])]
        }
        4 => {
            let (g, a) = (to8(sample(row, 2 * x, h.depth)), to8(sample(row, 2 * x + 1, h.depth)));
            [blend(g, a, bg[0]), blend(g, a, bg[1]), blend(g, a, bg[2])]
        }
        _ => {
            let c = [to8(sample(row, 4 * x, h.depth)), to8(sample(row, 4 * x + 1, h.depth)), to8(sample(row, 4 * x + 2, h.depth))];
            let a = to8(sample(row, 4 * x + 3, h.depth));
            [blend(c[0], a, bg[0]), blend(c[1], a, bg[1]), blend(c[2], a, bg[2])]
        }
    }
}

// ------------------------------------------------------------------------------------------ BMP

fn le16(b: &[u8], o: usize) -> Option<u16> { Some(u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?])) }
fn le32(b: &[u8], o: usize) -> Option<u32> { Some(u32::from_le_bytes([*b.get(o)?, *b.get(o + 1)?, *b.get(o + 2)?, *b.get(o + 3)?])) }

/// BMP: 1/4/8-bit palette, 16/24/32-bit, `BI_BITFIELDS`, RLE4/RLE8, top-down and bottom-up.
pub fn decode_bmp(b: &[u8], lim: &Limits, bg: [u8; 3]) -> Res<Image> {
    let bad = || ImageError::Truncated("the BMP header is cut short".into());
    if !b.starts_with(b"BM") { return corrupt("missing BMP signature"); }
    let data_off = le32(b, 10).ok_or_else(bad)? as usize;
    let dib = le32(b, 14).ok_or_else(bad)? as usize;
    let (w, hraw, bpp, comp, ncol, masks_at);
    if dib == 12 {
        w = le16(b, 18).ok_or_else(bad)? as i64;
        hraw = le16(b, 20).ok_or_else(bad)? as i64;
        bpp = le16(b, 24).ok_or_else(bad)? as usize;
        comp = 0u32;
        ncol = 0usize;
        masks_at = 0;
    } else if dib >= 40 {
        w = le32(b, 18).ok_or_else(bad)? as i32 as i64;
        hraw = le32(b, 22).ok_or_else(bad)? as i32 as i64;
        bpp = le16(b, 28).ok_or_else(bad)? as usize;
        comp = le32(b, 30).ok_or_else(bad)?;
        ncol = le32(b, 46).ok_or_else(bad)? as usize;
        masks_at = 54;
    } else { return unsupported(format!("BMP header of {dib} bytes")); }
    if le16(b, 26) != Some(1) && dib != 12 { return corrupt("BMP planes field is not 1"); }
    if w <= 0 || hraw == 0 { return corrupt("BMP with a zero or negative width or zero height"); }
    let (top_down, h) = (hraw < 0, hraw.unsigned_abs() as usize);
    let w = w as usize;
    lim.check(w, h)?;
    if !matches!(bpp, 1 | 4 | 8 | 16 | 24 | 32) { return unsupported(format!("BMP with {bpp} bits per pixel")); }
    if !matches!(comp, 0..=3 | 6) { return unsupported(format!("BMP compression method {comp}")); }
    if (comp == 1 && bpp != 8) || (comp == 2 && bpp != 4) || (comp == 3 || comp == 6) && !matches!(bpp, 16 | 32) {
        return corrupt("BMP compression does not match its bit depth");
    }
    // palette
    let pal_off = 14 + dib;
    let pal_entry = if dib == 12 { 3 } else { 4 };
    let mut palette: Vec<[u8; 3]> = vec![];
    if bpp <= 8 {
        let n = if ncol == 0 { 1usize << bpp } else { ncol.min(1 << bpp) };
        for i in 0..n {
            let Some(e) = b.get(pal_off + i * pal_entry..pal_off + i * pal_entry + 3) else { return trunc("the BMP palette is cut short"); };
            palette.push([e[2], e[1], e[0]]);
        }
    }
    // bit fields
    let (mut masks, mut have_alpha) = ([0u32; 4], false);
    if bpp == 16 { masks = [0x7c00, 0x03e0, 0x001f, 0]; } else if bpp == 32 { masks = [0x00ff_0000, 0x0000_ff00, 0x0000_00ff, 0]; }
    if (comp == 3 || comp == 6) && masks_at != 0 {
        let n = if comp == 6 || dib >= 56 { 4 } else { 3 };
        for (i, m) in masks.iter_mut().enumerate().take(n) { *m = le32(b, masks_at + 4 * i).ok_or_else(bad)?; }
        have_alpha = masks[3] != 0;
    }
    let mut out = alloc_rgb(w, h, lim)?;
    let put = |out: &mut [u8], x: usize, y: usize, rgb: [u8; 3]| {
        let yy = if top_down { y } else { h - 1 - y };
        let o = (yy * w + x) * 3;
        out[o..o + 3].copy_from_slice(&rgb);
    };
    if data_off > b.len() { return trunc("the BMP pixel data starts past the end of the file"); }
    let data = &b[data_off..];
    if comp == 1 || comp == 2 {
        // run-length coded
        let (mut x, mut y, mut i) = (0usize, 0usize, 0usize);
        let get = |i: usize| data.get(i).copied();
        'outer: while y < h {
            let (Some(n), Some(v)) = (get(i), get(i + 1)) else { return trunc("RLE data ends early"); };
            i += 2;
            if n > 0 {
                for k in 0..n as usize {
                    if x >= w { break; }
                    let idx = (if comp == 1 { v } else if k % 2 == 0 { v >> 4 } else { v & 15 }) as usize;
                    put(&mut out, x, y, palette.get(idx).copied().unwrap_or([0; 3]));
                    x += 1;
                }
            } else {
                match v {
                    0 => { x = 0; y += 1; }
                    1 => break 'outer,
                    2 => {
                        let (Some(dx), Some(dy)) = (get(i), get(i + 1)) else { return trunc("RLE data ends early"); };
                        i += 2;
                        x += dx as usize;
                        y += dy as usize;
                    }
                    cnt => {
                        let cnt = cnt as usize;
                        let nbytes = if comp == 1 { cnt } else { cnt.div_ceil(2) };
                        let Some(run) = data.get(i..i + nbytes) else { return trunc("RLE data ends early"); };
                        for k in 0..cnt {
                            if x >= w { break; }
                            let idx = (if comp == 1 { run[k] } else if k % 2 == 0 { run[k / 2] >> 4 } else { run[k / 2] & 15 }) as usize;
                            put(&mut out, x, y, palette.get(idx).copied().unwrap_or([0; 3]));
                            x += 1;
                        }
                        i += nbytes + (nbytes & 1);
                    }
                }
            }
        }
        return Ok(Image { width: w, height: h, rgb: out });
    }
    let stride = (w * bpp).div_ceil(32) * 4;
    if (stride as u64).checked_mul(h as u64).is_none_or(|n| n > data.len() as u64) { return trunc("the BMP pixel data is shorter than the picture needs"); }
    for y in 0..h {
        let row = &data[y * stride..(y + 1) * stride];
        for x in 0..w {
            let rgb = match bpp {
                1 | 4 | 8 => {
                    let per = 8 / bpp;
                    let idx = ((row[x / per] >> (8 - bpp * (x % per + 1))) & ((1u16 << bpp) - 1) as u8) as usize;
                    palette.get(idx).copied().unwrap_or([0; 3])
                }
                24 => [row[3 * x + 2], row[3 * x + 1], row[3 * x]],
                _ => {
                    let v = if bpp == 16 { u16::from_le_bytes([row[2 * x], row[2 * x + 1]]) as u32 } else { u32::from_le_bytes([row[4 * x], row[4 * x + 1], row[4 * x + 2], row[4 * x + 3]]) };
                    let ch = |m: u32| -> u8 {
                        if m == 0 { return 0; }
                        let sh = m.trailing_zeros();
                        let bits = (m >> sh).count_ones();
                        let x = (v & m) >> sh;
                        if bits >= 8 { (x >> (bits - 8)) as u8 } else { let mx = (1u32 << bits) - 1; ((x * 255 + mx / 2) / mx) as u8 }
                    };
                    let (r, g, bl) = (ch(masks[0]), ch(masks[1]), ch(masks[2]));
                    if have_alpha {
                        let a = ch(masks[3]);
                        [blend(r, a, bg[0]), blend(g, a, bg[1]), blend(bl, a, bg[2])]
                    } else { [r, g, bl] }
                }
            };
            put(&mut out, x, y, rgb);
        }
    }
    Ok(Image { width: w, height: h, rgb: out })
}

// ------------------------------------------------------------------------------------------ PNM

/// Netpbm: P1–P6 (ASCII or binary bitmap, greymap, pixmap), maximum value up to 65535.
pub fn decode_pnm(b: &[u8], lim: &Limits) -> Res<Image> {
    if b.len() < 3 || b[0] != b'P' { return corrupt("missing PNM magic"); }
    let kind = b[1];
    if !(b'1'..=b'6').contains(&kind) { return unsupported("PNM kind (only P1 to P6)"); }
    let mut pos = 2;
    let num = |pos: &mut usize| -> Res<usize> {
        loop {
            while *pos < b.len() && b[*pos].is_ascii_whitespace() { *pos += 1; }
            if *pos < b.len() && b[*pos] == b'#' { while *pos < b.len() && b[*pos] != b'\n' { *pos += 1; } } else { break; }
        }
        let start = *pos;
        let mut v = 0usize;
        while *pos < b.len() && b[*pos].is_ascii_digit() {
            v = v.checked_mul(10).and_then(|v| v.checked_add((b[*pos] - b'0') as usize)).ok_or_else(|| ImageError::Corrupt("PNM number overflow".into()))?;
            *pos += 1;
            if *pos - start > 10 { return corrupt("PNM number too long"); }
        }
        if *pos == start { return corrupt("PNM header number expected"); }
        Ok(v)
    };
    let (w, h) = (num(&mut pos)?, num(&mut pos)?);
    lim.check(w, h)?;
    let bitmap = kind == b'1' || kind == b'4';
    let maxv = if bitmap { 1 } else { num(&mut pos)? };
    if maxv == 0 || maxv > 65535 { return corrupt("PNM maximum value out of range"); }
    let ascii = kind <= b'3';
    let chans = if kind == b'3' || kind == b'6' { 3 } else { 1 };
    // exactly one whitespace byte separates the header from binary data
    if !ascii {
        if pos >= b.len() || !b[pos].is_ascii_whitespace() { return trunc("PNM header ends early"); }
        pos += 1;
    }
    let mut out = alloc_rgb(w, h, lim)?;
    let scale = |v: usize| -> u8 { if maxv == 255 { v as u8 } else { ((v * 255 + maxv / 2) / maxv).min(255) as u8 } };
    if kind == b'4' {
        let stride = w.div_ceil(8);
        let need = stride.checked_mul(h).ok_or_else(|| ImageError::TooLarge("PNM size".into()))?;
        let Some(d) = b.get(pos..pos + need) else { return trunc("PNM pixel data is shorter than the picture needs"); };
        for y in 0..h { for x in 0..w {
            let bit = (d[y * stride + x / 8] >> (7 - x % 8)) & 1;
            let v = if bit == 1 { 0 } else { 255 };
            let o = (y * w + x) * 3;
            out[o..o + 3].fill(v);
        } }
    } else if ascii {
        for y in 0..h { for x in 0..w {
            let o = (y * w + x) * 3;
            if bitmap {
                // P1 digits may be written without separators
                while pos < b.len() && b[pos].is_ascii_whitespace() { pos += 1; }
                let Some(&c) = b.get(pos) else { return trunc("PNM pixel data is shorter than the picture needs"); };
                pos += 1;
                let v = match c { b'0' => 255, b'1' => 0, _ => return corrupt("PNM bitmap digit expected") };
                out[o..o + 3].fill(v);
            } else {
                for c in 0..chans {
                    let v = num(&mut pos).map_err(|e| if pos >= b.len() { ImageError::Truncated("PNM pixel data is shorter than the picture needs".into()) } else { e })?;
                    if v > maxv { return corrupt("PNM sample above the maximum value"); }
                    if chans == 1 { out[o..o + 3].fill(scale(v)); } else { out[o + c] = scale(v); }
                }
            }
        } }
    } else {
        let bytes = if maxv > 255 { 2 } else { 1 };
        let need = w.checked_mul(h).and_then(|n| n.checked_mul(chans * bytes)).ok_or_else(|| ImageError::TooLarge("PNM size".into()))?;
        let Some(d) = b.get(pos..pos + need) else { return trunc("PNM pixel data is shorter than the picture needs"); };
        for i in 0..w * h {
            for c in 0..chans {
                let k = (i * chans + c) * bytes;
                let v = if bytes == 2 { u16::from_be_bytes([d[k], d[k + 1]]) as usize } else { d[k] as usize };
                if v > maxv { return corrupt("PNM sample above the maximum value"); }
                if chans == 1 { out[i * 3..i * 3 + 3].fill(scale(v)); } else { out[i * 3 + c] = scale(v); }
            }
        }
    }
    Ok(Image { width: w, height: h, rgb: out })
}

// ----------------------------------------------------------------------------------------- JPEG

const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

struct HuffTable { fast: [u16; 512], maxcode: [i32; 18], valptr: [i32; 17], mincode: [i32; 17], vals: Vec<u8> }
impl HuffTable {
    fn new(counts: &[u8; 16], vals: &[u8]) -> Res<HuffTable> {
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total > 256 || total != vals.len() { return corrupt("bad JPEG Huffman table"); }
        let mut t = HuffTable { fast: [0; 512], maxcode: [-1; 18], valptr: [0; 17], mincode: [0; 17], vals: vals.to_vec() };
        let (mut code, mut k) = (0i32, 0usize);
        for l in 1..=16 {
            t.valptr[l] = k as i32;
            t.mincode[l] = code;
            let n = counts[l - 1] as i32;
            if n > 0 {
                if code + n > (1 << l) { return corrupt("over-subscribed JPEG Huffman table"); }
                // fast lookup for codes of up to 9 bits
                if l <= 9 {
                    for i in 0..n {
                        let c = (code + i) as usize;
                        let lo = c << (9 - l);
                        for e in lo..lo + (1 << (9 - l)) { t.fast[e] = ((l as u16) << 8) | vals[k + i as usize] as u16; }
                    }
                }
                code += n;
                k += n as usize;
                t.maxcode[l] = code - 1;
            } else { t.maxcode[l] = -1; }
            code <<= 1;
        }
        t.maxcode[17] = i32::MAX;
        Ok(t)
    }
}

struct BitReader<'a> { d: &'a [u8], pos: usize, buf: u32, cnt: u32, marker: Option<u8>, overrun: u32 }
impl<'a> BitReader<'a> {
    fn new(d: &'a [u8], pos: usize) -> Self { BitReader { d, pos, buf: 0, cnt: 0, marker: None, overrun: 0 } }
    fn fill(&mut self) {
        while self.cnt <= 24 {
            let byte = if self.marker.is_some() || self.pos >= self.d.len() {
                self.overrun += 1;
                0
            } else {
                let b = self.d[self.pos];
                if b == 0xff {
                    let n = self.d.get(self.pos + 1).copied().unwrap_or(0xd9);
                    if n == 0 { self.pos += 2; 0xff }
                    else if n == 0xff { self.pos += 1; continue; }
                    else { self.marker = Some(n); self.overrun += 1; 0 }
                } else { self.pos += 1; b }
            };
            self.buf |= (byte as u32) << (24 - self.cnt);
            self.cnt += 8;
        }
    }
    fn get(&mut self, n: u32) -> Res<u32> {
        if n == 0 { return Ok(0); }
        if self.cnt < n { self.fill(); }
        self.check()?;
        let v = self.buf >> (32 - n);
        self.buf <<= n;
        self.cnt -= n;
        Ok(v)
    }
    /// A scan may run out of data a little (the padding bits); far past that it is truncated.
    fn check(&self) -> Res<()> { if self.overrun > 64 { trunc("the JPEG scan data ends early") } else { Ok(()) } }
    fn bit(&mut self) -> Res<u32> { self.get(1) }
    fn decode(&mut self, t: &HuffTable) -> Res<u8> {
        if self.cnt < 16 { self.fill(); }
        self.check()?;
        let e = t.fast[(self.buf >> 23) as usize];
        if e != 0 {
            let l = (e >> 8) as u32;
            self.buf <<= l;
            self.cnt -= l;
            return Ok((e & 0xff) as u8);
        }
        let mut l = 10usize;
        loop {
            if l > 16 { return corrupt("invalid JPEG Huffman code"); }
            let code = (self.buf >> (32 - l as u32)) as i32;
            if t.maxcode[l] >= 0 && code <= t.maxcode[l] && code >= t.mincode[l] { break; }
            l += 1;
        }
        let code = (self.buf >> (32 - l as u32)) as i32;
        self.buf <<= l as u32;
        self.cnt -= l as u32;
        t.vals.get((t.valptr[l] + code - t.mincode[l]) as usize).copied().ok_or_else(|| ImageError::Corrupt("JPEG Huffman index out of range".into()))
    }
    fn extend(&mut self, s: u32) -> Res<i32> {
        if s == 0 { return Ok(0); }
        if s > 16 { return corrupt("JPEG coefficient category out of range"); }
        let v = self.get(s)? as i32;
        Ok(if v < (1 << (s - 1)) { v - (1 << s) + 1 } else { v })
    }
    /// Skip to the byte boundary and swallow the restart marker that must follow.
    fn restart(&mut self, expect: u8) -> Res<()> {
        self.buf = 0;
        self.cnt = 0;
        self.overrun = 0;
        if self.marker.is_none() {
            // find the next marker (data not consumed because of bit buffering)
            while self.pos + 1 < self.d.len() && !(self.d[self.pos] == 0xff && self.d[self.pos + 1] != 0 && self.d[self.pos + 1] != 0xff) { self.pos += 1; }
            if self.pos + 1 < self.d.len() { self.marker = Some(self.d[self.pos + 1]); }
        }
        match self.marker {
            Some(m) if m == 0xd0 + expect => { self.marker = None; self.pos += 2; Ok(()) }
            Some(m) if (0xd0..=0xd7).contains(&m) => corrupt("JPEG restart marker out of sequence"),
            _ => corrupt("JPEG restart marker expected"),
        }
    }
}

#[derive(Clone)]
struct Comp { id: u8, h: usize, v: usize, tq: usize, td: usize, ta: usize, bw: usize, bh: usize, coef: Vec<i16>, pred: i32 }

struct Jpeg<'a> {
    d: &'a [u8], w: usize, h: usize, comps: Vec<Comp>, qt: [[u16; 64]; 4], dc: [Option<HuffTable>; 4], ac: [Option<HuffTable>; 4],
    progressive: bool, restart: usize, hmax: usize, vmax: usize, mcux: usize, mcuy: usize, adobe: Option<u8>, eobrun: u32,
}

/// JPEG: baseline, extended sequential and progressive Huffman-coded, 8-bit, greyscale or three
/// components (YCbCr, or RGB when an Adobe marker says so), any sampling factors up to 4×4
/// (2×1 and 2×2 chroma use the smooth "fancy" upsampling, others replicate), restart intervals.
pub fn decode_jpeg(d: &[u8], lim: &Limits) -> Res<Image> {
    if !d.starts_with(&[0xff, 0xd8]) { return corrupt("missing JPEG start marker"); }
    let mut j = Jpeg { d, w: 0, h: 0, comps: vec![], qt: [[0; 64]; 4], dc: [None, None, None, None], ac: [None, None, None, None], progressive: false,
        restart: 0, hmax: 1, vmax: 1, mcux: 0, mcuy: 0, adobe: None, eobrun: 0 };
    let mut pos = 2;
    let mut frame = false;
    let mut scans = 0;
    loop {
        // find the next marker
        while pos < d.len() && d[pos] != 0xff { pos += 1; }
        while pos < d.len() && d[pos] == 0xff { pos += 1; }
        let Some(&m) = d.get(pos) else { break; };
        pos += 1;
        if m == 0xd9 { break; }
        if m == 0 || (0xd0..=0xd7).contains(&m) || m == 0x01 { continue; }
        let Some(len) = le16_be(d, pos) else { return trunc("a JPEG segment is cut short"); };
        let len = len as usize;
        if len < 2 { return corrupt("JPEG segment shorter than its length field"); }
        let Some(seg) = d.get(pos + 2..pos + len) else { return trunc("a JPEG segment runs past the end of the file"); };
        match m {
            0xc0..=0xc2 => {
                if frame { return corrupt("a second JPEG frame header"); }
                if seg.len() < 6 { return corrupt("short JPEG frame header"); }
                if seg[0] != 8 { return unsupported(format!("JPEG with {}-bit samples", seg[0])); }
                j.h = u16::from_be_bytes([seg[1], seg[2]]) as usize;
                j.w = u16::from_be_bytes([seg[3], seg[4]]) as usize;
                if j.h == 0 { return unsupported("JPEG whose height is given by a later DNL marker"); }
                let n = seg[5] as usize;
                if n != 1 && n != 3 { return unsupported(format!("JPEG with {n} components")); }
                if seg.len() < 6 + 3 * n { return corrupt("short JPEG frame header"); }
                lim.check(j.w, j.h)?;
                for i in 0..n {
                    let (id, hv, tq) = (seg[6 + 3 * i], seg[7 + 3 * i], seg[8 + 3 * i] as usize);
                    let (h, v) = ((hv >> 4) as usize, (hv & 15) as usize);
                    if !(1..=4).contains(&h) || !(1..=4).contains(&v) || tq > 3 { return corrupt("bad JPEG component description"); }
                    j.comps.push(Comp { id, h, v, tq, td: 0, ta: 0, bw: 0, bh: 0, coef: vec![], pred: 0 });
                }
                j.hmax = j.comps.iter().map(|c| c.h).max().unwrap_or(1);
                j.vmax = j.comps.iter().map(|c| c.v).max().unwrap_or(1);
                j.mcux = j.w.div_ceil(8 * j.hmax);
                j.mcuy = j.h.div_ceil(8 * j.vmax);
                j.progressive = m == 0xc2;
                let mut padded = 0u64;
                for c in &mut j.comps {
                    c.bw = j.mcux * c.h;
                    c.bh = j.mcuy * c.v;
                    padded += (c.bw * c.bh * 64) as u64;
                }
                if padded > lim.max_pixels.saturating_mul(4) { return Err(ImageError::TooLarge("the JPEG's coefficient planes exceed the pixel limit".into())); }
                for c in &mut j.comps { c.coef = vec![0i16; c.bw * c.bh * 64]; }
                frame = true;
            }
            0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => return unsupported("JPEG lossless, hierarchical or arithmetic-coded"),
            0xc4 => {
                let mut s = seg;
                while !s.is_empty() {
                    if s.len() < 17 { return corrupt("short JPEG Huffman table definition"); }
                    let (tc, th) = (s[0] >> 4, (s[0] & 15) as usize);
                    if th > 3 || tc > 1 { return corrupt("bad JPEG Huffman table id"); }
                    let counts: [u8; 16] = s[1..17].try_into().unwrap_or([0; 16]);
                    let n: usize = counts.iter().map(|&c| c as usize).sum();
                    let Some(vals) = s.get(17..17 + n) else { return corrupt("short JPEG Huffman table"); };
                    let t = HuffTable::new(&counts, vals)?;
                    if tc == 0 { j.dc[th] = Some(t); } else { j.ac[th] = Some(t); }
                    s = &s[17 + n..];
                }
            }
            0xdb => {
                let mut s = seg;
                while !s.is_empty() {
                    let (pq, tq) = (s[0] >> 4, (s[0] & 15) as usize);
                    if tq > 3 || pq > 1 { return corrupt("bad JPEG quantisation table id"); }
                    let n = if pq == 0 { 64 } else { 128 };
                    let Some(body) = s.get(1..1 + n) else { return corrupt("short JPEG quantisation table"); };
                    for (k, &z) in ZIGZAG.iter().enumerate() {
                        j.qt[tq][z as usize] = if pq == 0 { body[k] as u16 } else { u16::from_be_bytes([body[2 * k], body[2 * k + 1]]) };
                    }
                    s = &s[1 + n..];
                }
            }
            0xdd => { if seg.len() < 2 { return corrupt("short JPEG restart interval"); } j.restart = u16::from_be_bytes([seg[0], seg[1]]) as usize; }
            0xee => { if seg.len() >= 12 && seg.starts_with(b"Adobe") { j.adobe = Some(seg[11]); } }
            0xda => {
                if !frame { return corrupt("a JPEG scan before the frame header"); }
                scans += 1;
                if scans > 4096 { return corrupt("too many JPEG scans"); }
                pos = j.scan(seg, pos + len)?;
                continue;
            }
            _ => {}
        }
        pos += len;
    }
    if !frame { return corrupt("JPEG without a frame header"); }
    if scans == 0 { return corrupt("JPEG without image data"); }
    j.finish(lim)
}

fn le16_be(d: &[u8], o: usize) -> Option<u16> { Some(u16::from_be_bytes([*d.get(o)?, *d.get(o + 1)?])) }

impl Jpeg<'_> {
    /// Decode one scan whose header is `seg`; entropy data starts at `start`. Returns the position
    /// just past the entropy-coded segment.
    fn scan(&mut self, seg: &[u8], start: usize) -> Res<usize> {
        let ns = *seg.first().ok_or_else(|| ImageError::Corrupt("empty JPEG scan header".into()))? as usize;
        if ns == 0 || ns > self.comps.len() || seg.len() < 1 + 2 * ns + 3 { return corrupt("bad JPEG scan header"); }
        let mut idx = vec![];
        for i in 0..ns {
            let (cid, t) = (seg[1 + 2 * i], seg[2 + 2 * i]);
            let Some(ci) = self.comps.iter().position(|c| c.id == cid) else { return corrupt("JPEG scan names an unknown component"); };
            if idx.contains(&ci) { return corrupt("JPEG scan repeats a component"); }
            self.comps[ci].td = (t >> 4) as usize;
            self.comps[ci].ta = (t & 15) as usize;
            if self.comps[ci].td > 3 || self.comps[ci].ta > 3 { return corrupt("bad JPEG Huffman table selector"); }
            idx.push(ci);
        }
        let (ss, se, a) = (seg[1 + 2 * ns] as usize, seg[2 + 2 * ns] as usize, seg[3 + 2 * ns]);
        let (ah, al) = ((a >> 4) as u32, (a & 15) as u32);
        if self.progressive && (ss > se || se > 63 || al > 13 || (ss == 0 && se != 0) || (ss > 0 && ns != 1)) { return corrupt("bad progressive JPEG scan parameters"); }
        let (ss, se, ah, al) = if self.progressive { (ss, se, ah, al) } else { (0, 63, 0, 0) };
        let mut br = BitReader::new(self.d, start);
        let mut rst = 0u8;
        for &ci in &idx { self.comps[ci].pred = 0; }
        self.eobrun = 0;
        // block geometry
        let single = ns == 1;
        let (units_x, units_y) = if single {
            let c = &self.comps[idx[0]];
            ((self.w * c.h).div_ceil(self.hmax).div_ceil(8), (self.h * c.v).div_ceil(self.vmax).div_ceil(8))
        } else { (self.mcux, self.mcuy) };
        let total = units_x * units_y;
        let mut count = 0usize;
        for u in 0..total {
            if self.restart > 0 && count == self.restart {
                br.restart(rst)?;
                rst = (rst + 1) & 7;
                for &ci in &idx { self.comps[ci].pred = 0; }
                self.eobrun = 0;
                count = 0;
            }
            count += 1;
            let (ux, uy) = (u % units_x, u / units_x);
            for &ci in &idx {
                let (bh, bv) = if single { (1, 1) } else { (self.comps[ci].h, self.comps[ci].v) };
                for by in 0..bv {
                    for bx in 0..bh {
                        let (x, y) = if single { (ux, uy) } else { (ux * bh + bx, uy * bv + by) };
                        self.block(&mut br, ci, x, y, ss, se, ah, al)?;
                    }
                }
            }
        }
        // position after the entropy-coded data: where the decoder stopped, then on to the next marker
        let mut p = br.pos;
        if br.marker.is_none() {
            while p + 1 < self.d.len() && !(self.d[p] == 0xff && self.d[p + 1] != 0 && !(0xd0..=0xd7).contains(&self.d[p + 1])) { p += 1; }
        }
        Ok(p)
    }

    #[allow(clippy::too_many_arguments)]
    fn block(&mut self, br: &mut BitReader, ci: usize, bx: usize, by: usize, ss: usize, se: usize, ah: u32, al: u32) -> Res<()> {
        let c = &mut self.comps[ci];
        if bx >= c.bw || by >= c.bh { return corrupt("JPEG block outside its plane"); }
        let off = (by * c.bw + bx) * 64;
        let coef = &mut c.coef[off..off + 64];
        if !self.progressive {
            let dc = self.dc[c.td].as_ref().ok_or_else(|| ImageError::Corrupt("JPEG scan uses a missing DC table".into()))?;
            let ac = self.ac[c.ta].as_ref().ok_or_else(|| ImageError::Corrupt("JPEG scan uses a missing AC table".into()))?;
            let s = br.decode(dc)? as u32;
            c.pred = c.pred.wrapping_add(br.extend(s)?);
            coef[0] = c.pred.clamp(-32768, 32767) as i16;
            let mut k = 1;
            while k < 64 {
                let rs = br.decode(ac)?;
                let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                if s == 0 {
                    if r == 15 { k += 16; continue; }
                    break;
                }
                k += r;
                if k > 63 { return corrupt("JPEG AC coefficient index past the block"); }
                coef[ZIGZAG[k] as usize] = br.extend(s)?.clamp(-32768, 32767) as i16;
                k += 1;
            }
            return Ok(());
        }
        if ss == 0 {
            if ah == 0 {
                let dc = self.dc[c.td].as_ref().ok_or_else(|| ImageError::Corrupt("JPEG scan uses a missing DC table".into()))?;
                let s = br.decode(dc)? as u32;
                c.pred = c.pred.wrapping_add(br.extend(s)?);
                coef[0] = (c.pred.clamp(-32768, 32767) << al).clamp(-32768, 32767) as i16;
            } else if br.bit()? == 1 { coef[0] |= (1i32 << al) as i16; }
            return Ok(());
        }
        let ac = self.ac[c.ta].as_ref().ok_or_else(|| ImageError::Corrupt("JPEG scan uses a missing AC table".into()))?;
        if ah == 0 {
            if self.eobrun > 0 { self.eobrun -= 1; return Ok(()); }
            let mut k = ss;
            while k <= se {
                let rs = br.decode(ac)?;
                let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                if s == 0 {
                    if r < 15 {
                        self.eobrun = (1u32 << r) - 1;
                        if r > 0 { self.eobrun += br.get(r as u32)?; }
                        break;
                    }
                    k += 16;
                    continue;
                }
                k += r;
                if k > 63 { return corrupt("JPEG AC coefficient index past the block"); }
                coef[ZIGZAG[k] as usize] = (br.extend(s)? << al).clamp(-32768, 32767) as i16;
                k += 1;
            }
            return Ok(());
        }
        // successive-approximation refinement of AC coefficients
        let (p1, m1) = (1i16 << al, -1i16 << al);
        let mut k = ss;
        if self.eobrun == 0 {
            while k <= se {
                let rs = br.decode(ac)?;
                let (mut r, s) = ((rs >> 4) as i32, rs & 15);
                let mut value = 0i16;
                if s != 0 {
                    if s != 1 { return corrupt("bad JPEG refinement symbol"); }
                    value = if br.bit()? == 1 { p1 } else { m1 };
                } else if r != 15 {
                    self.eobrun = 1u32 << r;
                    if r > 0 { self.eobrun += br.get(r as u32)?; }
                    break;
                }
                while k <= se {
                    let z = ZIGZAG[k] as usize;
                    if coef[z] != 0 {
                        if br.bit()? == 1 && (coef[z] & p1) == 0 { coef[z] = coef[z].wrapping_add(if coef[z] >= 0 { p1 } else { m1 }); }
                    } else {
                        r -= 1;
                        if r < 0 { break; }
                    }
                    k += 1;
                }
                if value != 0 && k <= se { coef[ZIGZAG[k] as usize] = value; }
                k += 1;
            }
        }
        if self.eobrun > 0 {
            while k <= se {
                let z = ZIGZAG[k] as usize;
                if coef[z] != 0 && br.bit()? == 1 && (coef[z] & p1) == 0 { coef[z] = coef[z].wrapping_add(if coef[z] >= 0 { p1 } else { m1 }); }
                k += 1;
            }
            self.eobrun -= 1;
        }
        Ok(())
    }

    /// Dequantise, inverse DCT, upsample and convert to RGB.
    fn finish(self, lim: &Limits) -> Res<Image> {
        let (w, h) = (self.w, self.h);
        let mut out = alloc_rgb(w, h, lim)?;
        let basis = idct_basis();
        // spatial planes, full padded size per component
        let mut planes: Vec<(Vec<u8>, usize, usize)> = vec![];
        for c in &self.comps {
            let (pw, ph) = (c.bw * 8, c.bh * 8);
            let mut plane = vec![0u8; pw * ph];
            let q = &self.qt[c.tq];
            let mut blk = [0f32; 64];
            let mut tmp = [0f32; 64];
            for by in 0..c.bh {
                for bx in 0..c.bw {
                    let co = &c.coef[(by * c.bw + bx) * 64..][..64];
                    let only_dc = co[1..].iter().all(|&v| v == 0);
                    if only_dc {
                        let v = (co[0] as f32 * q[0] as f32 * 0.125 + 128.0).round().clamp(0.0, 255.0) as u8;
                        for y in 0..8 { plane[(by * 8 + y) * pw + bx * 8..][..8].fill(v); }
                        continue;
                    }
                    for i in 0..64 { blk[i] = co[i] as f32 * q[i] as f32; }
                    // rows then columns: out[y][x] = sum_u sum_v C(u)C(v) in[v][u] cos cos
                    for v in 0..8 { for x in 0..8 {
                        let mut s = 0.0;
                        for u in 0..8 { s += basis[x][u] * blk[v * 8 + u]; }
                        tmp[v * 8 + x] = s;
                    } }
                    for x in 0..8 { for y in 0..8 {
                        let mut s = 0.0;
                        for v in 0..8 { s += basis[y][v] * tmp[v * 8 + x]; }
                        plane[(by * 8 + y) * pw + bx * 8 + x] = (s + 128.0).round().clamp(0.0, 255.0) as u8;
                    } }
                }
            }
            planes.push((plane, pw, ph));
        }
        // upsample every component to full resolution
        let mut full: Vec<Vec<u8>> = vec![];
        for (c, (plane, pw, _ph)) in self.comps.iter().zip(&planes) {
            let cw = (w * c.h).div_ceil(self.hmax);
            let ch = (h * c.v).div_ceil(self.vmax);
            full.push(upsample(plane, *pw, cw, ch, c.h, c.v, self.hmax, self.vmax, w, h));
        }
        if full.len() == 1 {
            for i in 0..w * h { out[i * 3..i * 3 + 3].fill(full[0][i]); }
        } else {
            let rgb_direct = self.adobe == Some(0) || (self.comps[0].id == b'R' && self.comps[1].id == b'G' && self.comps[2].id == b'B');
            for i in 0..w * h {
                let (a, b2, c) = (full[0][i], full[1][i], full[2][i]);
                let px = if rgb_direct { [a, b2, c] } else { ycc_to_rgb(a, b2, c) };
                out[i * 3..i * 3 + 3].copy_from_slice(&px);
            }
        }
        Ok(Image { width: w, height: h, rgb: out })
    }
}

fn ycc_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let (y, cb, cr) = (y as f32, cb as f32 - 128.0, cr as f32 - 128.0);
    let r = y + 1.402 * cr;
    let g = y - 0.344_136 * cb - 0.714_136 * cr;
    let b = y + 1.772 * cb;
    [r.round().clamp(0.0, 255.0) as u8, g.round().clamp(0.0, 255.0) as u8, b.round().clamp(0.0, 255.0) as u8]
}

/// `basis[x][u] = C(u)/2 · cos((2x+1)uπ/16)`.
fn idct_basis() -> [[f32; 8]; 8] {
    let mut t = [[0f32; 8]; 8];
    for (x, row) in t.iter_mut().enumerate() {
        for (u, v) in row.iter_mut().enumerate() {
            let cu = if u == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
            *v = (0.5 * cu * (((2 * x + 1) as f64) * u as f64 * std::f64::consts::PI / 16.0).cos()) as f32;
        }
    }
    t
}

#[allow(clippy::too_many_arguments)]
fn upsample(plane: &[u8], pw: usize, cw: usize, ch: usize, ch_h: usize, ch_v: usize, hmax: usize, vmax: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let (fx, fy) = (hmax / ch_h.max(1), vmax / ch_v.max(1));
    let exact = hmax.is_multiple_of(ch_h) && vmax.is_multiple_of(ch_v);
    if exact && fx == 1 && fy == 1 {
        for y in 0..h { out[y * w..(y + 1) * w].copy_from_slice(&plane[y * pw..y * pw + w]); }
    } else if exact && fx == 2 && fy == 1 {
        for y in 0..h {
            let row = &plane[y * pw..y * pw + cw];
            for x in 0..w {
                let i = x / 2;
                out[y * w + x] = if cw == 1 { row[0] } else if x % 2 == 0 {
                    if i == 0 { row[0] } else { ((3 * row[i] as u32 + row[i - 1] as u32 + 1) >> 2) as u8 }
                } else if i + 1 >= cw { row[i] } else { ((3 * row[i] as u32 + row[i + 1] as u32 + 2) >> 2) as u8 };
            }
        }
    } else if exact && fx == 2 && fy == 2 {
        let mut col = vec![0u32; cw];
        for y in 0..h {
            let cy = (y / 2).min(ch - 1);
            let other = if y % 2 == 0 { cy.saturating_sub(1) } else { (cy + 1).min(ch - 1) };
            let (r0, r1) = (&plane[cy * pw..cy * pw + cw], &plane[other * pw..other * pw + cw]);
            for i in 0..cw { col[i] = 3 * r0[i] as u32 + r1[i] as u32; }
            for x in 0..w {
                let i = (x / 2).min(cw - 1);
                let v = if cw == 1 { (col[0] * 4 + if x % 2 == 0 { 8 } else { 7 }) >> 4 }
                else if x % 2 == 0 { if i == 0 { (col[0] * 4 + 8) >> 4 } else { (3 * col[i] + col[i - 1] + 8) >> 4 } }
                else if i + 1 >= cw { (col[i] * 4 + 7) >> 4 } else { (3 * col[i] + col[i + 1] + 7) >> 4 };
                out[y * w + x] = v as u8;
            }
        }
    } else {
        for y in 0..h {
            let cy = (y * ch_v / vmax).min(ch.max(1) - 1);
            for x in 0..w {
                let cx = (x * ch_h / hmax).min(cw.max(1) - 1);
                out[y * w + x] = plane[cy * pw + cx];
            }
        }
    }
    out
}

// -------------------------------------------------------------------------------------- resampling

/// How [`resize`] weighs source pixels. All but `Nearest` widen their support when shrinking, so
/// large reductions average instead of skipping pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter { Nearest, Bilinear, Bicubic, Area, Lanczos3 }

impl Filter {
    fn support(self) -> f64 { match self { Filter::Nearest => 0.5, Filter::Bilinear => 1.0, Filter::Bicubic => 2.0, Filter::Area => 0.5, Filter::Lanczos3 => 3.0 } }
    fn weight(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Nearest | Filter::Area => if x < 0.5 { 1.0 } else if x == 0.5 { 0.5 } else { 0.0 },
            Filter::Bilinear => (1.0 - x).max(0.0),
            Filter::Bicubic => {
                const A: f64 = -0.5;
                if x < 1.0 { ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0 } else if x < 2.0 { (((x - 5.0) * x + 8.0) * x - 4.0) * A } else { 0.0 }
            }
            Filter::Lanczos3 => {
                if x == 0.0 { 1.0 } else if x < 3.0 {
                    let px = std::f64::consts::PI * x;
                    3.0 * px.sin() * (px / 3.0).sin() / (px * px)
                } else { 0.0 }
            }
        }
    }
}

/// For each output index, the first source index and the normalised weights.
fn coefficients(src: usize, dst: usize, f: Filter) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    let fscale = scale.max(1.0);
    let support = f.support() * fscale;
    (0..dst).map(|i| {
        if f == Filter::Nearest {
            return (((i as f64 + 0.5) * scale).floor().min(src as f64 - 1.0) as usize, vec![1.0]);
        }
        let center = (i as f64 + 0.5) * scale;
        let lo = ((center - support).floor().max(0.0)) as usize;
        let hi = ((center + support).ceil().min(src as f64)) as usize;
        let hi = hi.max(lo + 1).min(src);
        let lo = lo.min(hi - 1);
        let mut w: Vec<f64> = (lo..hi).map(|s| f.weight((s as f64 + 0.5 - center) / fscale)).collect();
        let sum: f64 = w.iter().sum();
        if sum.abs() < 1e-12 { let mid = (w.len() - 1) / 2; w.iter_mut().for_each(|v| *v = 0.0); w[mid] = 1.0; } else { w.iter_mut().for_each(|v| *v /= sum); }
        (lo, w.into_iter().map(|v| v as f32).collect())
    }).collect()
}

/// Resample to `w × h` (the sizes may be larger or smaller than the source). Separable, with
/// 32-bit float intermediate values, rounded back to 8 bits at the end.
pub fn resize(img: &Image, w: usize, h: usize, f: Filter) -> Res<Image> {
    resize_with(img, w, h, f, &Limits::default())
}
pub fn resize_with(img: &Image, w: usize, h: usize, f: Filter, lim: &Limits) -> Res<Image> {
    lim.check(w, h)?;
    if img.width == 0 || img.height == 0 || img.rgb.len() != img.width * img.height * 3 { return corrupt("the image's buffer does not match its size"); }
    if (img.width, img.height) == (w, h) { return Ok(img.clone()); }
    let (sw, sh) = (img.width, img.height);
    let cx = coefficients(sw, w, f);
    let cy = coefficients(sh, h, f);
    // horizontal pass over every source row
    let mut mid = vec![0f32; w * sh * 3];
    for y in 0..sh {
        let row = &img.rgb[y * sw * 3..(y + 1) * sw * 3];
        for (x, (lo, ws)) in cx.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (k, &wt) in ws.iter().enumerate() {
                let p = (lo + k) * 3;
                acc[0] += wt * row[p] as f32;
                acc[1] += wt * row[p + 1] as f32;
                acc[2] += wt * row[p + 2] as f32;
            }
            mid[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&acc);
        }
    }
    let mut out = vec![0u8; w * h * 3];
    for (y, (lo, ws)) in cy.iter().enumerate() {
        for x in 0..w * 3 {
            let mut acc = 0f32;
            for (k, &wt) in ws.iter().enumerate() { acc += wt * mid[(lo + k) * w * 3 + x]; }
            out[y * w * 3 + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(Image { width: w, height: h, rgb: out })
}

/// The `w × h` rectangle whose top-left corner is `(x, y)`; the rectangle must lie inside.
pub fn crop(img: &Image, x: usize, y: usize, w: usize, h: usize) -> Res<Image> {
    if w == 0 || h == 0 || x.checked_add(w).is_none_or(|e| e > img.width) || y.checked_add(h).is_none_or(|e| e > img.height) { return corrupt("crop rectangle outside the image"); }
    let mut rgb = Vec::with_capacity(w * h * 3);
    for r in y..y + h { rgb.extend_from_slice(&img.rgb[(r * img.width + x) * 3..(r * img.width + x + w) * 3]); }
    Ok(Image { width: w, height: h, rgb })
}

/// Centre crop to `w × h` (the image is first enlarged by padding with `bg` if it is smaller).
pub fn center_crop(img: &Image, w: usize, h: usize, bg: [u8; 3]) -> Res<Image> {
    let padded = if img.width < w || img.height < h { pad_to(img, w.max(img.width), h.max(img.height), bg)? } else { img.clone() };
    crop(&padded, (padded.width - w) / 2, (padded.height - h) / 2, w, h)
}

/// Centre the image on a `w × h` canvas of colour `bg` (the image must fit).
pub fn pad_to(img: &Image, w: usize, h: usize, bg: [u8; 3]) -> Res<Image> {
    if img.width > w || img.height > h { return corrupt("the image is larger than the canvas"); }
    Limits::default().check(w, h)?;
    let mut rgb: Vec<u8> = (0..w * h).flat_map(|_| bg).collect();
    let (ox, oy) = ((w - img.width) / 2, (h - img.height) / 2);
    for y in 0..img.height {
        rgb[((oy + y) * w + ox) * 3..((oy + y) * w + ox + img.width) * 3].copy_from_slice(&img.rgb[y * img.width * 3..(y + 1) * img.width * 3]);
    }
    Ok(Image { width: w, height: h, rgb })
}

/// Pad to a square on the longer side, as LLaVA-style models do before resizing.
pub fn pad_to_square(img: &Image, bg: [u8; 3]) -> Res<Image> { let s = img.width.max(img.height); pad_to(img, s, s, bg) }

/// Scale to fit inside `w × h` keeping the aspect ratio.
pub fn fit_inside(img: &Image, w: usize, h: usize, f: Filter) -> Res<Image> {
    let s = (w as f64 / img.width as f64).min(h as f64 / img.height as f64);
    let (nw, nh) = (((img.width as f64 * s).round() as usize).clamp(1, w), ((img.height as f64 * s).round() as usize).clamp(1, h));
    resize(img, nw, nh, f)
}

/// Scale so the shorter side is `short` (aspect ratio kept).
pub fn resize_short_side(img: &Image, short: usize, f: Filter) -> Res<Image> {
    let s = short as f64 / img.width.min(img.height) as f64;
    resize(img, ((img.width as f64 * s).round() as usize).max(1), ((img.height as f64 * s).round() as usize).max(1), f)
}

/// Planar `[3][h][w]` floats, `(value / 255 - mean) / std` per channel.
pub fn to_chw(img: &Image, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let n = img.width * img.height;
    let mut out = vec![0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 { out[c * n + i] = (img.rgb[i * 3 + c] as f32 / 255.0 - mean[c]) / std[c]; }
    }
    out
}

// ------------------------------------------------------------------------------------- encoders

/// Binary PPM (P6).
pub fn encode_ppm(img: &Image) -> Vec<u8> {
    let mut v = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    v.extend_from_slice(&img.rgb);
    v
}

/// 24-bit bottom-up BMP.
pub fn encode_bmp(img: &Image) -> Vec<u8> {
    let stride = (img.width * 3).div_ceil(4) * 4;
    let size = 54 + stride * img.height;
    let mut v = Vec::with_capacity(size);
    v.extend_from_slice(b"BM");
    v.extend_from_slice(&(size as u32).to_le_bytes());
    v.extend_from_slice(&[0; 4]);
    v.extend_from_slice(&54u32.to_le_bytes());
    v.extend_from_slice(&40u32.to_le_bytes());
    v.extend_from_slice(&(img.width as i32).to_le_bytes());
    v.extend_from_slice(&(img.height as i32).to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&24u16.to_le_bytes());
    v.extend_from_slice(&[0; 24]);
    for y in (0..img.height).rev() {
        for x in 0..img.width {
            let p = (y * img.width + x) * 3;
            v.extend_from_slice(&[img.rgb[p + 2], img.rgb[p + 1], img.rgb[p]]);
        }
        v.resize(v.len() + stride - img.width * 3, 0);
    }
    v
}

/// PNG with stored (uncompressed) deflate blocks and filter 0: large, but valid and simple.
pub fn encode_png(img: &Image) -> Vec<u8> {
    let stride = img.width * 3;
    let mut raw = Vec::with_capacity((stride + 1) * img.height);
    for y in 0..img.height { raw.push(0); raw.extend_from_slice(&img.rgb[y * stride..(y + 1) * stride]); }
    let mut z = vec![0x78, 0x01];
    let mut chunks = raw.chunks(65535).peekable();
    if raw.is_empty() { z.extend_from_slice(&[1, 0, 0, 0xff, 0xff]); }
    while let Some(c) = chunks.next() {
        z.push(if chunks.peek().is_none() { 1 } else { 0 });
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut chunk = |ty: &[u8; 4], body: &[u8]| {
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(ty);
        out.extend_from_slice(body);
        out.extend_from_slice(&crc32_update(crc32(ty), body).to_be_bytes());
    };
    let mut ihdr = vec![];
    ihdr.extend_from_slice(&(img.width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(img.height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    out
}

/// A deterministic test picture: smooth gradients plus a few hard edges.
pub fn synthetic(width: usize, height: usize, seed: u32) -> Image {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f32 / width.max(1) as f32, y as f32 / height.max(1) as f32);
            let edge = if (x / 8 + y / 8 + seed as usize).is_multiple_of(5) { 60.0 } else { 0.0 };
            rgb.push((fx * 200.0 + edge + (seed % 7) as f32 * 4.0).min(255.0) as u8);
            rgb.push((fy * 200.0 + (seed % 5) as f32 * 8.0).min(255.0) as u8);
            rgb.push((((fx + fy) * 0.5) * 220.0 + edge * 0.5).min(255.0) as u8);
        }
    }
    Image { width, height, rgb }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::identity_op, clippy::type_complexity)]
mod tests {
    use super::*;
    include!("imageio_test_vectors.in");

    fn unhex(s: &str) -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() }
    fn lim() -> Limits { Limits { max_pixels: 1 << 20, max_dim: 4096, max_input: 1 << 24 } }
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn below(&mut self, n: usize) -> usize { (self.next() % n.max(1) as u64) as usize }
    }
    fn mean_err(a: &[u8], b: &[u8]) -> (f64, u8) {
        assert_eq!(a.len(), b.len());
        let mut s = 0u64; let mut m = 0u8;
        for (x, y) in a.iter().zip(b) { let d = x.abs_diff(*y); s += d as u64; m = m.max(d); }
        (s as f64 / a.len().max(1) as f64, m)
    }

    // ---------------------------------------------------------------- checksums and inflate

    #[test]
    fn checksums_match_known_values() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b""), 0);
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
        assert_eq!(adler32(b""), 1);
        let big = vec![0xffu8; 100_000];
        assert_eq!(adler32(&big), { let (mut a, mut b) = (1u64, 0u64); for &x in &big { a = (a + x as u64) % 65521; b = (b + a) % 65521; } ((b << 16) | a) as u32 });
    }

    #[test]
    fn zlib_streams_from_a_real_compressor_decode() {
        let mut txt = Vec::new();
        for _ in 0..20 { txt.extend_from_slice(b"the quick brown fox jumps over the lazy dog. "); }
        for _ in 0..3 { txt.extend((0..=255u8).collect::<Vec<_>>()); }
        for _ in 0..10 { txt.extend_from_slice(b"abcabcabcabcabc"); }
        let out = zlib_decompress(&unhex(Z_TEXT), 1 << 20).unwrap();
        assert_eq!(out, txt);
        let zeros = zlib_decompress(&unhex(Z_ZEROS), 2_000_000).unwrap();
        assert_eq!(zeros.len(), 1_000_000);
        assert!(zeros.iter().all(|&b| b == 0));
    }

    #[test]
    fn a_deflate_bomb_is_refused_before_it_grows() {
        let z = unhex(Z_ZEROS);
        assert!(matches!(zlib_decompress(&z, 10_000), Err(ImageError::TooLarge(_))));
        assert!(matches!(zlib_decompress(&z, 999_999), Err(ImageError::TooLarge(_))));
        assert!(zlib_decompress(&z, 1_000_000).is_ok());
    }

    #[test]
    fn damaged_zlib_streams_are_errors() {
        let z = unhex(Z_TEXT);
        for n in 0..z.len() { assert!(zlib_decompress(&z[..n], 1 << 20).is_err(), "prefix {n}"); }
        let mut bad = z.clone();
        let l = bad.len();
        bad[l - 1] ^= 1;
        assert!(matches!(zlib_decompress(&bad, 1 << 20), Err(ImageError::Corrupt(_))));
        assert!(zlib_decompress(&[0x78, 0x9d, 0, 0, 0, 0], 100).is_err());
        assert!(zlib_decompress(&[0x78, 0x01, 0x07, 0, 0, 0, 0, 0], 100).is_err()); // reserved block type 3
        // stored block with a wrong complement
        assert!(zlib_decompress(&[0x78, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x41, 0, 0, 0, 0], 100).is_err());
    }

    // ----------------------------------------------------------------------------- PNG

    fn expect(name: &str) -> Vec<u8> { unhex(match name {
        "png_rgb" => E_PNG_RGB, "png_rgba" => E_PNG_RGBA, "png_pal" => E_PNG_PAL, "png_pal4" => E_PNG_PAL4, "png_gray8" => E_PNG_GRAY8,
        "png_gray1" => E_PNG_GRAY1, "png_gray16" => E_PNG_GRAY16, "png_interlaced" => E_PNG_INTERLACED, "png_rgba_interlaced" => E_PNG_RGBA_INTERLACED,
        "bmp_24" => E_BMP_24, "bmp_8" => E_BMP_8, "bmp_1" => E_BMP_1, "ppm" => E_PPM,
        "jpg_420" => E_JPG_420, "jpg_422" => E_JPG_422, "jpg_444" => E_JPG_444, "jpg_gray" => E_JPG_GRAY, "jpg_gray_prog" => E_JPG_GRAY_PROG,
        "jpg_prog420" => E_JPG_PROG420, "jpg_prog444" => E_JPG_PROG444, _ => panic!("{name}") }) }
    fn file(name: &str) -> Vec<u8> { unhex(match name {
        "png_rgb" => F_PNG_RGB, "png_rgba" => F_PNG_RGBA, "png_pal" => F_PNG_PAL, "png_pal4" => F_PNG_PAL4, "png_gray8" => F_PNG_GRAY8,
        "png_gray1" => F_PNG_GRAY1, "png_gray16" => F_PNG_GRAY16, "png_interlaced" => F_PNG_INTERLACED, "png_rgba_interlaced" => F_PNG_RGBA_INTERLACED,
        "bmp_24" => F_BMP_24, "bmp_8" => F_BMP_8, "bmp_1" => F_BMP_1, "ppm" => F_PPM,
        "jpg_420" => F_JPG_420, "jpg_422" => F_JPG_422, "jpg_444" => F_JPG_444, "jpg_gray" => F_JPG_GRAY, "jpg_gray_prog" => F_JPG_GRAY_PROG,
        "jpg_prog420" => F_JPG_PROG420, "jpg_prog444" => F_JPG_PROG444, _ => panic!("{name}") }) }
    const LOSSLESS: [&str; 13] = ["png_rgb", "png_rgba", "png_pal", "png_pal4", "png_gray8", "png_gray1", "png_gray16", "png_interlaced", "png_rgba_interlaced", "bmp_24", "bmp_8", "bmp_1", "ppm"];
    const JPEGS: [&str; 7] = ["jpg_420", "jpg_422", "jpg_444", "jpg_gray", "jpg_gray_prog", "jpg_prog420", "jpg_prog444"];

    #[test]
    fn lossless_formats_decode_exactly_like_the_reference() {
        for n in LOSSLESS {
            let img = decode(&file(n)).unwrap_or_else(|e| panic!("{n}: {e}"));
            assert_eq!((img.width, img.height), (13, 11), "{n}");
            assert_eq!(img.rgb, expect(n), "{n}");
        }
    }

    #[test]
    fn jpeg_matches_the_reference_decoder() {
        for n in JPEGS {
            let img = decode(&file(n)).unwrap_or_else(|e| panic!("{n}: {e}"));
            assert_eq!((img.width, img.height), (24, 20), "{n}");
            let (mean, max) = mean_err(&img.rgb, &expect(n));
            assert!(mean < 1.0 && max <= 12, "{n}: mean error {mean}, max {max}");
        }
    }

    #[test]
    fn progressive_and_sequential_encodings_of_one_picture_agree() {
        // jpg_420 and jpg_prog420 hold the same picture, entropy coded differently (the quantised
        // coefficients are identical, so the pixels must match exactly)
        let a = decode(&file("jpg_420")).unwrap();
        let b = decode(&file("jpg_prog420")).unwrap();
        assert_eq!(a.rgb, b.rgb);
        let a = decode(&file("jpg_gray")).unwrap();
        let b = decode(&file("jpg_gray_prog")).unwrap();
        assert_eq!(a.rgb, b.rgb);
    }

    fn png_chunk(out: &mut Vec<u8>, ty: &[u8; 4], body: &[u8]) {
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(ty);
        out.extend_from_slice(body);
        out.extend_from_slice(&crc32_update(crc32(ty), body).to_be_bytes());
    }
    fn png_with(w: u32, h: u32, depth: u8, ctype: u8, il: u8, idat_zlib: &[u8]) -> Vec<u8> {
        let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let mut ihdr = vec![];
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[depth, ctype, 0, 0, il]);
        png_chunk(&mut out, b"IHDR", &ihdr);
        png_chunk(&mut out, b"IDAT", idat_zlib);
        png_chunk(&mut out, b"IEND", &[]);
        out
    }
    fn stored_zlib(raw: &[u8]) -> Vec<u8> {
        let mut z = vec![0x78, 0x01];
        let mut it = raw.chunks(65535).peekable();
        if raw.is_empty() { z.extend_from_slice(&[1, 0, 0, 0xff, 0xff]); }
        while let Some(c) = it.next() {
            z.push(it.peek().is_none() as u8);
            z.extend_from_slice(&(c.len() as u16).to_le_bytes());
            z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
            z.extend_from_slice(c);
        }
        z.extend_from_slice(&adler32(raw).to_be_bytes());
        z
    }

    #[test]
    fn png_filters_are_all_undone() {
        // 4x4 RGB image; each row uses a different filter type, built from known pixels
        let (w, h) = (4usize, 5usize);
        let px: Vec<u8> = (0..w * h * 3).map(|i| ((i * 37 + (i / 7) * 11) % 251) as u8).collect();
        let stride = w * 3;
        let bpp = 3;
        let mut raw = vec![];
        for y in 0..h {
            let ft = (y % 5) as u8;
            raw.push(ft);
            for x in 0..stride {
                let cur = px[y * stride + x] as i32;
                let a = if x >= bpp { px[y * stride + x - bpp] as i32 } else { 0 };
                let b = if y > 0 { px[(y - 1) * stride + x] as i32 } else { 0 };
                let c = if y > 0 && x >= bpp { px[(y - 1) * stride + x - bpp] as i32 } else { 0 };
                let pred = match ft { 0 => 0, 1 => a, 2 => b, 3 => (a + b) / 2, _ => { let p = a + b - c; let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs()); if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c } } };
                raw.push((cur - pred) as u8);
            }
        }
        let png = png_with(w as u32, h as u32, 8, 2, 0, &stored_zlib(&raw));
        let img = decode(&png).unwrap();
        assert_eq!(img.rgb, px);
    }

    #[test]
    fn png_bit_depths_and_transparency_keys() {
        // 2-bit grey 5x1: values 0,1,2,3,1 with a colour key on value 2 (becomes the background)
        let raw = vec![0u8, 0b0001_1011, 0b0100_0000];
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let mut ihdr = vec![];
        ihdr.extend_from_slice(&5u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[2, 0, 0, 0, 0]);
        png_chunk(&mut png, b"IHDR", &ihdr);
        png_chunk(&mut png, b"tRNS", &[0, 2]);
        png_chunk(&mut png, b"IDAT", &stored_zlib(&raw));
        png_chunk(&mut png, b"IEND", &[]);
        let img = decode_with(&png, &lim(), [9, 8, 7]).unwrap();
        let g: Vec<u8> = img.rgb.chunks(3).map(|p| p[0]).collect();
        assert_eq!(g, vec![0, 85, 9, 255, 85]);
        assert_eq!(&img.rgb[6..9], &[9, 8, 7]);
    }

    #[test]
    fn png_headers_are_checked_before_anything_is_allocated() {
        let z = stored_zlib(&[0, 0, 0, 0]);
        // gigantic dimensions
        assert!(matches!(decode(&png_with(0x7fff_ffff, 0x7fff_ffff, 8, 6, 0, &z)), Err(ImageError::TooLarge(_))));
        assert!(matches!(decode(&png_with(65535, 65535, 8, 2, 1, &z)), Err(ImageError::TooLarge(_))));
        assert!(decode(&png_with(0, 5, 8, 2, 0, &z)).is_err());
        assert!(decode(&png_with(1, 1, 3, 2, 0, &z)).is_err()); // bad depth
        assert!(decode(&png_with(1, 1, 8, 5, 0, &z)).is_err()); // bad colour type
        assert!(decode(&png_with(1, 1, 8, 2, 2, &z)).is_err()); // bad interlace
        // a one-pixel picture whose data stream expands to a megabyte
        let bomb = png_with(1, 1, 8, 2, 0, &unhex(Z_ZEROS));
        assert!(matches!(decode(&bomb), Err(ImageError::TooLarge(_))));
        // data shorter than the picture
        assert!(decode(&png_with(10, 10, 8, 2, 0, &stored_zlib(&[0; 20]))).is_err());
        // palette image without a palette
        assert!(decode(&png_with(1, 1, 8, 3, 0, &stored_zlib(&[0, 0]))).is_err());
        // a custom limit is honoured
        let small = Limits { max_pixels: 100, max_dim: 100, max_input: 1 << 20 };
        assert!(matches!(decode_with(&file("png_rgb"), &small, [0; 3]), Err(ImageError::TooLarge(_))));
    }

    #[test]
    fn png_crc_errors_are_fatal_for_critical_chunks_only() {
        let mut p = file("png_rgb");
        p[8 + 8 + 2] ^= 1; // inside IHDR
        assert!(matches!(decode(&p), Err(ImageError::Corrupt(_))));
        // an ancillary chunk with a bad CRC is skipped
        let mut q = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let mut ihdr = vec![];
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        png_chunk(&mut q, b"IHDR", &ihdr);
        let at = q.len();
        png_chunk(&mut q, b"tEXt", b"k\0v");
        q[at + 8] ^= 1;
        png_chunk(&mut q, b"IDAT", &stored_zlib(&[0, 10, 20, 30]));
        png_chunk(&mut q, b"IEND", &[]);
        assert_eq!(decode(&q).unwrap().rgb, vec![10, 20, 30]);
    }

    // ----------------------------------------------------------------------------- BMP, PNM

    #[test]
    fn bmp_variants() {
        // 16-bit 5-6-5 with bit fields, top-down (negative height), 3x2
        let (w, h) = (3usize, 2usize);
        let mut v = b"BM".to_vec();
        let mut body = vec![];
        let pixels: [u16; 6] = [0xf800, 0x07e0, 0x001f, 0xffff, 0x0000, 0x8410];
        for y in 0..h { for x in 0..w { body.extend_from_slice(&pixels[y * w + x].to_le_bytes()); } body.extend_from_slice(&[0, 0]); }
        let off = 14 + 40 + 12;
        v.extend_from_slice(&((off + body.len()) as u32).to_le_bytes());
        v.extend_from_slice(&[0; 4]);
        v.extend_from_slice(&(off as u32).to_le_bytes());
        v.extend_from_slice(&40u32.to_le_bytes());
        v.extend_from_slice(&(w as i32).to_le_bytes());
        v.extend_from_slice(&(-(h as i32)).to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&[0; 20]);
        for m in [0xf800u32, 0x07e0, 0x001f] { v.extend_from_slice(&m.to_le_bytes()); }
        v.extend_from_slice(&body);
        let img = decode(&v).unwrap();
        assert_eq!((img.width, img.height), (3, 2));
        assert_eq!(&img.rgb[0..9], &[255, 0, 0, 0, 255, 0, 0, 0, 255]);
        assert_eq!(&img.rgb[9..18], &[255, 255, 255, 0, 0, 0, 132, 130, 132]);
    }

    #[test]
    fn bmp_rle8_runs_and_absolute_mode() {
        let mut v = b"BM".to_vec();
        // palette: 0 black, 1 red, 2 green, 3 blue
        let pal = [0u8, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0];
        // 4x2, bottom-up: first coded row is the bottom: run of 4 x colour 1; end of line; absolute 3 pixels (2,3,2) + pad; one pixel colour 3; end of bitmap
        let data = [4u8, 1, 0, 0, 0, 3, 2, 3, 2, 0, 1, 3, 0, 1];
        let off = 14 + 40 + pal.len();
        v.extend_from_slice(&((off + data.len()) as u32).to_le_bytes());
        v.extend_from_slice(&[0; 4]);
        v.extend_from_slice(&(off as u32).to_le_bytes());
        v.extend_from_slice(&40u32.to_le_bytes());
        v.extend_from_slice(&4i32.to_le_bytes());
        v.extend_from_slice(&2i32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&8u16.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&[0; 12]);
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(&[0; 4]);
        v.extend_from_slice(&pal);
        v.extend_from_slice(&data);
        let img = decode(&v).unwrap();
        // bottom row (last in rgb) is all red; top row is green, blue, green, blue
        assert_eq!(&img.rgb[12..24], &[255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]);
        assert_eq!(&img.rgb[0..12], &[0, 255, 0, 0, 0, 255, 0, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn bmp_hostile_headers() {
        let good = file("bmp_24");
        for n in 0..good.len() { assert!(decode_bmp(&good[..n], &lim(), [255; 3]).is_err(), "prefix {n}"); }
        let mut huge = good.clone();
        huge[18..22].copy_from_slice(&0x7fff_ffffu32.to_le_bytes());
        assert!(matches!(decode(&huge), Err(ImageError::TooLarge(_))));
        let mut neg = good.clone();
        neg[18..22].copy_from_slice(&(-5i32).to_le_bytes());
        assert!(decode(&neg).is_err());
        let mut bpp = good.clone();
        bpp[28] = 7;
        assert!(matches!(decode(&bpp), Err(ImageError::Unsupported(_))));
        let mut off = good;
        off[10..14].copy_from_slice(&0xffff_fff0u32.to_le_bytes());
        assert!(decode(&off).is_err());
    }

    #[test]
    fn pnm_variants() {
        // P1 bitmap with comments and no separators
        let img = decode(b"P1\n# a comment\n3 2\n010\n111\n").unwrap();
        assert_eq!(img.rgb.chunks(3).map(|p| p[0]).collect::<Vec<_>>(), vec![255, 0, 255, 0, 0, 0]);
        // P2 with maxval 15
        let img = decode(b"P2 2 2 15\n0 15\n5 10").unwrap();
        assert_eq!(img.rgb.chunks(3).map(|p| p[0]).collect::<Vec<_>>(), vec![0, 255, 85, 170]);
        // P5 16-bit
        let mut p5 = b"P5 2 1 65535\n".to_vec();
        p5.extend_from_slice(&[0xff, 0xff, 0x80, 0x00]);
        assert_eq!(decode(&p5).unwrap().rgb, vec![255, 255, 255, 128, 128, 128]);
        // P4 packed bits
        let mut p4 = b"P4 10 1\n".to_vec();
        p4.extend_from_slice(&[0b1010_0000, 0b1100_0000]);
        let img = decode(&p4).unwrap();
        assert_eq!(img.rgb.chunks(3).map(|p| p[0]).collect::<Vec<_>>(), vec![0, 255, 0, 255, 255, 255, 255, 255, 0, 0]);
        // errors
        assert!(decode(b"P6 2 2 255\n\x00\x01").is_err());
        assert!(decode(b"P3 1 1 255\n300 0 0").is_err());
        assert!(matches!(decode(b"P6 99999 99999 255\n"), Err(ImageError::TooLarge(_))));
        assert!(decode(b"P6 1 1 0\n").is_err());
        assert!(decode(b"P6 1 1 99999999999999999999\n").is_err());
        assert!(decode(b"P7\n").is_err());
    }

    #[test]
    fn nothing_recognisable_is_unsupported() {
        assert!(matches!(decode(b"GIF89a....."), Err(ImageError::Unsupported(_))));
        assert!(decode(&[]).is_err());
        assert_eq!(sniff(&file("jpg_420")), Some(Format::Jpeg));
        assert_eq!(sniff(&file("png_rgb")), Some(Format::Png));
        assert_eq!(sniff(&file("bmp_24")), Some(Format::Bmp));
        assert_eq!(sniff(&file("ppm")), Some(Format::Pnm));
        let tiny = Limits { max_pixels: 1 << 20, max_dim: 4096, max_input: 10 };
        assert!(matches!(decode_with(&file("png_rgb"), &tiny, [0; 3]), Err(ImageError::TooLarge(_))));
    }

    // -------------------------------------------------------------------------- encoders

    #[test]
    fn our_own_encoders_round_trip() {
        for (w, h) in [(1, 1), (3, 5), (37, 11), (200, 130)] {
            let img = synthetic(w, h, 3);
            assert_eq!(decode(&encode_png(&img)).unwrap(), img, "png {w}x{h}");
            assert_eq!(decode(&encode_bmp(&img)).unwrap(), img, "bmp {w}x{h}");
            assert_eq!(decode(&encode_ppm(&img)).unwrap(), img, "ppm {w}x{h}");
        }
    }

    // -------------------------------------------------------- a small JPEG encoder for the tests

    struct BitW { out: Vec<u8>, acc: u32, n: u32 }
    impl BitW {
        fn put(&mut self, code: u32, len: u32) {
            for i in (0..len).rev() {
                self.acc = (self.acc << 1) | ((code >> i) & 1);
                self.n += 1;
                if self.n == 8 { let b = self.acc as u8; self.out.push(b); if b == 0xff { self.out.push(0); } self.acc = 0; self.n = 0; }
            }
        }
        fn flush(&mut self) { while self.n != 0 { self.put(1, 1); } }
    }
    fn fdct(f: &[f32; 64]) -> [f32; 64] {
        let mut o = [0f32; 64];
        for v in 0..8 { for u in 0..8 {
            let mut s = 0.0f64;
            for y in 0..8 { for x in 0..8 {
                s += f[y * 8 + x] as f64 * (((2 * x + 1) * u) as f64 * std::f64::consts::PI / 16.0).cos() * (((2 * y + 1) * v) as f64 * std::f64::consts::PI / 16.0).cos();
            } }
            let c = |k: usize| if k == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
            o[v * 8 + u] = (0.25 * c(u) * c(v) * s) as f32;
        } }
        o
    }
    /// Baseline 4:4:4 (or grey) encoder. `deep` makes every Huffman code at least 10 bits long so
    /// the slow decoding path is used; `restart` is the restart interval in MCUs (0 = none).
    fn encode_jpeg(img: &Image, gray: bool, q: u16, restart: usize, deep: bool) -> Vec<u8> {
        let (w, h) = (img.width, img.height);
        let nc = if gray { 1 } else { 3 };
        let mut o = vec![0xff, 0xd8];
        let seg = |o: &mut Vec<u8>, m: u8, body: &[u8]| { o.extend_from_slice(&[0xff, m]); o.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes()); o.extend_from_slice(body); };
        let mut dqt = vec![0u8];
        dqt.extend(std::iter::repeat_n(q as u8, 64));
        seg(&mut o, 0xdb, &dqt);
        let mut sof = vec![8];
        sof.extend_from_slice(&(h as u16).to_be_bytes());
        sof.extend_from_slice(&(w as u16).to_be_bytes());
        sof.push(nc as u8);
        for c in 0..nc { sof.extend_from_slice(&[c as u8 + 1, 0x11, 0]); }
        seg(&mut o, 0xc0, &sof);
        // DC: 12 symbols, codes of 4 bits (or 11 bits when deep)
        let (dc_len, ac_split) = if deep { (11u32, 9usize) } else { (4, 7) };
        let mut dht = vec![0x00];
        let mut counts = [0u8; 16];
        counts[dc_len as usize - 1] = 12;
        dht.extend_from_slice(&counts);
        dht.extend(0..12u8);
        seg(&mut o, 0xc4, &dht);
        // AC: 256 symbols, 128 with codes of length ac_split+1 and 128 one bit longer
        let mut dht = vec![0x10];
        let mut counts = [0u8; 16];
        counts[ac_split] = 128;
        counts[ac_split + 1] = 128;
        dht.extend_from_slice(&counts);
        dht.extend((0..=255u8).collect::<Vec<_>>());
        seg(&mut o, 0xc4, &dht);
        let ac_code = |sym: usize| -> (u32, u32) {
            let l1 = ac_split as u32 + 1;
            if sym < 128 { (sym as u32, l1) } else { ((128 + (sym as u32 - 128)) , l1 + 1) }
        };
        if restart > 0 { seg(&mut o, 0xdd, &(restart as u16).to_be_bytes()); }
        let mut sos = vec![nc as u8];
        for c in 0..nc { sos.extend_from_slice(&[c as u8 + 1, 0x00]); }
        sos.extend_from_slice(&[0, 63, 0]);
        seg(&mut o, 0xda, &sos);
        // The canonical codes: DC symbol s gets code s; AC symbols get codes in order.
        let ac_code_canon = |sym: usize| -> (u32, u32) {
            let l1 = ac_split as u32 + 1;
            if sym < 128 { (sym as u32, l1) } else { (((128u32) << 1) + (sym as u32 - 128), l1 + 1) }
        };
        let _ = ac_code;
        let mut bw = BitW { out: vec![], acc: 0, n: 0 };
        let (mx, my) = (w.div_ceil(8), h.div_ceil(8));
        let mut pred = [0i32; 3];
        let mut rst = 0u8;
        for m in 0..mx * my {
            if restart > 0 && m > 0 && m % restart == 0 {
                bw.flush();
                bw.out.extend_from_slice(&[0xff, 0xd0 + rst]);
                rst = (rst + 1) & 7;
                pred = [0; 3];
            }
            let (bx, by) = (m % mx, m / mx);
            for c in 0..nc {
                let mut blk = [0f32; 64];
                for y in 0..8 { for x in 0..8 {
                    let (px, py) = ((bx * 8 + x).min(w - 1), (by * 8 + y).min(h - 1));
                    let p = (py * w + px) * 3;
                    let (r, g, b) = (img.rgb[p] as f32, img.rgb[p + 1] as f32, img.rgb[p + 2] as f32);
                    let v = if gray { 0.299 * r + 0.587 * g + 0.114 * b } else { match c {
                        0 => 0.299 * r + 0.587 * g + 0.114 * b,
                        1 => -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0,
                        _ => 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0,
                    } };
                    blk[y * 8 + x] = v - 128.0;
                } }
                let co = fdct(&blk);
                let mut zz = [0i32; 64];
                for (k, &z) in ZIGZAG.iter().enumerate() { zz[k] = (co[z as usize] / q as f32).round() as i32; }
                let cat = |v: i32| 32 - (v.unsigned_abs()).leading_zeros();
                let bits = |bw: &mut BitW, v: i32, s: u32| { if s > 0 { let u = if v < 0 { (v - 1) as u32 & ((1 << s) - 1) } else { v as u32 }; bw.put(u, s); } };
                let diff = zz[0] - pred[c];
                pred[c] = zz[0];
                let s = cat(diff);
                bw.put(s, dc_len);
                bits(&mut bw, diff, s);
                let mut run = 0;
                for &v in &zz[1..] {
                    if v == 0 { run += 1; continue; }
                    while run > 15 { let (c0, l0) = ac_code_canon(0xf0); bw.put(c0, l0); run -= 16; }
                    let s = cat(v);
                    let (c1, l1) = ac_code_canon(((run as usize) << 4) | s as usize);
                    bw.put(c1, l1);
                    bits(&mut bw, v, s);
                    run = 0;
                }
                if run > 0 { let (c0, l0) = ac_code_canon(0); bw.put(c0, l0); }
            }
        }
        bw.flush();
        o.extend_from_slice(&bw.out);
        o.extend_from_slice(&[0xff, 0xd9]);
        o
    }

    #[test]
    fn jpeg_encoder_round_trip_with_restart_intervals_and_long_codes() {
        let src = synthetic(37, 29, 2);
        for (gray, restart, deep) in [(false, 0, false), (false, 1, false), (false, 3, false), (false, 0, true), (false, 2, true), (true, 0, false), (true, 4, true)] {
            let j = encode_jpeg(&src, gray, 2, restart, deep);
            let img = decode(&j).unwrap_or_else(|e| panic!("gray {gray} restart {restart} deep {deep}: {e}"));
            assert_eq!((img.width, img.height), (37, 29));
            let reference: Vec<u8> = if gray {
                src.rgb.chunks(3).flat_map(|p| { let y = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).round() as u8; [y, y, y] }).collect()
            } else { src.rgb.clone() };
            let (mean, max) = mean_err(&img.rgb, &reference);
            assert!(mean < 2.0 && max < 16, "gray {gray} restart {restart} deep {deep}: mean {mean} max {max}");
        }
    }

    #[test]
    fn a_missing_or_wrong_restart_marker_is_an_error() {
        let src = synthetic(24, 16, 1);
        let j = encode_jpeg(&src, false, 2, 1, false);
        let pos = j.windows(2).position(|w| w == [0xff, 0xd0]).unwrap();
        let mut wrong = j.clone();
        wrong[pos + 1] = 0xd3;
        assert!(decode(&wrong).is_err());
        let mut gone = j.clone();
        gone[pos] = 0x11;
        gone[pos + 1] = 0x22;
        assert!(decode(&gone).is_err());
    }

    #[test]
    fn jpeg_headers_are_checked() {
        let good = file("jpg_420");
        // oversize frame
        let sof = good.windows(2).position(|w| w == [0xff, 0xc0]).unwrap();
        let mut big = good.clone();
        big[sof + 5..sof + 9].copy_from_slice(&[0xff, 0xff, 0xff, 0xff]);
        assert!(matches!(decode(&big), Err(ImageError::TooLarge(_))));
        let mut zero = good.clone();
        zero[sof + 5..sof + 7].copy_from_slice(&[0, 0]);
        assert!(decode(&zero).is_err());
        let mut prec = good.clone();
        prec[sof + 4] = 12;
        assert!(matches!(decode(&prec), Err(ImageError::Unsupported(_))));
        let mut arith = good.clone();
        arith[sof + 1] = 0xc9;
        assert!(matches!(decode(&arith), Err(ImageError::Unsupported(_))));
        assert!(decode(&[0xff, 0xd8]).is_err());
        assert!(decode(&[0xff, 0xd8, 0xff, 0xd9]).is_err());
    }

    // ------------------------------------------------------------------------------ fuzzing

    #[test]
    fn truncating_any_file_anywhere_never_panics() {
        for n in ["png_rgb", "png_rgba_interlaced", "png_pal4", "bmp_8", "bmp_1", "ppm", "jpg_420", "jpg_prog420", "jpg_gray_prog", "jpg_444"] {
            let f = file(n);
            for len in 0..f.len() { let _ = decode_with(&f[..len], &lim(), [255; 3]); }
        }
    }

    #[test]
    fn mutating_files_never_panics_or_hangs() {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let names = ["png_rgb", "png_rgba", "png_pal", "png_interlaced", "png_gray16", "bmp_24", "bmp_8", "ppm", "jpg_420", "jpg_422", "jpg_444", "jpg_gray", "jpg_prog420", "jpg_prog444", "jpg_gray_prog"];
        let t0 = std::time::Instant::now();
        let mut decoded = 0;
        for round in 0..3000 {
            let n = names[round % names.len()];
            let mut f = file(n);
            for _ in 0..1 + r.below(6) {
                let i = r.below(f.len());
                match r.below(4) {
                    0 => f[i] ^= 1 << r.below(8),
                    1 => f[i] = r.next() as u8,
                    2 => { let l = r.below(8); f.drain(i..(i + l).min(f.len())); if f.is_empty() { f.push(0); } }
                    _ => { let b = r.next() as u8; f.insert(i, b); }
                }
            }
            if decode_with(&f, &lim(), [255; 3]).is_ok() { decoded += 1; }
        }
        assert!(t0.elapsed().as_secs() < 60, "fuzzing took {:?}", t0.elapsed());
        assert!(decoded > 0);
    }

    #[test]
    fn random_garbage_with_valid_signatures_never_panics() {
        let mut r = Rng(12345);
        for round in 0..1500 {
            let mut f: Vec<u8> = match round % 4 { 0 => vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a], 1 => vec![0xff, 0xd8, 0xff], 2 => b"BM".to_vec(), _ => b"P6 ".to_vec() };
            for _ in 0..r.below(400) { f.push(r.next() as u8); }
            let _ = decode_with(&f, &lim(), [0; 3]);
        }
    }

    // ----------------------------------------------------------------------------- resampling

    #[test]
    fn resize_keeps_flat_images_flat_and_sizes_right() {
        let flat = Image { width: 9, height: 7, rgb: [10u8, 200, 90].repeat(63) };
        for f in [Filter::Nearest, Filter::Bilinear, Filter::Bicubic, Filter::Area, Filter::Lanczos3] {
            for (w, h) in [(9, 7), (3, 2), (20, 33), (1, 1), (100, 1)] {
                let r = resize(&flat, w, h, f).unwrap();
                assert_eq!((r.width, r.height, r.rgb.len()), (w, h, w * h * 3));
                assert!(r.rgb.chunks(3).all(|p| p == [10, 200, 90]), "{f:?} {w}x{h}");
            }
        }
    }

    #[test]
    fn area_halving_is_the_exact_mean() {
        let img = synthetic(8, 6, 1);
        let r = resize(&img, 4, 3, Filter::Area).unwrap();
        for y in 0..3 { for x in 0..4 { for c in 0..3 {
            let s: u32 = [(0, 0), (1, 0), (0, 1), (1, 1)].iter().map(|&(dx, dy)| img.rgb[((2 * y + dy) * 8 + 2 * x + dx) * 3 + c] as u32).sum();
            assert_eq!(r.rgb[(y * 4 + x) * 3 + c] as u32, (s as f32 / 4.0).round() as u32);
        } } }
    }

    #[test]
    fn linear_ramps_stay_linear_under_smooth_filters() {
        let mut rgb = vec![];
        for _ in 0..4 { for x in 0..32 { rgb.extend_from_slice(&[(x * 8) as u8; 3]); } }
        let ramp = Image { width: 32, height: 4, rgb };
        for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
            let r = resize(&ramp, 16, 4, f).unwrap();
            let row: Vec<i32> = (0..16).map(|x| r.rgb[x * 3] as i32).collect();
            for k in 1..15 { assert!((row[k] - row[k - 1] - 16).abs() <= 2, "{f:?} {row:?}"); }
        }
    }

    #[test]
    fn nearest_picks_pixel_centres() {
        let img = Image { width: 4, height: 1, rgb: vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] };
        let up = resize(&img, 8, 1, Filter::Nearest).unwrap();
        assert_eq!(up.rgb.chunks(3).map(|p| p[0]).collect::<Vec<_>>(), vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let down = resize(&img, 2, 1, Filter::Nearest).unwrap();
        assert_eq!(down.rgb.chunks(3).map(|p| p[0]).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn crop_pad_and_fit() {
        let img = synthetic(10, 6, 0);
        let c = crop(&img, 2, 1, 4, 3).unwrap();
        assert_eq!(&c.rgb[0..3], &img.rgb[(1 * 10 + 2) * 3..(1 * 10 + 2) * 3 + 3]);
        assert!(crop(&img, 8, 0, 4, 1).is_err());
        assert!(crop(&img, 0, 0, 0, 1).is_err());
        let p = pad_to_square(&img, [1, 2, 3]).unwrap();
        assert_eq!((p.width, p.height), (10, 10));
        assert_eq!(&p.rgb[0..3], &[1, 2, 3]);
        assert_eq!(&p.rgb[(2 * 10) * 3..(2 * 10) * 3 + 3], &img.rgb[0..3]);
        let cc = center_crop(&img, 6, 6, [0; 3]).unwrap();
        assert_eq!((cc.width, cc.height), (6, 6));
        assert_eq!(&cc.rgb[0..3], &img.rgb[2 * 3..2 * 3 + 3]);
        let big = center_crop(&img, 12, 12, [9, 9, 9]).unwrap();
        assert_eq!((big.width, big.height), (12, 12));
        let f = fit_inside(&img, 5, 5, Filter::Bilinear).unwrap();
        assert_eq!((f.width, f.height), (5, 3));
        let s = resize_short_side(&img, 12, Filter::Bicubic).unwrap();
        assert_eq!((s.width, s.height), (20, 12));
        assert!(pad_to(&img, 5, 5, [0; 3]).is_err());
    }

    #[test]
    fn chw_layout_and_normalisation() {
        let img = Image { width: 2, height: 1, rgb: vec![0, 128, 255, 255, 0, 128] };
        let t = to_chw(&img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        assert_eq!(t.len(), 6);
        assert!((t[0] + 1.0).abs() < 1e-6 && (t[1] - 1.0).abs() < 1e-6); // channel 0: pixels (0, 255)
        assert!((t[2] - (128.0 / 255.0 - 0.5) / 0.5).abs() < 1e-6 && (t[3] + 1.0).abs() < 1e-6); // channel 1: (128, 0)
        assert!((t[4] - 1.0).abs() < 1e-6 && (t[5] - (128.0 / 255.0 - 0.5) / 0.5).abs() < 1e-6);
        let bad = Image { width: 2, height: 2, rgb: vec![0; 3] };
        assert!(resize(&bad, 1, 1, Filter::Area).is_err());
        assert!(matches!(resize(&img, 100_000, 100_000, Filter::Area), Err(ImageError::TooLarge(_))));
    }
}
