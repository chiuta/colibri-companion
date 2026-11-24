//! Audio front-end and a Whisper-layout speech-to-text runner. Pure Rust, no extra dependencies,
//! written from the public format descriptions (RIFF/WAVE, the Whisper paper's log-mel recipe and
//! its published architecture); no code from any speech engine was read or copied.
//!
//! * [`read_wav`] / [`write_wav`] – PCM 8/16/24/32-bit, IEEE float 32/64, `WAVE_FORMAT_EXTENSIBLE`,
//!   with hostile-file limits.
//! * [`to_mono`], [`resample`] – channel mixdown and Kaiser-windowed-sinc resampling.
//! * [`Fft`] – mixed-radix complex FFT (any length; cost is `n · Σ prime factors`, so lengths with
//!   huge prime factors are slow but still correct).
//! * [`MelExtractor`] – the Whisper log-mel front-end (Slaney mel filters, periodic Hann window,
//!   reflect-padded centred STFT, last frame dropped, `max-8` clamp, `(x+4)/4`), configurable.
//! * [`detect_speech`] – an energy + zero-crossing voice-activity segmenter.
//! * [`Whisper`] – encoder (conv stem, sinusoidal positions, pre-norm blocks) and decoder (learned
//!   positions, causal self-attention with a KV cache, cross-attention), greedy / sampled / beam
//!   decoding with optional timestamp rules, long-audio windowing.
//! * [`AudioProjector`] / [`audio_embeddings`] – encoder frames → projector → rows of embeddings
//!   that an LLM front-end can splice into its input. They are returned as plain `f32` rows;
//!   feeding them into [`crate::model::Model`] needs an embedding-override hook that this module
//!   does not add.
//!
//! Weights are held as `f32`, so a large checkpoint needs about four bytes per parameter of RAM.
//! Tensor names follow the Hugging Face `transformers` Whisper layout as remembered from its
//! public documentation; nothing here has been run against real Whisper weights.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::loader::{LoadError, SafeTensors, TensorSource};
use crate::ops;
use crate::simd;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioError {
    /// The bytes are not a valid file of the claimed format.
    Format(String),
    /// A valid file the module does not handle (compressed WAV, say).
    Unsupported(String),
    /// Dimensions, lengths or parameters that do not fit together.
    Shape(String),
    /// A named tensor is absent.
    Missing(String),
    /// A size or count beyond the configured limits.
    Limit(String),
    /// Reading weights failed.
    Load(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::Format(s) => write!(f, "audio format: {s}"),
            AudioError::Unsupported(s) => write!(f, "audio unsupported: {s}"),
            AudioError::Shape(s) => write!(f, "audio shape: {s}"),
            AudioError::Missing(s) => write!(f, "audio: missing tensor {s}"),
            AudioError::Limit(s) => write!(f, "audio limit: {s}"),
            AudioError::Load(s) => write!(f, "audio load: {s}"),
        }
    }
}
impl std::error::Error for AudioError {}
impl From<LoadError> for AudioError {
    fn from(e: LoadError) -> Self {
        match e {
            LoadError::Missing(n) => AudioError::Missing(n),
            other => AudioError::Load(other.to_string()),
        }
    }
}
pub type Result<T> = std::result::Result<T, AudioError>;

fn fmt_err<T>(s: impl Into<String>) -> Result<T> {
    Err(AudioError::Format(s.into()))
}
fn shape_err<T>(s: impl Into<String>) -> Result<T> {
    Err(AudioError::Shape(s.into()))
}

// ---------------------------------------------------------------------------------------------
// WAV
// ---------------------------------------------------------------------------------------------

const WAVE_PCM: u16 = 1;
const WAVE_FLOAT: u16 = 3;
const WAVE_EXT: u16 = 0xFFFE;
/// Largest number of samples (all channels together) `read_wav` accepts: 2^28, about 4.6 hours
/// of 16 kHz mono.
pub const DEFAULT_MAX_SAMPLES: usize = 1 << 28;

/// Decoded audio: interleaved `f32` samples in `[-1, 1]` (floats outside that range are kept).
#[derive(Clone, Debug, PartialEq)]
pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl Wav {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }
    pub fn duration_s(&self) -> f64 {
        self.frames() as f64 / self.sample_rate.max(1) as f64
    }
    pub fn to_mono(&self) -> Vec<f32> {
        to_mono(&self.samples, self.channels)
    }
}

/// Stored sample type for [`write_wav`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    Pcm8,
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
    Float64,
}

impl SampleFormat {
    fn bits(self) -> u16 {
        match self {
            SampleFormat::Pcm8 => 8,
            SampleFormat::Pcm16 => 16,
            SampleFormat::Pcm24 => 24,
            SampleFormat::Pcm32 | SampleFormat::Float32 => 32,
            SampleFormat::Float64 => 64,
        }
    }
    fn is_float(self) -> bool {
        matches!(self, SampleFormat::Float32 | SampleFormat::Float64)
    }
}

struct Fmt {
    tag: u16,
    channels: u16,
    rate: u32,
    bits: u16,
}

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn parse_fmt(b: &[u8]) -> Result<Fmt> {
    let mut tag = le16(b, 0);
    let channels = le16(b, 2);
    let rate = le32(b, 4);
    let bits = le16(b, 14);
    if tag == WAVE_EXT {
        if b.len() < 40 {
            return fmt_err("truncated WAVE_FORMAT_EXTENSIBLE header");
        }
        if le16(b, 16) < 22 {
            return fmt_err("extensible header with a short extension");
        }
        tag = le16(b, 24);
    }
    Ok(Fmt {
        tag,
        channels,
        rate,
        bits,
    })
}

/// Parses a WAV file with the default sample limit.
pub fn read_wav(bytes: &[u8]) -> Result<Wav> {
    read_wav_with(bytes, DEFAULT_MAX_SAMPLES)
}

/// Parses a WAV file. A `data` chunk that claims more bytes than the file holds (streamed or cut
/// files) is read up to the end of the file; a trailing partial frame is dropped; non-finite float
/// samples become 0. Anything malformed is an error, never a panic.
pub fn read_wav_with(bytes: &[u8], max_samples: usize) -> Result<Wav> {
    if bytes.len() < 12 {
        return fmt_err("shorter than a RIFF header");
    }
    let rf64 = &bytes[0..4] == b"RF64";
    if !(rf64 || &bytes[0..4] == b"RIFF") || &bytes[8..12] != b"WAVE" {
        return fmt_err("not a RIFF/WAVE file");
    }
    let mut pos = 12usize;
    let mut fmt: Option<Fmt> = None;
    let mut data: Option<(usize, usize)> = None;
    while pos + 8 <= bytes.len() {
        let tag = [bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]];
        let size = le32(bytes, pos + 4) as usize;
        let body = pos + 8;
        let avail = bytes.len() - body;
        let adv;
        match tag {
            [b'f', b'm', b't', b' '] => {
                if size < 16 || size > avail {
                    return fmt_err("truncated fmt chunk");
                }
                if fmt.is_none() {
                    fmt = Some(parse_fmt(&bytes[body..body + size])?);
                }
                adv = size;
            }
            [b'd', b'a', b't', b'a'] => {
                let n = if size > avail { avail } else { size };
                if data.is_none() {
                    data = Some((body, n));
                }
                adv = n;
            }
            _ => {
                if size > avail {
                    break;
                }
                adv = size;
            }
        }
        pos = body + adv + (adv & 1);
    }
    let f = fmt.ok_or_else(|| AudioError::Format("no fmt chunk".into()))?;
    let (off, len) = data.ok_or_else(|| AudioError::Format("no data chunk".into()))?;
    if f.channels == 0 || f.channels > 64 {
        return fmt_err(format!("{} channels", f.channels));
    }
    if f.rate == 0 || f.rate > 768_000 {
        return fmt_err(format!("sample rate {}", f.rate));
    }
    let ok = match (f.tag, f.bits) {
        (WAVE_PCM, 8 | 16 | 24 | 32) | (WAVE_FLOAT, 32 | 64) => true,
        (WAVE_PCM | WAVE_FLOAT, b) => {
            return Err(AudioError::Unsupported(format!(
                "{b}-bit samples with tag {}",
                f.tag
            )));
        }
        (t, _) => {
            return Err(AudioError::Unsupported(format!(
                "format tag {t:#06x} (only PCM and IEEE float)"
            )));
        }
    };
    debug_assert!(ok);
    let width = (f.bits / 8) as usize;
    let frame_bytes = width * f.channels as usize;
    let n_frames = len / frame_bytes;
    let n = n_frames * f.channels as usize;
    if n > max_samples {
        return Err(AudioError::Limit(format!(
            "{n} samples exceed the limit of {max_samples}"
        )));
    }
    let d = &bytes[off..off + n * width];
    let mut samples = Vec::with_capacity(n);
    match (f.tag, f.bits) {
        (WAVE_PCM, 8) => samples.extend(d.iter().map(|&b| (b as f32 - 128.0) / 128.0)),
        (WAVE_PCM, 16) => samples.extend(
            d.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0),
        ),
        (WAVE_PCM, 24) => samples.extend(d.chunks_exact(3).map(|c| {
            let v = ((c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16)) << 8 >> 8;
            v as f32 / 8_388_608.0
        })),
        (WAVE_PCM, 32) => samples.extend(d.chunks_exact(4).map(|c| {
            (i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64 / 2_147_483_648.0) as f32
        })),
        (WAVE_FLOAT, 32) => samples.extend(
            d.chunks_exact(4)
                .map(|c| finite(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
        ),
        _ => samples.extend(d.chunks_exact(8).map(|c| {
            finite(f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
        })),
    }
    Ok(Wav {
        sample_rate: f.rate,
        channels: f.channels,
        samples,
    })
}

fn finite(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Serialises `wav`. Samples are clamped to `[-1, 1]` for the integer formats and rounded to the
/// nearest step (the round trip is within one step). Mono/stereo 8/16-bit PCM and floats use the
/// classic header; wider or multichannel audio uses `WAVE_FORMAT_EXTENSIBLE`.
pub fn write_wav(wav: &Wav, format: SampleFormat) -> Result<Vec<u8>> {
    if wav.channels == 0 || wav.channels > 64 {
        return shape_err(format!("{} channels", wav.channels));
    }
    if wav.sample_rate == 0 || wav.sample_rate > 768_000 {
        return shape_err(format!("sample rate {}", wav.sample_rate));
    }
    if !wav.samples.len().is_multiple_of(wav.channels as usize) {
        return shape_err("sample count is not a multiple of the channel count");
    }
    let bits = format.bits();
    let width = (bits / 8) as usize;
    let data_len = wav
        .samples
        .len()
        .checked_mul(width)
        .filter(|&n| n < u32::MAX as usize - 1024)
        .ok_or_else(|| AudioError::Limit("data chunk larger than 4 GiB".into()))?;
    let ext = wav.channels > 2 || bits > 16 && !format.is_float() || bits == 64;
    let tag = if format.is_float() {
        WAVE_FLOAT
    } else {
        WAVE_PCM
    };
    let fmt_len: u32 = if ext { 40 } else { 16 };
    let mut out = Vec::with_capacity(data_len + 64);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(
        &((4 + 8 + fmt_len as usize + 8 + data_len + (data_len & 1)) as u32).to_le_bytes(),
    );
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&fmt_len.to_le_bytes());
    out.extend_from_slice(&(if ext { WAVE_EXT } else { tag }).to_le_bytes());
    out.extend_from_slice(&wav.channels.to_le_bytes());
    out.extend_from_slice(&wav.sample_rate.to_le_bytes());
    let block = wav.channels as u32 * width as u32;
    out.extend_from_slice(&(wav.sample_rate.saturating_mul(block)).to_le_bytes());
    out.extend_from_slice(&(block as u16).to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    if ext {
        out.extend_from_slice(&22u16.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
        ]);
    }
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in &wav.samples {
        let s = finite(s);
        let c = s.clamp(-1.0, 1.0) as f64;
        match format {
            SampleFormat::Pcm8 => out.push(((c * 128.0).round() + 128.0).clamp(0.0, 255.0) as u8),
            SampleFormat::Pcm16 => out.extend_from_slice(
                &((c * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes(),
            ),
            SampleFormat::Pcm24 => out.extend_from_slice(
                &((c * 8_388_608.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32).to_le_bytes()
                    [..3],
            ),
            SampleFormat::Pcm32 => out.extend_from_slice(
                &((c * 2_147_483_648.0)
                    .round()
                    .clamp(-2_147_483_648.0, 2_147_483_647.0) as i32)
                    .to_le_bytes(),
            ),
            SampleFormat::Float32 => out.extend_from_slice(&s.to_le_bytes()),
            SampleFormat::Float64 => out.extend_from_slice(&(s as f64).to_le_bytes()),
        }
    }
    if data_len & 1 == 1 {
        out.push(0);
    }
    Ok(out)
}

/// Averages interleaved channels into one. `channels == 0` or a ragged tail yields what is whole.
pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let c = channels.max(1) as usize;
    if c == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(c)
        .map(|f| f.iter().sum::<f32>() / c as f32)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------------------------

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, q) = (1.0f64, 1.0f64, x * x / 4.0);
    for k in 1..60 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// Resamples with a Kaiser-windowed sinc (16 zero crossings each side, β = 8.6, about 90 dB of
/// stopband). Samples outside the signal count as zero. Output length is `round(n · to / from)`.
pub fn resample(x: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    resample_with(x, from, to, 16)
}

/// [`resample`] with an explicit half-width in zero crossings (1..=64).
pub fn resample_with(x: &[f32], from: u32, to: u32, zeros: usize) -> Result<Vec<f32>> {
    if from == 0 || to == 0 {
        return shape_err("sample rates must be positive");
    }
    if !(1..=64).contains(&zeros) {
        return shape_err("zero crossings must be in 1..=64");
    }
    if from == to {
        return Ok(x.to_vec());
    }
    let out_len_u = (x.len() as u128 * to as u128 + from as u128 / 2) / from as u128;
    if out_len_u > (1u128 << 31) {
        return Err(AudioError::Limit("resampled output too long".into()));
    }
    let out_len = out_len_u as usize;
    let fc = (to as f64 / from as f64).min(1.0);
    let hw = zeros as f64 / fc;
    const RES: f64 = 512.0;
    let table_len = (hw * RES).ceil() as usize + 2;
    let beta = 8.6f64;
    let i0b = bessel_i0(beta);
    let table: Vec<f32> = (0..table_len)
        .map(|k| {
            let u = k as f64 / RES;
            if u > hw {
                return 0.0;
            }
            let px = std::f64::consts::PI * fc * u;
            let sinc = if px.abs() < 1e-12 { 1.0 } else { px.sin() / px };
            let r = u / hw;
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            (fc * sinc * w) as f32
        })
        .collect();
    let step = from as f64 / to as f64;
    let mut out = Vec::with_capacity(out_len);
    for j in 0..out_len {
        let t = j as f64 * step;
        let lo = (t - hw).ceil().max(0.0) as usize;
        let hi = ((t + hw).floor() as isize).min(x.len() as isize - 1);
        let mut acc = 0.0f64;
        if hi >= lo as isize {
            for i in lo..=(hi as usize) {
                let pos = (i as f64 - t).abs() * RES;
                let k = pos as usize;
                let fr = (pos - k as f64) as f32;
                let w = table[k] * (1.0 - fr) + table[(k + 1).min(table_len - 1)] * fr;
                acc += (w * x[i]) as f64;
            }
        }
        out.push(acc as f32);
    }
    Ok(out)
}

/// Decodes WAV bytes straight to 16 kHz mono, the form the Whisper front-end expects.
pub fn load_audio_16k(bytes: &[u8]) -> Result<Vec<f32>> {
    let w = read_wav(bytes)?;
    resample(&w.to_mono(), w.sample_rate, 16_000)
}

// ---------------------------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}
impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }
    #[inline]
    fn add(self, o: Complex) -> Complex {
        Complex {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    #[inline]
    fn sub(self, o: Complex) -> Complex {
        Complex {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
    #[inline]
    fn mul(self, o: Complex) -> Complex {
        Complex {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    #[inline]
    fn conj(self) -> Complex {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// A forward complex FFT of fixed length, mixed radix (factors 4, 2, 3, 5, then any prime).
pub struct Fft {
    n: usize,
    tw: Vec<Complex>,
    factors: Vec<(usize, usize)>,
}

impl Fft {
    pub fn new(n: usize) -> Result<Fft> {
        if n == 0 || n > (1 << 24) {
            return shape_err(format!("FFT length {n} outside 1..=2^24"));
        }
        let tw = (0..n)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                Complex::new(a.cos(), a.sin())
            })
            .collect();
        let mut rest = n;
        let mut ps = vec![];
        for p in [4usize, 2, 3, 5] {
            while rest.is_multiple_of(p) {
                ps.push(p);
                rest /= p;
            }
        }
        let mut p = 7;
        while rest > 1 && p * p <= rest {
            while rest.is_multiple_of(p) {
                ps.push(p);
                rest /= p;
            }
            p += 2;
        }
        if rest > 1 {
            ps.push(rest);
        }
        let mut m = n;
        let factors = ps
            .into_iter()
            .map(|p| {
                m /= p;
                (p, m)
            })
            .collect();
        Ok(Fft { n, tw, factors })
    }

    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        false
    }

    /// `out[k] = Σ input[j] · e^{-2πi jk/n}`. Both slices must have length `n`.
    pub fn forward(&self, input: &[Complex], out: &mut [Complex]) -> Result<()> {
        if input.len() != self.n || out.len() != self.n {
            return shape_err("FFT buffers must have the planned length");
        }
        if self.n == 1 {
            out[0] = input[0];
            return Ok(());
        }
        self.rec(out, input, 1, 0);
        Ok(())
    }

    /// Inverse transform (scaled by `1/n`), via conjugation.
    pub fn inverse(&self, input: &[Complex], out: &mut [Complex]) -> Result<()> {
        let c: Vec<Complex> = input.iter().map(|v| v.conj()).collect();
        self.forward(&c, out)?;
        let k = 1.0 / self.n as f64;
        for v in out.iter_mut() {
            *v = Complex::new(v.re * k, -v.im * k);
        }
        Ok(())
    }

    fn rec(&self, out: &mut [Complex], inp: &[Complex], stride: usize, fidx: usize) {
        let (p, m) = self.factors[fidx];
        if m == 1 {
            for q in 0..p {
                out[q] = inp[q * stride];
            }
        } else {
            for q in 0..p {
                self.rec(
                    &mut out[q * m..(q + 1) * m],
                    &inp[q * stride..],
                    stride * p,
                    fidx + 1,
                );
            }
        }
        self.butterfly(out, stride, m, p);
    }

    fn butterfly(&self, out: &mut [Complex], fstride: usize, m: usize, p: usize) {
        let mut scratch = [Complex::default(); 8];
        let mut big: Vec<Complex> = if p > 8 {
            vec![Complex::default(); p]
        } else {
            vec![]
        };
        let s: &mut [Complex] = if p > 8 { &mut big } else { &mut scratch[..p] };
        for u in 0..m {
            for q in 0..p {
                s[q] = out[u + q * m];
            }
            for q1 in 0..p {
                let k = u + q1 * m;
                let mut acc = s[0];
                let mut twidx = 0usize;
                for q in 1..p {
                    twidx += fstride * k;
                    if twidx >= self.n {
                        twidx -= self.n;
                    }
                    acc = acc.add(s[q].mul(self.tw[twidx]));
                }
                out[k] = acc;
            }
        }
    }

    /// Transforms two real signals with one complex FFT; returns bins `0..=n/2` of each.
    pub fn forward_real_pair(
        &self,
        a: &[f32],
        b: &[f32],
        out_a: &mut [Complex],
        out_b: &mut [Complex],
    ) -> Result<()> {
        let n = self.n;
        if a.len() != n || b.len() != n || out_a.len() < n / 2 + 1 || out_b.len() < n / 2 + 1 {
            return shape_err("real-pair FFT buffers");
        }
        let z: Vec<Complex> = (0..n)
            .map(|i| Complex::new(a[i] as f64, b[i] as f64))
            .collect();
        let mut zz = vec![Complex::default(); n];
        self.forward(&z, &mut zz)?;
        for k in 0..=n / 2 {
            let zk = zz[k];
            let zn = zz[(n - k) % n].conj();
            let s = zk.add(zn);
            let d = zk.sub(zn);
            out_a[k] = Complex::new(s.re * 0.5, s.im * 0.5);
            // (zk - conj(z[n-k])) / (2i) = -i/2 · d
            out_b[k] = Complex::new(d.im * 0.5, -d.re * 0.5);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Log-mel front-end
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
    /// Window the model sees, padded or trimmed to this length by `log_mel(.., true)`.
    pub chunk_seconds: f32,
}

impl Default for MelConfig {
    /// Whisper: 16 kHz, 25 ms window, 10 ms hop, 80 mels up to 8 kHz, 30 s chunks.
    fn default() -> Self {
        MelConfig {
            sample_rate: 16_000,
            n_fft: 400,
            hop: 160,
            n_mels: 80,
            f_min: 0.0,
            f_max: 8000.0,
            chunk_seconds: 30.0,
        }
    }
}

impl MelConfig {
    /// Whisper large-v3 uses 128 mel bins.
    pub fn whisper_128() -> Self {
        MelConfig {
            n_mels: 128,
            ..Default::default()
        }
    }
    pub fn chunk_samples(&self) -> usize {
        (self.sample_rate as f64 * self.chunk_seconds as f64).round() as usize
    }
    fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 || self.sample_rate > 768_000 {
            return shape_err("sample rate");
        }
        if !(2..=65_536).contains(&self.n_fft) {
            return shape_err("n_fft must be in 2..=65536");
        }
        if self.hop == 0 || self.hop > self.n_fft * 8 {
            return shape_err("hop must be in 1..=8·n_fft");
        }
        if !(1..=512).contains(&self.n_mels) {
            return shape_err("n_mels must be in 1..=512");
        }
        if !(self.f_min.is_finite() && self.f_max.is_finite())
            || self.f_min < 0.0
            || self.f_min >= self.f_max
            || self.f_max > self.sample_rate as f32 / 2.0 + 0.5
        {
            return shape_err("need 0 <= f_min < f_max <= sample_rate/2");
        }
        if !(self.chunk_seconds.is_finite()
            && self.chunk_seconds > 0.0
            && self.chunk_seconds <= 3600.0)
        {
            return shape_err("chunk_seconds");
        }
        Ok(())
    }
}

/// A log-mel spectrogram, mel-major: `data[m * frames + t]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mel {
    pub n_mels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
    /// Frames that come from real audio rather than zero padding.
    pub valid_frames: usize,
}

impl Mel {
    pub fn at(&self, m: usize, t: usize) -> f32 {
        self.data[m * self.frames + t]
    }
    /// Time-major copy `[frames][n_mels]`, the layout the encoder's first convolution reads.
    pub fn time_major(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.data.len()];
        for m in 0..self.n_mels {
            for t in 0..self.frames {
                out[t * self.n_mels + m] = self.data[m * self.frames + t];
            }
        }
        out
    }
}

const MIN_LOG_HZ: f64 = 1000.0;
const F_SP: f64 = 200.0 / 3.0;

fn log_step() -> f64 {
    6.4f64.ln() / 27.0
}

/// Slaney mel scale: linear below 1 kHz (200/3 Hz per mel), logarithmic above.
pub fn hz_to_mel(f: f64) -> f64 {
    if f >= MIN_LOG_HZ {
        MIN_LOG_HZ / F_SP + (f / MIN_LOG_HZ).ln() / log_step()
    } else {
        f / F_SP
    }
}
pub fn mel_to_hz(m: f64) -> f64 {
    let min_mel = MIN_LOG_HZ / F_SP;
    if m >= min_mel {
        MIN_LOG_HZ * ((m - min_mel) * log_step()).exp()
    } else {
        m * F_SP
    }
}

struct Filter {
    start: usize,
    w: Vec<f32>,
}

pub struct MelExtractor {
    cfg: MelConfig,
    fft: Fft,
    window: Vec<f32>,
    filters: Vec<Filter>,
    centers: Vec<f64>,
}

impl MelExtractor {
    pub fn new(cfg: MelConfig) -> Result<MelExtractor> {
        cfg.validate()?;
        let fft = Fft::new(cfg.n_fft)?;
        let window = (0..cfg.n_fft)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / cfg.n_fft as f64).cos())
                    as f32
            })
            .collect();
        let bins = cfg.n_fft / 2 + 1;
        let (m0, m1) = (hz_to_mel(cfg.f_min as f64), hz_to_mel(cfg.f_max as f64));
        let pts: Vec<f64> = (0..cfg.n_mels + 2)
            .map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (cfg.n_mels + 1) as f64))
            .collect();
        let freq = |k: usize| k as f64 * cfg.sample_rate as f64 / cfg.n_fft as f64;
        let mut filters = Vec::with_capacity(cfg.n_mels);
        for m in 0..cfg.n_mels {
            let (lo, mid, hi) = (pts[m], pts[m + 1], pts[m + 2]);
            let enorm = 2.0 / (hi - lo).max(1e-9);
            let mut start = usize::MAX;
            let mut w = vec![];
            for k in 0..bins {
                let f = freq(k);
                let up = (f - lo) / (mid - lo).max(1e-9);
                let down = (hi - f) / (hi - mid).max(1e-9);
                let v = up.min(down).max(0.0) * enorm;
                if v > 0.0 {
                    if start == usize::MAX {
                        start = k;
                    }
                    while start + w.len() < k {
                        w.push(0.0);
                    }
                    w.push(v as f32);
                }
            }
            if start == usize::MAX {
                start = 0;
            }
            filters.push(Filter { start, w });
        }
        let centers = (0..cfg.n_mels).map(|m| pts[m + 1]).collect();
        Ok(MelExtractor {
            cfg,
            fft,
            window,
            filters,
            centers,
        })
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }
    /// Centre frequency in Hz of each mel filter.
    pub fn center_hz(&self) -> &[f64] {
        &self.centers
    }

    /// Whisper's log-mel spectrogram. With `pad_to_chunk` the audio is zero-padded (or trimmed) to
    /// `chunk_seconds` first, which gives exactly `chunk_seconds · sample_rate / hop` frames
    /// (3000 for the defaults). Non-finite samples count as zero.
    pub fn log_mel(&self, samples: &[f32], pad_to_chunk: bool) -> Result<Mel> {
        let c = &self.cfg;
        let real_len = samples.len();
        let x: Vec<f32> = if pad_to_chunk {
            let n = c.chunk_samples();
            let mut v: Vec<f32> = samples.iter().take(n).map(|&s| finite(s)).collect();
            v.resize(n, 0.0);
            v
        } else {
            samples.iter().map(|&s| finite(s)).collect()
        };
        let len = x.len();
        let frames = len / c.hop;
        let valid_frames = (real_len.min(len) / c.hop).min(frames);
        if len == 0 || frames == 0 {
            return Ok(Mel {
                n_mels: c.n_mels,
                frames: 0,
                data: vec![],
                valid_frames: 0,
            });
        }
        let h = c.n_fft / 2;
        let period = if len > 1 { 2 * (len - 1) } else { 1 };
        let refl = |i: isize| -> usize {
            if len == 1 {
                return 0;
            }
            let r = i.rem_euclid(period as isize) as usize;
            if r >= len { period - r } else { r }
        };
        let padded: Vec<f32> = (0..len + 2 * h)
            .map(|i| x[refl(i as isize - h as isize)])
            .collect();
        let bins = c.n_fft / 2 + 1;
        let mut mel = vec![0f32; c.n_mels * frames];
        let (mut fa, mut fb) = (vec![0f32; c.n_fft], vec![0f32; c.n_fft]);
        let (mut xa, mut xb) = (
            vec![Complex::default(); bins],
            vec![Complex::default(); bins],
        );
        let (mut pa, mut pb) = (vec![0f32; bins], vec![0f32; bins]);
        let mut t = 0;
        while t < frames {
            let second = t + 1 < frames;
            for i in 0..c.n_fft {
                fa[i] = padded[t * c.hop + i] * self.window[i];
                fb[i] = if second {
                    padded[(t + 1) * c.hop + i] * self.window[i]
                } else {
                    0.0
                };
            }
            self.fft.forward_real_pair(&fa, &fb, &mut xa, &mut xb)?;
            for k in 0..bins {
                pa[k] = xa[k].norm_sqr() as f32;
                pb[k] = xb[k].norm_sqr() as f32;
            }
            for (m, f) in self.filters.iter().enumerate() {
                let r = f.start..f.start + f.w.len();
                mel[m * frames + t] = simd::dot_f32(&f.w, &pa[r.clone()]);
                if second {
                    mel[m * frames + t + 1] = simd::dot_f32(&f.w, &pb[r]);
                }
            }
            t += 2;
        }
        let mut mx = f32::NEG_INFINITY;
        for v in mel.iter_mut() {
            *v = v.max(1e-10).log10();
            mx = mx.max(*v);
        }
        for v in mel.iter_mut() {
            *v = (v.max(mx - 8.0) + 4.0) / 4.0;
        }
        Ok(Mel {
            n_mels: c.n_mels,
            frames,
            data: mel,
            valid_frames,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Voice activity detection
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct VadConfig {
    pub sample_rate: u32,
    pub frame_ms: f32,
    pub hop_ms: f32,
    /// How far above the noise floor a frame must be to count as speech (shrunk to half the
    /// dynamic range when the recording is nearly flat).
    pub margin_db: f32,
    /// Below this level (dBFS, RMS) a recording with no dynamic range is silence.
    pub abs_threshold_db: f32,
    pub min_speech_ms: f32,
    /// Gaps shorter than this between speech runs are bridged.
    pub min_silence_ms: f32,
    pub pad_ms: f32,
    /// Let noisy low-energy frames with many zero crossings (fricatives) count as speech.
    pub zcr_rescue: bool,
    /// Split segments longer than this at their quietest frame.
    pub max_segment_s: Option<f32>,
}

impl Default for VadConfig {
    fn default() -> Self {
        VadConfig {
            sample_rate: 16_000,
            frame_ms: 30.0,
            hop_ms: 10.0,
            margin_db: 12.0,
            abs_threshold_db: -50.0,
            min_speech_ms: 250.0,
            min_silence_ms: 300.0,
            pad_ms: 100.0,
            zcr_rescue: true,
            max_segment_s: None,
        }
    }
}

/// A span of samples `start..end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    sorted[(((sorted.len() - 1) as f32) * p).round() as usize]
}

/// Finds speech spans. The noise floor is the 10th percentile of frame energies, so the
/// recording needs a little non-speech to anchor on; a flat recording is judged by
/// `abs_threshold_db` alone. Heuristic: expect to tune `margin_db` for unusual material.
pub fn detect_speech(x: &[f32], cfg: &VadConfig) -> Result<Vec<Span>> {
    if cfg.sample_rate == 0
        || !(cfg.frame_ms > 0.0
            && cfg.hop_ms > 0.0
            && cfg.frame_ms.is_finite()
            && cfg.hop_ms.is_finite())
    {
        return shape_err("VAD frame sizes");
    }
    for v in [
        cfg.margin_db,
        cfg.abs_threshold_db,
        cfg.min_speech_ms,
        cfg.min_silence_ms,
        cfg.pad_ms,
    ] {
        if !v.is_finite() {
            return shape_err("VAD parameters must be finite");
        }
    }
    if x.is_empty() {
        return Ok(vec![]);
    }
    let ms = |v: f32| {
        (cfg.sample_rate as f64 * v as f64 / 1000.0)
            .round()
            .max(0.0) as usize
    };
    let (fl, hop) = (ms(cfg.frame_ms).max(2), ms(cfg.hop_ms).max(1));
    let nf = if x.len() <= fl {
        1
    } else {
        (x.len() - fl) / hop + 1
    };
    let mut e = Vec::with_capacity(nf);
    let mut zc = Vec::with_capacity(nf);
    for i in 0..nf {
        let fr = &x[i * hop..(i * hop + fl).min(x.len())];
        let fr: Vec<f32> = fr.iter().map(|&s| finite(s)).collect();
        let ms_ = fr.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>() / fr.len() as f64;
        e.push((10.0 * (ms_ + 1e-12).log10()) as f32);
        let crossings = fr
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        zc.push(crossings as f32 / (fr.len().max(2) - 1) as f32);
    }
    let mut sorted = e.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let (noise, peak) = (percentile(&sorted, 0.10), percentile(&sorted, 0.95));
    let range = peak - noise;
    let mut speech: Vec<bool> = if range < 6.0 {
        vec![peak > cfg.abs_threshold_db; nf]
    } else {
        let thr = noise + cfg.margin_db.min(range * 0.5).max(6.0);
        (0..nf)
            .map(|i| e[i] > thr || (cfg.zcr_rescue && e[i] > thr - 6.0 && zc[i] > 0.25))
            .collect()
    };
    let (gap, minrun) = (
        ms(cfg.min_silence_ms).div_ceil(hop).max(1),
        ms(cfg.min_speech_ms).div_ceil(hop).max(1),
    );
    // Bridge short gaps between speech runs.
    let mut i = 0;
    while i < nf {
        if !speech[i] {
            let j = (i..nf).find(|&j| speech[j]).unwrap_or(nf);
            if i > 0 && j < nf && j - i < gap {
                for s in &mut speech[i..j] {
                    *s = true;
                }
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    // Drop short runs.
    let mut runs: Vec<(usize, usize)> = vec![];
    let mut i = 0;
    while i < nf {
        if speech[i] {
            let j = (i..nf).find(|&j| !speech[j]).unwrap_or(nf);
            if j - i >= minrun {
                runs.push((i, j));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    let pad = ms(cfg.pad_ms);
    let mut spans: Vec<Span> = vec![];
    for (a, b) in runs {
        let s = (a * hop).saturating_sub(pad);
        let en = ((b - 1) * hop + fl + pad).min(x.len());
        match spans.last_mut() {
            Some(l) if s <= l.end => l.end = l.end.max(en),
            _ => spans.push(Span {
                start: s,
                end: en.max(s),
            }),
        }
    }
    if let Some(maxs) = cfg.max_segment_s {
        let maxn = ms(maxs * 1000.0).max(hop * 4);
        let mut out = vec![];
        for sp in spans {
            let mut cur = sp.start;
            while sp.end - cur > maxn {
                // quietest frame in the last quarter of the allowed length
                let (lo, hi) = (cur + maxn * 3 / 4, cur + maxn);
                let (f0, f1) = ((lo / hop).min(nf - 1), (hi / hop).min(nf - 1));
                let cut = (f0..=f1)
                    .min_by(|&a, &b| e[a].total_cmp(&e[b]))
                    .map_or(hi, |f| (f * hop + fl / 2).clamp(cur + 1, hi));
                out.push(Span {
                    start: cur,
                    end: cut,
                });
                cur = cut;
            }
            out.push(Span {
                start: cur,
                end: sp.end,
            });
        }
        spans = out;
    }
    Ok(spans)
}

/// Groups consecutive spans into windows of at most `max_len` samples (a span longer than that is
/// cut at `max_len`), so each window fits one model chunk.
pub fn plan_windows(spans: &[Span], max_len: usize) -> Vec<Span> {
    let max_len = max_len.max(1);
    let mut out: Vec<Span> = vec![];
    for sp in spans {
        let mut s = sp.start;
        while sp.end.saturating_sub(s) > max_len {
            out.push(Span {
                start: s,
                end: s + max_len,
            });
            s += max_len;
        }
        if sp.end <= s {
            continue;
        }
        match out.last_mut() {
            Some(l) if sp.end - l.start <= max_len && s >= l.end => l.end = sp.end,
            _ => out.push(Span {
                start: s,
                end: sp.end,
            }),
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------------------------

fn n_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8)
}

/// Runs `f(first_row, rows)` over disjoint row blocks of `out`, on several threads when `work`
/// (multiply-adds) is large. Rows are independent, so the result does not depend on the split.
fn par_rows<F: Fn(usize, &mut [f32]) + Sync>(out: &mut [f32], row_len: usize, work: usize, f: F) {
    let rows = out.len().checked_div(row_len).unwrap_or(0);
    let t = if work < 2_000_000 {
        1
    } else {
        n_threads().min(rows.max(1))
    };
    if t <= 1 || rows == 0 {
        f(0, out);
        return;
    }
    let per = rows.div_ceil(t);
    std::thread::scope(|s| {
        for (i, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || f(i * per, chunk));
        }
    });
}

fn erf(x: f64) -> f64 {
    let a = x.abs();
    let r = if a < 2.5 {
        let x2 = a * a;
        let (mut term, mut sum, mut n) = (a, a, 0.0f64);
        loop {
            n += 1.0;
            term *= -x2 / n;
            let t = term / (2.0 * n + 1.0);
            sum += t;
            if t.abs() < 1e-17 * sum.abs() || n > 200.0 {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else if a > 6.0 {
        1.0
    } else {
        let mut t = a;
        for k in (1..=60).rev() {
            t = a + (k as f64 / 2.0) / t;
        }
        1.0 - (-a * a).exp() / (std::f64::consts::PI.sqrt() * t)
    };
    if x < 0.0 { -r } else { r }
}

/// Exact (erf-based) GELU, as Whisper uses.
pub fn gelu(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))) as f32
}

/// Whisper's fixed sinusoidal position table, `len × d` (d even): `[sin | cos]` halves.
pub fn sinusoids(len: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let inc = if half > 1 {
        10000f64.ln() / (half - 1) as f64
    } else {
        0.0
    };
    let mut out = vec![0.0f32; len * d];
    for t in 0..len {
        for i in 0..half {
            let a = t as f64 * (-(inc * i as f64)).exp();
            out[t * d + i] = a.sin() as f32;
            out[t * d + half + i] = a.cos() as f32;
        }
    }
    out
}

fn layer_norm(x: &[f32], d: usize, g: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = xr
            .iter()
            .map(|&v| {
                let c = v as f64 - mean;
                c * c
            })
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..d {
            or[i] = ((xr[i] as f64 - mean) * inv) as f32 * g[i] + b[i];
        }
    }
    out
}

fn add_in_place(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += *b;
    }
}

/// Multi-head attention over rows of `d` floats. `causal_base = Some(b)` lets query `i` see keys
/// `0..=b+i`; `None` lets every query see every key.
fn attention(
    q: &[f32],
    nq: usize,
    k: &[f32],
    v: &[f32],
    nk: usize,
    d: usize,
    heads: usize,
    causal_base: Option<usize>,
) -> Vec<f32> {
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; nq * d];
    par_rows(&mut out, d, nq * nk * d, |r0, ch| {
        let mut sc = vec![0f32; nk.max(1)];
        for (i, row) in ch.chunks_mut(d).enumerate() {
            let qi = r0 + i;
            let lim = match causal_base {
                Some(b) => (b + qi + 1).min(nk),
                None => nk,
            };
            if lim == 0 {
                continue;
            }
            for h in 0..heads {
                let qh = &q[qi * d + h * hd..qi * d + (h + 1) * hd];
                for j in 0..lim {
                    sc[j] = simd::dot_f32(qh, &k[j * d + h * hd..j * d + (h + 1) * hd]) * scale;
                }
                ops::softmax(&mut sc[..lim]);
                let o = &mut row[h * hd..(h + 1) * hd];
                for j in 0..lim {
                    simd::axpy(sc[j], &v[j * d + h * hd..j * d + (h + 1) * hd], o);
                }
            }
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------
// Whisper model
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperConfig {
    pub n_mels: usize,
    pub d_model: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub enc_heads: usize,
    pub dec_heads: usize,
    pub enc_ffn: usize,
    pub dec_ffn: usize,
    pub vocab: usize,
    /// Encoder positions (1500 for the 30 s models).
    pub max_source: usize,
    /// Decoder positions (448 for the released models).
    pub max_target: usize,
}

impl WhisperConfig {
    /// Reads a Hugging Face `config.json` (`num_mel_bins`, `d_model`, `encoder_layers`, …).
    pub fn from_json(text: &str) -> Result<WhisperConfig> {
        let v: Value = serde_json::from_str(text)
            .map_err(|e| AudioError::Format(format!("config.json: {e}")))?;
        let get = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| AudioError::Format(format!("config.json: missing or invalid `{k}`")))
        };
        let d = get("d_model")?;
        let cfg = WhisperConfig {
            n_mels: v
                .get("num_mel_bins")
                .and_then(Value::as_u64)
                .map_or(Ok(80), |n| {
                    usize::try_from(n).map_err(|_| AudioError::Format("num_mel_bins".into()))
                })?,
            d_model: d,
            enc_layers: get("encoder_layers")?,
            dec_layers: get("decoder_layers")?,
            enc_heads: get("encoder_attention_heads")?,
            dec_heads: get("decoder_attention_heads")?,
            enc_ffn: v
                .get("encoder_ffn_dim")
                .and_then(Value::as_u64)
                .map_or(4 * d, |n| n as usize),
            dec_ffn: v
                .get("decoder_ffn_dim")
                .and_then(Value::as_u64)
                .map_or(4 * d, |n| n as usize),
            vocab: get("vocab_size")?,
            max_source: v
                .get("max_source_positions")
                .and_then(Value::as_u64)
                .map_or(1500, |n| n as usize),
            max_target: v
                .get("max_target_positions")
                .and_then(Value::as_u64)
                .map_or(448, |n| n as usize),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let c = self;
        if !(1..=8192).contains(&c.d_model) || !c.d_model.is_multiple_of(2) {
            return shape_err("d_model must be even and in 2..=8192");
        }
        if !(1..=512).contains(&c.n_mels) {
            return shape_err("n_mels must be in 1..=512");
        }
        if c.enc_layers == 0 || c.enc_layers > 256 || c.dec_layers == 0 || c.dec_layers > 256 {
            return shape_err("layer counts must be in 1..=256");
        }
        for h in [c.enc_heads, c.dec_heads] {
            if h == 0 || h > c.d_model || !c.d_model.is_multiple_of(h) {
                return shape_err("heads must divide d_model");
            }
        }
        if c.enc_ffn == 0 || c.enc_ffn > 65_536 || c.dec_ffn == 0 || c.dec_ffn > 65_536 {
            return shape_err("ffn width");
        }
        if !(1..=2_000_000).contains(&c.vocab) {
            return shape_err("vocab size");
        }
        if !(1..=100_000).contains(&c.max_source) || !(2..=100_000).contains(&c.max_target) {
            return shape_err("position counts");
        }
        Ok(())
    }

    /// Encoder rows produced for `frames` mel frames.
    pub fn encoder_rows(frames: usize) -> usize {
        frames.div_ceil(2)
    }
}

struct Linear {
    w: Vec<f32>,
    out: usize,
    inn: usize,
    b: Option<Vec<f32>>,
}

impl Linear {
    fn apply(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * self.out];
        par_rows(&mut out, self.out, n * self.out * self.inn, |r0, ch| {
            for (i, row) in ch.chunks_mut(self.out).enumerate() {
                let xr = &x[(r0 + i) * self.inn..(r0 + i + 1) * self.inn];
                for (o, y) in row.iter_mut().enumerate() {
                    *y = simd::dot_f32(xr, &self.w[o * self.inn..(o + 1) * self.inn])
                        + self.b.as_ref().map_or(0.0, |b| b[o]);
                }
            }
        });
        out
    }
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}
struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}
struct EncLayer {
    ln1: Norm,
    attn: Attn,
    ln2: Norm,
    fc1: Linear,
    fc2: Linear,
}
struct DecLayer {
    ln1: Norm,
    attn: Attn,
    ln_x: Norm,
    cross: Attn,
    ln2: Norm,
    fc1: Linear,
    fc2: Linear,
}
/// Kernel size 3, padding 1; weights stored `[out][tap][in]`.
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    out: usize,
    inn: usize,
}

impl Conv {
    fn apply(&self, x: &[f32], t_in: usize, stride: usize) -> (Vec<f32>, usize) {
        let t_out = (t_in + 2 - 3) / stride + 1;
        let mut out = vec![0.0f32; t_out * self.out];
        par_rows(
            &mut out,
            self.out,
            t_out * self.out * self.inn * 3,
            |r0, ch| {
                for (i, row) in ch.chunks_mut(self.out).enumerate() {
                    let t = r0 + i;
                    for (o, y) in row.iter_mut().enumerate() {
                        let mut acc = self.b[o];
                        for tap in 0..3 {
                            let ti = (t * stride + tap) as isize - 1;
                            if ti < 0 || ti as usize >= t_in {
                                continue;
                            }
                            let ti = ti as usize;
                            acc += simd::dot_f32(
                                &self.w[(o * 3 + tap) * self.inn..(o * 3 + tap + 1) * self.inn],
                                &x[ti * self.inn..(ti + 1) * self.inn],
                            );
                        }
                        *y = acc;
                    }
                }
            },
        );
        (out, t_out)
    }
}

pub struct Whisper {
    cfg: WhisperConfig,
    conv1: Conv,
    conv2: Conv,
    enc_pos: Vec<f32>,
    enc_layers: Vec<EncLayer>,
    enc_ln: Norm,
    tok_emb: Vec<f32>,
    dec_pos: Vec<f32>,
    dec_layers: Vec<DecLayer>,
    dec_ln: Norm,
    proj_out: Option<Vec<f32>>,
}

struct Loader<'a> {
    src: &'a dyn TensorSource,
    prefix: &'static str,
}

impl Loader<'_> {
    fn raw(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let v = self.src.read(name)?;
        if v.len() != len {
            return shape_err(format!("{name}: {} values, expected {len}", v.len()));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(AudioError::Format(format!(
                "{name}: contains NaN or infinity"
            )));
        }
        Ok(v)
    }
    fn name(&self, s: &str) -> String {
        format!("{}{}", self.prefix, s)
    }
    fn linear(&self, base: &str, out: usize, inn: usize, bias: bool) -> Result<Linear> {
        let w = self.raw(&self.name(&format!("{base}.weight")), out * inn)?;
        let bn = self.name(&format!("{base}.bias"));
        let b = if bias || self.src.has(&bn) {
            if self.src.has(&bn) {
                Some(self.raw(&bn, out)?)
            } else if bias {
                return Err(AudioError::Missing(bn));
            } else {
                None
            }
        } else {
            None
        };
        Ok(Linear { w, out, inn, b })
    }
    fn norm(&self, base: &str, d: usize) -> Result<Norm> {
        Ok(Norm {
            g: self.raw(&self.name(&format!("{base}.weight")), d)?,
            b: self.raw(&self.name(&format!("{base}.bias")), d)?,
        })
    }
    fn attn(&self, base: &str, d: usize) -> Result<Attn> {
        Ok(Attn {
            q: self.linear(&format!("{base}.q_proj"), d, d, true)?,
            k: self.linear(&format!("{base}.k_proj"), d, d, false)?,
            v: self.linear(&format!("{base}.v_proj"), d, d, true)?,
            o: self.linear(&format!("{base}.out_proj"), d, d, true)?,
        })
    }
    fn conv(&self, base: &str, out: usize, inn: usize) -> Result<Conv> {
        let hf = self.raw(&self.name(&format!("{base}.weight")), out * inn * 3)?;
        let mut w = vec![0.0; hf.len()];
        for o in 0..out {
            for i in 0..inn {
                for k in 0..3 {
                    w[(o * 3 + k) * inn + i] = hf[(o * inn + i) * 3 + k];
                }
            }
        }
        Ok(Conv {
            w,
            b: self.raw(&self.name(&format!("{base}.bias")), out)?,
            out,
            inn,
        })
    }
}

impl Whisper {
    /// Builds the model from any tensor source in the Hugging Face layout (names with or without
    /// the leading `model.`).
    pub fn load(cfg: WhisperConfig, src: &dyn TensorSource) -> Result<Whisper> {
        cfg.validate()?;
        let prefix: &'static str = if src.has("model.decoder.embed_tokens.weight") {
            "model."
        } else {
            ""
        };
        let l = Loader { src, prefix };
        let d = cfg.d_model;
        let conv1 = l.conv("encoder.conv1", d, cfg.n_mels)?;
        let conv2 = l.conv("encoder.conv2", d, d)?;
        let pos_name = l.name("encoder.embed_positions.weight");
        let enc_pos = if src.has(&pos_name) {
            l.raw(&pos_name, cfg.max_source * d)?
        } else {
            sinusoids(cfg.max_source, d)
        };
        let mut enc_layers = vec![];
        for i in 0..cfg.enc_layers {
            let b = format!("encoder.layers.{i}");
            enc_layers.push(EncLayer {
                ln1: l.norm(&format!("{b}.self_attn_layer_norm"), d)?,
                attn: l.attn(&format!("{b}.self_attn"), d)?,
                ln2: l.norm(&format!("{b}.final_layer_norm"), d)?,
                fc1: l.linear(&format!("{b}.fc1"), cfg.enc_ffn, d, true)?,
                fc2: l.linear(&format!("{b}.fc2"), d, cfg.enc_ffn, true)?,
            });
        }
        let enc_ln = l.norm("encoder.layer_norm", d)?;
        let tok_emb = l.raw(&l.name("decoder.embed_tokens.weight"), cfg.vocab * d)?;
        let dec_pos = l.raw(
            &l.name("decoder.embed_positions.weight"),
            cfg.max_target * d,
        )?;
        let mut dec_layers = vec![];
        for i in 0..cfg.dec_layers {
            let b = format!("decoder.layers.{i}");
            dec_layers.push(DecLayer {
                ln1: l.norm(&format!("{b}.self_attn_layer_norm"), d)?,
                attn: l.attn(&format!("{b}.self_attn"), d)?,
                ln_x: l.norm(&format!("{b}.encoder_attn_layer_norm"), d)?,
                cross: l.attn(&format!("{b}.encoder_attn"), d)?,
                ln2: l.norm(&format!("{b}.final_layer_norm"), d)?,
                fc1: l.linear(&format!("{b}.fc1"), cfg.dec_ffn, d, true)?,
                fc2: l.linear(&format!("{b}.fc2"), d, cfg.dec_ffn, true)?,
            });
        }
        let dec_ln = l.norm("decoder.layer_norm", d)?;
        let proj_out = if src.has("proj_out.weight") {
            Some(l.raw("proj_out.weight", cfg.vocab * d)?)
        } else {
            None
        };
        Ok(Whisper {
            cfg,
            conv1,
            conv2,
            enc_pos,
            enc_layers,
            enc_ln,
            tok_emb,
            dec_pos,
            dec_layers,
            dec_ln,
            proj_out,
        })
    }

    /// Opens a directory with `config.json` and safetensors weights (single file or sharded).
    pub fn open_dir(dir: &Path) -> Result<Whisper> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| AudioError::Load(format!("config.json: {e}")))?;
        let cfg = WhisperConfig::from_json(&text)?;
        let st = SafeTensors::open(dir)?;
        Whisper::load(cfg, &st)
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.cfg
    }

    /// Runs the encoder over a log-mel spectrogram; returns `rows × d_model` values, row-major.
    /// The official models were trained on exactly 30 s (3000 frames); shorter input works but
    /// the model has not seen it.
    pub fn encode(&self, mel: &Mel) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if mel.n_mels != c.n_mels {
            return shape_err(format!(
                "mel has {} bins, the model wants {}",
                mel.n_mels, c.n_mels
            ));
        }
        if mel.frames == 0 {
            return shape_err("empty spectrogram");
        }
        if mel.data.len() != mel.n_mels * mel.frames {
            return shape_err("mel buffer length");
        }
        let rows = WhisperConfig::encoder_rows(mel.frames);
        if rows > c.max_source {
            return Err(AudioError::Limit(format!(
                "{} frames give {rows} encoder rows, the model has {} positions",
                mel.frames, c.max_source
            )));
        }
        let d = c.d_model;
        let x0 = mel.time_major();
        let (mut h, t1) = self.conv1.apply(&x0, mel.frames, 1);
        for v in h.iter_mut() {
            *v = gelu(*v);
        }
        let (mut x, t2) = self.conv2.apply(&h, t1, 2);
        for v in x.iter_mut() {
            *v = gelu(*v);
        }
        debug_assert_eq!(t2, rows);
        add_in_place(&mut x, &self.enc_pos[..t2 * d]);
        for l in &self.enc_layers {
            let h = layer_norm(&x, d, &l.ln1.g, &l.ln1.b);
            let (q, k, v) = (
                l.attn.q.apply(&h, t2),
                l.attn.k.apply(&h, t2),
                l.attn.v.apply(&h, t2),
            );
            let a = attention(&q, t2, &k, &v, t2, d, c.enc_heads, None);
            add_in_place(&mut x, &l.attn.o.apply(&a, t2));
            let h = layer_norm(&x, d, &l.ln2.g, &l.ln2.b);
            let mut m = l.fc1.apply(&h, t2);
            for v in m.iter_mut() {
                *v = gelu(*v);
            }
            add_in_place(&mut x, &l.fc2.apply(&m, t2));
        }
        Ok(layer_norm(&x, d, &self.enc_ln.g, &self.enc_ln.b))
    }

    /// Starts a decoding session over encoder output `enc` (`rows × d_model`).
    pub fn start(&self, enc: &[f32]) -> Result<DecoderCache> {
        let d = self.cfg.d_model;
        if enc.is_empty() || !enc.len().is_multiple_of(d) {
            return shape_err("encoder output must be a whole number of rows");
        }
        let n = enc.len() / d;
        let cross: Vec<(Vec<f32>, Vec<f32>)> = self
            .dec_layers
            .iter()
            .map(|l| (l.cross.k.apply(enc, n), l.cross.v.apply(enc, n)))
            .collect();
        let nl = self.dec_layers.len();
        Ok(DecoderCache {
            self_k: vec![vec![]; nl],
            self_v: vec![vec![]; nl],
            cross: Arc::new(cross),
            n_src: n,
            pos: 0,
        })
    }

    /// Feeds `tokens` (a prompt chunk or one new token) and returns the logits that follow the
    /// last one. Positions continue from the cache; token ids and the position limit are checked
    /// before anything is changed.
    pub fn step(&self, cache: &mut DecoderCache, tokens: &[usize]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let d = c.d_model;
        let n = tokens.len();
        if n == 0 {
            return shape_err("no tokens");
        }
        if cache.pos + n > c.max_target {
            return Err(AudioError::Limit(format!(
                "{} tokens exceed the {} decoder positions",
                cache.pos + n,
                c.max_target
            )));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t >= c.vocab) {
            return shape_err(format!("token {t} outside the vocabulary of {}", c.vocab));
        }
        if cache.cross.len() != self.dec_layers.len() {
            return shape_err("cache belongs to a different model");
        }
        let base = cache.pos;
        let mut x = vec![0.0f32; n * d];
        for (i, &t) in tokens.iter().enumerate() {
            for j in 0..d {
                x[i * d + j] = self.tok_emb[t * d + j] + self.dec_pos[(base + i) * d + j];
            }
        }
        for (li, l) in self.dec_layers.iter().enumerate() {
            let h = layer_norm(&x, d, &l.ln1.g, &l.ln1.b);
            let (q, k, v) = (
                l.attn.q.apply(&h, n),
                l.attn.k.apply(&h, n),
                l.attn.v.apply(&h, n),
            );
            cache.self_k[li].extend_from_slice(&k);
            cache.self_v[li].extend_from_slice(&v);
            let a = attention(
                &q,
                n,
                &cache.self_k[li],
                &cache.self_v[li],
                base + n,
                d,
                c.dec_heads,
                Some(base),
            );
            add_in_place(&mut x, &l.attn.o.apply(&a, n));
            let h = layer_norm(&x, d, &l.ln_x.g, &l.ln_x.b);
            let q = l.cross.q.apply(&h, n);
            let (ck, cv) = &cache.cross[li];
            let a = attention(&q, n, ck, cv, cache.n_src, d, c.dec_heads, None);
            add_in_place(&mut x, &l.cross.o.apply(&a, n));
            let h = layer_norm(&x, d, &l.ln2.g, &l.ln2.b);
            let mut m = l.fc1.apply(&h, n);
            for v in m.iter_mut() {
                *v = gelu(*v);
            }
            add_in_place(&mut x, &l.fc2.apply(&m, n));
        }
        cache.pos += n;
        let last = layer_norm(&x[(n - 1) * d..], d, &self.dec_ln.g, &self.dec_ln.b);
        let emb = self.proj_out.as_deref().unwrap_or(&self.tok_emb);
        let mut logits = vec![0.0f32; c.vocab];
        par_rows(&mut logits, 1, c.vocab * d, |r0, ch| {
            for (i, y) in ch.iter_mut().enumerate() {
                *y = simd::dot_f32(&last, &emb[(r0 + i) * d..(r0 + i + 1) * d]);
            }
        });
        Ok(logits)
    }
}

/// Decoder state: self-attention K/V per layer (grows), cross-attention K/V (shared between
/// clones, so beam hypotheses do not copy it) and the position.
#[derive(Clone)]
pub struct DecoderCache {
    self_k: Vec<Vec<f32>>,
    self_v: Vec<Vec<f32>>,
    cross: Arc<Vec<(Vec<f32>, Vec<f32>)>>,
    n_src: usize,
    pos: usize,
}

impl DecoderCache {
    pub fn position(&self) -> usize {
        self.pos
    }
}

// ---------------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------------

/// Whisper's timestamp rules: tokens `begin..` encode time in `precision_s` steps.
#[derive(Clone, Debug, PartialEq)]
pub struct TimestampRules {
    pub begin: usize,
    pub no_timestamps: usize,
    pub precision_s: f32,
    /// Largest first timestamp, in steps (Whisper uses 1.0 s).
    pub max_initial_steps: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// Forced prefix, for example `[sot, language, task, no_timestamps]`.
    pub prompt: Vec<usize>,
    pub eot: usize,
    pub max_new_tokens: usize,
    /// 1 = greedy (or sampled when `temperature > 0`); larger = beam search.
    pub beam_size: usize,
    pub temperature: f32,
    pub seed: u64,
    /// Ranked score is `sum_logprob / len^length_alpha` (Whisper's default is 1).
    pub length_alpha: f32,
    pub suppress: Vec<usize>,
    pub suppress_at_start: Vec<usize>,
    pub timestamps: Option<TimestampRules>,
}

impl DecodeOptions {
    pub fn new(prompt: Vec<usize>, eot: usize) -> Self {
        DecodeOptions {
            prompt,
            eot,
            max_new_tokens: 224,
            beam_size: 1,
            temperature: 0.0,
            seed: 0,
            length_alpha: 1.0,
            suppress: vec![],
            suppress_at_start: vec![],
            timestamps: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TsSegment {
    pub start_s: f32,
    pub end_s: f32,
    pub tokens: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    /// Generated tokens, without the prompt and without the end token.
    pub tokens: Vec<usize>,
    pub sum_logprob: f32,
    /// `sum_logprob` divided by the number of scored tokens (end token included when emitted).
    pub avg_logprob: f32,
    pub finished: bool,
    pub segments: Vec<TsSegment>,
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    let s: f64 = logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v - m) as f64).exp()
            } else {
                0.0
            }
        })
        .sum();
    let ln = s.ln() as f32;
    logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                v - m - ln
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

fn filter_logits(logits: &mut [f32], seq: &[usize], o: &DecodeOptions) {
    for &t in &o.suppress {
        if t < logits.len() {
            logits[t] = f32::NEG_INFINITY;
        }
    }
    if seq.is_empty() {
        for &t in &o.suppress_at_start {
            if t < logits.len() {
                logits[t] = f32::NEG_INFINITY;
            }
        }
    }
    let Some(ts) = &o.timestamps else { return };
    let (begin, n) = (ts.begin.min(logits.len()), logits.len());
    if ts.no_timestamps < n {
        logits[ts.no_timestamps] = f32::NEG_INFINITY;
    }
    let last_ts = seq.last().is_some_and(|&t| t >= ts.begin);
    let penult_ts = seq.len() < 2 || seq[seq.len() - 2] >= ts.begin;
    if last_ts {
        if penult_ts {
            for v in &mut logits[begin..] {
                *v = f32::NEG_INFINITY;
            }
        } else {
            for v in &mut logits[..o.eot.min(n)] {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    let stamps: Vec<usize> = seq.iter().copied().filter(|&t| t >= ts.begin).collect();
    if let Some(&last) = stamps.last() {
        let floor = if last_ts && !penult_ts {
            last
        } else {
            last + 1
        };
        for v in &mut logits[begin..floor.min(n).max(begin)] {
            *v = f32::NEG_INFINITY;
        }
    }
    if seq.is_empty() {
        for v in &mut logits[..begin] {
            *v = f32::NEG_INFINITY;
        }
        if let Some(m) = ts.max_initial_steps {
            for v in &mut logits[(begin + m + 1).min(n)..] {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    let lp = log_softmax(logits);
    let tsl =
        lp[begin..]
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, |a, v| {
                let v = v as f64;
                if a == f64::NEG_INFINITY {
                    v
                } else {
                    let m = a.max(v);
                    m + ((a - m).exp() + (v - m).exp()).ln()
                }
            });
    let max_text = lp[..begin]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    if tsl > max_text {
        for v in &mut logits[..begin] {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn segments_from(tokens: &[usize], ts: Option<&TimestampRules>) -> Vec<TsSegment> {
    let Some(ts) = ts else { return vec![] };
    let mut out = vec![];
    let mut cur: Option<(f32, Vec<usize>)> = None;
    for &t in tokens {
        if t >= ts.begin {
            let time = (t - ts.begin) as f32 * ts.precision_s;
            match cur.take() {
                Some((s, toks)) => out.push(TsSegment {
                    start_s: s,
                    end_s: time,
                    tokens: toks,
                }),
                None => cur = Some((time, vec![])),
            }
        } else if let Some((_, toks)) = cur.as_mut() {
            toks.push(t);
        }
    }
    if let Some((s, toks)) = cur.filter(|(_, t)| !t.is_empty()) {
        out.push(TsSegment {
            start_s: s,
            end_s: s,
            tokens: toks,
        });
    }
    out
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl Whisper {
    fn check_opts(&self, o: &DecodeOptions) -> Result<()> {
        let v = self.cfg.vocab;
        if o.prompt.is_empty() {
            return shape_err("the prompt needs at least the start token");
        }
        if o.prompt.iter().any(|&t| t >= v) || o.eot >= v {
            return shape_err("prompt or end token outside the vocabulary");
        }
        if o.suppress
            .iter()
            .chain(&o.suppress_at_start)
            .any(|&t| t >= v)
        {
            return shape_err("suppressed token outside the vocabulary");
        }
        if o.beam_size == 0 || o.beam_size > 64 {
            return shape_err("beam size must be in 1..=64");
        }
        if !(o.temperature.is_finite() && o.temperature >= 0.0 && o.length_alpha.is_finite()) {
            return shape_err("temperature / length_alpha");
        }
        if let Some(ts) = &o.timestamps {
            if ts.begin >= v
                || ts.no_timestamps >= v
                || !(ts.precision_s.is_finite() && ts.precision_s > 0.0)
            {
                return shape_err("timestamp rules");
            }
            if o.eot > ts.begin {
                return shape_err("the end token must come before the timestamp tokens");
            }
        }
        Ok(())
    }

    fn budget(&self, o: &DecodeOptions) -> usize {
        o.max_new_tokens
            .min(self.cfg.max_target.saturating_sub(o.prompt.len()))
    }

    /// Decodes from encoder output: greedy or sampled (`beam_size == 1`), else beam search.
    pub fn decode(&self, enc: &[f32], o: &DecodeOptions) -> Result<Decoded> {
        self.check_opts(o)?;
        if o.beam_size == 1 {
            self.greedy(enc, o)
        } else {
            self.beam(enc, o)
        }
    }

    fn finish(&self, toks: Vec<usize>, sum: f64, finished: bool, o: &DecodeOptions) -> Decoded {
        let scored = toks.len() + usize::from(finished);
        Decoded {
            segments: segments_from(&toks, o.timestamps.as_ref()),
            tokens: toks,
            sum_logprob: sum as f32,
            avg_logprob: if scored == 0 {
                0.0
            } else {
                (sum / scored as f64) as f32
            },
            finished,
        }
    }

    fn greedy(&self, enc: &[f32], o: &DecodeOptions) -> Result<Decoded> {
        let mut cache = self.start(enc)?;
        let mut logits = self.step(&mut cache, &o.prompt)?;
        let (mut toks, mut sum) = (vec![], 0.0f64);
        let mut rng = Rng(o.seed ^ 0x9E37_79B9_7F4A_7C15 | 1);
        for _ in 0..self.budget(o) {
            filter_logits(&mut logits, &toks, o);
            let lp = log_softmax(&logits);
            let next = if o.temperature > 0.0 {
                let scaled: Vec<f32> = logits.iter().map(|&v| v / o.temperature).collect();
                let p = log_softmax(&scaled);
                let r = rng.next();
                let mut acc = 0.0;
                let mut pick = None;
                for (i, &l) in p.iter().enumerate() {
                    if l.is_finite() {
                        acc += (l as f64).exp();
                        pick = Some(i);
                        if acc >= r {
                            break;
                        }
                    }
                }
                pick
            } else {
                lp.iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .fold(None, |b: Option<(usize, f32)>, (i, &v)| match b {
                        Some((_, bv)) if bv >= v => b,
                        _ => Some((i, v)),
                    })
                    .map(|b| b.0)
            };
            let Some(next) = next else { break };
            sum += lp[next] as f64;
            if next == o.eot {
                return Ok(self.finish(toks, sum, true, o));
            }
            toks.push(next);
            if toks.len() >= self.budget(o) {
                break;
            }
            logits = self.step(&mut cache, &[next])?;
        }
        Ok(self.finish(toks, sum, false, o))
    }

    fn beam(&self, enc: &[f32], o: &DecodeOptions) -> Result<Decoded> {
        struct B {
            toks: Vec<usize>,
            cache: DecoderCache,
            sum: f64,
            next: Vec<f32>,
        }
        let mut cache = self.start(enc)?;
        let first = self.step(&mut cache, &o.prompt)?;
        let mut beams = vec![B {
            toks: vec![],
            cache,
            sum: 0.0,
            next: first,
        }];
        let mut done: Vec<(Vec<usize>, f64)> = vec![];
        let score = |sum: f64, len: usize| sum / (len.max(1) as f64).powf(o.length_alpha as f64);
        let budget = self.budget(o);
        for step in 0..budget {
            let mut cands: Vec<(usize, usize, f64)> = vec![];
            for (bi, b) in beams.iter().enumerate() {
                let mut lg = b.next.clone();
                filter_logits(&mut lg, &b.toks, o);
                let lp = log_softmax(&lg);
                let mut idx: Vec<usize> = (0..lp.len()).filter(|&i| lp[i].is_finite()).collect();
                idx.sort_by(|&a, &c| lp[c].total_cmp(&lp[a]).then(a.cmp(&c)));
                for &t in idx.iter().take(o.beam_size + 1) {
                    cands.push((bi, t, b.sum + lp[t] as f64));
                }
            }
            cands.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.1.cmp(&b.1)).then(a.0.cmp(&b.0)));
            let mut keep: Vec<(usize, usize, f64)> = vec![];
            for (bi, t, s) in cands {
                if t == o.eot {
                    done.push((beams[bi].toks.clone(), s));
                } else if keep.len() < o.beam_size {
                    keep.push((bi, t, s));
                }
                if keep.len() == o.beam_size {
                    break;
                }
            }
            if keep.is_empty() || done.len() >= o.beam_size {
                beams.clear();
                break;
            }
            if step + 1 == budget {
                let mut last = vec![];
                for (bi, t, s) in keep {
                    let mut tk = beams[bi].toks.clone();
                    tk.push(t);
                    last.push(B {
                        toks: tk,
                        cache: beams[bi].cache.clone(),
                        sum: s,
                        next: vec![],
                    });
                }
                beams = last;
                break;
            }
            let mut nb = vec![];
            for (bi, t, s) in keep {
                let mut c = beams[bi].cache.clone();
                let next = self.step(&mut c, &[t])?;
                let mut tk = beams[bi].toks.clone();
                tk.push(t);
                nb.push(B {
                    toks: tk,
                    cache: c,
                    sum: s,
                    next,
                });
            }
            beams = nb;
        }
        let mut pool: Vec<(Vec<usize>, f64, bool)> =
            done.into_iter().map(|(t, s)| (t, s, true)).collect();
        if pool.is_empty() {
            pool = beams.into_iter().map(|b| (b.toks, b.sum, false)).collect();
        }
        let mut best: Option<(Vec<usize>, f64, bool, f64)> = None;
        for (toks, sum, fin) in pool {
            let sc = score(sum, toks.len() + usize::from(fin));
            if best.as_ref().is_none_or(|b| sc > b.3) {
                best = Some((toks, sum, fin, sc));
            }
        }
        match best {
            Some((t, s, f, _)) => Ok(self.finish(t, s, f, o)),
            None => Ok(self.finish(vec![], 0.0, false, o)),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Long-audio transcription
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    pub decode: DecodeOptions,
    /// Window the audio with the voice-activity detector; `None` cuts fixed chunks.
    pub vad: Option<VadConfig>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    /// Absolute times in seconds. Without timestamp rules each window is one segment.
    pub segments: Vec<TsSegment>,
    pub tokens: Vec<usize>,
    pub windows: usize,
}

impl Whisper {
    /// Transcribes 16 kHz (or the extractor's rate) mono audio in model-sized windows.
    pub fn transcribe(
        &self,
        audio: &[f32],
        mel: &MelExtractor,
        o: &TranscribeOptions,
    ) -> Result<Transcript> {
        let mc = mel.config();
        if mc.n_mels != self.cfg.n_mels {
            return shape_err("the mel extractor and the model disagree on the number of bins");
        }
        let chunk = mc.chunk_samples().max(1);
        let spans: Vec<Span> = match &o.vad {
            Some(v) => {
                let mut v = v.clone();
                v.sample_rate = mc.sample_rate;
                plan_windows(&detect_speech(audio, &v)?, chunk)
            }
            None => (0..audio.len().div_ceil(chunk))
                .map(|i| Span {
                    start: i * chunk,
                    end: ((i + 1) * chunk).min(audio.len()),
                })
                .collect(),
        };
        let mut out = Transcript {
            segments: vec![],
            tokens: vec![],
            windows: 0,
        };
        for sp in spans {
            let m = mel.log_mel(&audio[sp.start..sp.end], true)?;
            let enc = self.encode(&m)?;
            let dec = self.decode(&enc, &o.decode)?;
            let t0 = sp.start as f32 / mc.sample_rate as f32;
            out.windows += 1;
            if o.decode.timestamps.is_some() {
                out.segments.extend(dec.segments.iter().map(|s| TsSegment {
                    start_s: s.start_s + t0,
                    end_s: s.end_s + t0,
                    tokens: s.tokens.clone(),
                }));
            } else {
                out.segments.push(TsSegment {
                    start_s: t0,
                    end_s: sp.end as f32 / mc.sample_rate as f32,
                    tokens: dec.tokens.clone(),
                });
            }
            out.tokens.extend(dec.tokens);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// Audio embeddings for a language model
// ---------------------------------------------------------------------------------------------

/// Maps encoder frames to LLM embedding rows: stack `stack` neighbouring frames, then a linear
/// layer, optionally GELU and a second linear layer.
pub struct AudioProjector {
    stack: usize,
    l1: Linear,
    l2: Option<Linear>,
}

/// Rows of embeddings ready to splice into a language model's input.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioEmbeds {
    pub rows: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl AudioProjector {
    /// Builds a projector from explicit weights: `w1` is `hidden × (stack·d_model)`, `w2`
    /// (optional) `out × hidden`; all row-major.
    pub fn new(
        stack: usize,
        d_model: usize,
        w1: Vec<f32>,
        b1: Option<Vec<f32>>,
        w2: Option<(Vec<f32>, Option<Vec<f32>>)>,
    ) -> Result<AudioProjector> {
        if stack == 0 || stack > 64 {
            return shape_err("stack must be in 1..=64");
        }
        let inn = stack * d_model;
        if inn == 0 || w1.is_empty() || !w1.len().is_multiple_of(inn) {
            return shape_err("w1 is not a whole number of rows of stack·d_model");
        }
        let hidden = w1.len() / inn;
        if b1.as_ref().is_some_and(|b| b.len() != hidden) {
            return shape_err("b1 length");
        }
        let l1 = Linear {
            w: w1,
            out: hidden,
            inn,
            b: b1,
        };
        let l2 = match w2 {
            None => None,
            Some((w, b)) => {
                if w.is_empty() || w.len() % hidden != 0 {
                    return shape_err("w2 is not a whole number of rows of the hidden width");
                }
                let out = w.len() / hidden;
                if b.as_ref().is_some_and(|b| b.len() != out) {
                    return shape_err("b2 length");
                }
                Some(Linear {
                    w,
                    out,
                    inn: hidden,
                    b,
                })
            }
        };
        if l1
            .w
            .iter()
            .chain(l2.iter().flat_map(|l| l.w.iter()))
            .any(|v| !v.is_finite())
        {
            return Err(AudioError::Format(
                "projector weights contain NaN or infinity".into(),
            ));
        }
        Ok(AudioProjector { stack, l1, l2 })
    }

    /// Reads `{prefix}linear_1.{weight,bias}` and optionally `{prefix}linear_2.*` (names are a
    /// convention of this module, not taken from a particular checkpoint).
    pub fn from_source(
        src: &dyn TensorSource,
        prefix: &str,
        stack: usize,
        d_model: usize,
        hidden: usize,
        out: Option<usize>,
    ) -> Result<AudioProjector> {
        let read = |n: String, len: usize| -> Result<Vec<f32>> {
            let v = src.read(&n)?;
            if v.len() != len {
                return shape_err(format!("{n}: {} values, expected {len}", v.len()));
            }
            Ok(v)
        };
        let w1 = read(format!("{prefix}linear_1.weight"), hidden * stack * d_model)?;
        let b1n = format!("{prefix}linear_1.bias");
        let b1 = if src.has(&b1n) {
            Some(read(b1n, hidden)?)
        } else {
            None
        };
        let w2 = match out {
            None => None,
            Some(o) => {
                let b2n = format!("{prefix}linear_2.bias");
                let b2 = if src.has(&b2n) {
                    Some(read(b2n, o)?)
                } else {
                    None
                };
                Some((read(format!("{prefix}linear_2.weight"), o * hidden)?, b2))
            }
        };
        AudioProjector::new(stack, d_model, w1, b1, w2)
    }

    pub fn out_dim(&self) -> usize {
        self.l2.as_ref().map_or(self.l1.out, |l| l.out)
    }

    /// Projects `rows × d_model` encoder output; trailing frames that do not fill a stack are
    /// dropped.
    pub fn project(&self, enc: &[f32], d_model: usize) -> Result<AudioEmbeds> {
        if d_model == 0 || self.l1.inn != self.stack * d_model {
            return shape_err("projector and encoder widths differ");
        }
        if !enc.len().is_multiple_of(d_model) {
            return shape_err("encoder output must be a whole number of rows");
        }
        let n = enc.len() / d_model / self.stack;
        let mut h = self.l1.apply(&enc[..n * self.stack * d_model], n);
        let data = match &self.l2 {
            Some(l2) => {
                for v in h.iter_mut() {
                    *v = gelu(*v);
                }
                l2.apply(&h, n)
            }
            None => h,
        };
        Ok(AudioEmbeds {
            rows: n,
            dim: self.out_dim(),
            data,
        })
    }
}

/// Encodes `mel` and projects only the rows that come from real audio (not zero padding).
pub fn audio_embeddings(model: &Whisper, mel: &Mel, proj: &AudioProjector) -> Result<AudioEmbeds> {
    let d = model.cfg.d_model;
    let enc = model.encode(mel)?;
    let rows = (enc.len() / d).min(WhisperConfig::encoder_rows(mel.valid_frames.max(1)));
    proj.project(&enc[..rows * d], d)
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::approx_constant, clippy::unnecessary_cast, clippy::useless_vec)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Xs(u64);
    impl Xs {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn sine(f: f32, sr: u32, secs: f32, amp: f32) -> Vec<f32> {
        (0..(sr as f32 * secs) as usize)
            .map(|i| amp * (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin())
            .collect()
    }
    fn noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut r = Xs(seed | 1);
        (0..n).map(|_| amp * r.unit()).collect()
    }

    // ----- WAV -----

    #[test]
    fn wav_round_trips_in_every_format() {
        let mono: Vec<f32> = sine(440.0, 16_000, 0.05, 0.8);
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, -s * 0.5]).collect();
        let cases = [
            (SampleFormat::Pcm8, 1.0 / 128.0),
            (SampleFormat::Pcm16, 1.0 / 32768.0),
            (SampleFormat::Pcm24, 1.0 / 8_388_608.0),
            (SampleFormat::Pcm32, 1e-6),
            (SampleFormat::Float32, 0.0),
            (SampleFormat::Float64, 0.0),
        ];
        for (fmt, tol) in cases {
            for (ch, data) in [(1u16, &mono), (2u16, &stereo)] {
                let w = Wav {
                    sample_rate: 22_050,
                    channels: ch,
                    samples: data.clone(),
                };
                let bytes = write_wav(&w, fmt).unwrap();
                let back = read_wav(&bytes).unwrap();
                assert_eq!(
                    (back.sample_rate, back.channels, back.samples.len()),
                    (22_050, ch, data.len()),
                    "{fmt:?}"
                );
                let err = back
                    .samples
                    .iter()
                    .zip(data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    err <= tol * 1.01 + 1e-9,
                    "{fmt:?} {ch}ch error {err} > {tol}"
                );
            }
        }
        let w = Wav {
            sample_rate: 8000,
            channels: 1,
            samples: vec![0.1; 10],
        };
        assert_eq!(
            le16(&write_wav(&w, SampleFormat::Pcm24).unwrap(), 20),
            WAVE_EXT,
            "24-bit audio uses the extensible header"
        );
        assert_eq!(
            le16(&write_wav(&w, SampleFormat::Pcm16).unwrap(), 20),
            WAVE_PCM
        );
        assert_eq!(
            le16(&write_wav(&w, SampleFormat::Float32).unwrap(), 20),
            WAVE_FLOAT
        );
        let six = Wav {
            sample_rate: 48_000,
            channels: 6,
            samples: vec![0.25; 60],
        };
        assert_eq!(
            read_wav(&write_wav(&six, SampleFormat::Pcm16).unwrap())
                .unwrap()
                .channels,
            6
        );
    }

    #[test]
    fn wav_clamps_and_cleans_samples() {
        let w = Wav {
            sample_rate: 8000,
            channels: 1,
            samples: vec![2.0, -2.0, f32::NAN, f32::INFINITY],
        };
        let b = read_wav(&write_wav(&w, SampleFormat::Pcm16).unwrap()).unwrap();
        assert!(
            b.samples[0] > 0.999
                && b.samples[1] <= -0.999
                && b.samples[2] == 0.0
                && b.samples[3] == 0.0
        );
        assert_eq!(
            read_wav(&write_wav(&w, SampleFormat::Float32).unwrap())
                .unwrap()
                .samples[2..],
            [0.0, 0.0]
        );
    }

    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = id.to_vec();
        v.extend_from_slice(&(body.len() as u32).to_le_bytes());
        v.extend_from_slice(body);
        if body.len() % 2 == 1 {
            v.push(0);
        }
        v
    }
    fn riff(chunks: &[Vec<u8>]) -> Vec<u8> {
        let body: Vec<u8> = chunks.concat();
        let mut v = b"RIFF".to_vec();
        v.extend_from_slice(&(4 + body.len() as u32).to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend(body);
        v
    }
    fn fmt16(tag: u16, ch: u16, rate: u32, bits: u16) -> Vec<u8> {
        let mut b = vec![];
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ch.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&((rate as u64 * ch as u64 * bits as u64 / 8) as u32).to_le_bytes());
        b.extend_from_slice(&(ch * bits / 8).to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b
    }

    #[test]
    fn wav_skips_unknown_and_odd_sized_chunks() {
        let data: Vec<u8> = [1i16, -1, 16384]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f = riff(&[
            chunk(b"LIST", b"abc"),
            chunk(b"fmt ", &fmt16(1, 1, 8000, 16)),
            chunk(b"junk", b"12345"),
            chunk(b"data", &data),
            chunk(b"id3 ", b"tail"),
        ]);
        let w = read_wav(&f).unwrap();
        assert_eq!(w.samples, vec![1.0 / 32768.0, -1.0 / 32768.0, 0.5]);
    }

    #[test]
    fn wav_extensible_float_is_read() {
        let mut fmt = fmt16(WAVE_EXT, 1, 8000, 32);
        fmt.extend_from_slice(&22u16.to_le_bytes());
        fmt.extend_from_slice(&32u16.to_le_bytes());
        fmt.extend_from_slice(&4u32.to_le_bytes());
        fmt.extend_from_slice(&WAVE_FLOAT.to_le_bytes());
        fmt.extend_from_slice(&[0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        let data: Vec<u8> = [0.25f32, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w = read_wav(&riff(&[chunk(b"fmt ", &fmt), chunk(b"data", &data)])).unwrap();
        assert_eq!(w.samples, vec![0.25, -0.5]);
        let short = fmt16(WAVE_EXT, 1, 8000, 32);
        assert!(read_wav(&riff(&[chunk(b"fmt ", &short), chunk(b"data", &data)])).is_err());
    }

    #[test]
    fn wav_streamed_size_and_partial_frames() {
        let mut f = riff(&[chunk(b"fmt ", &fmt16(1, 2, 8000, 16))]);
        f.extend_from_slice(b"data");
        f.extend_from_slice(&u32::MAX.to_le_bytes());
        f.extend_from_slice(&[0, 0, 0, 64, 0, 0, 0, 64, 9, 9, 9]);
        let w = read_wav(&f).unwrap();
        assert_eq!(
            (w.frames(), w.samples.len()),
            (2, 4),
            "the partial trailing frame is dropped"
        );
        assert_eq!(w.samples[1], 0.5);
    }

    #[test]
    fn wav_hostile_files_are_errors_not_panics() {
        assert!(read_wav(&[]).is_err());
        assert!(read_wav(b"RIFF\0\0\0\0WAVE").is_err());
        assert!(read_wav(b"RIFX\0\0\0\0WAVE").is_err());
        let ok = riff(&[
            chunk(b"fmt ", &fmt16(1, 1, 8000, 16)),
            chunk(b"data", &[0, 1, 2, 3]),
        ]);
        assert!(read_wav(&ok).is_ok());
        for n in 0..ok.len() {
            let _ = read_wav(&ok[..n]);
        }
        for (what, bytes) in [
            (
                "zero channels",
                riff(&[
                    chunk(b"fmt ", &fmt16(1, 0, 8000, 16)),
                    chunk(b"data", &[0; 4]),
                ]),
            ),
            (
                "zero rate",
                riff(&[chunk(b"fmt ", &fmt16(1, 1, 0, 16)), chunk(b"data", &[0; 4])]),
            ),
            (
                "huge rate",
                riff(&[
                    chunk(b"fmt ", &fmt16(1, 1, 4_000_000_000, 16)),
                    chunk(b"data", &[0; 4]),
                ]),
            ),
            (
                "65 channels",
                riff(&[
                    chunk(b"fmt ", &fmt16(1, 65, 8000, 16)),
                    chunk(b"data", &[0; 4]),
                ]),
            ),
            ("no data", riff(&[chunk(b"fmt ", &fmt16(1, 1, 8000, 16))])),
            ("no fmt", riff(&[chunk(b"data", &[0; 4])])),
            (
                "12-bit",
                riff(&[
                    chunk(b"fmt ", &fmt16(1, 1, 8000, 12)),
                    chunk(b"data", &[0; 4]),
                ]),
            ),
        ] {
            assert!(read_wav(&bytes).is_err(), "{what}");
        }
        let adpcm = riff(&[
            chunk(b"fmt ", &fmt16(2, 1, 8000, 4)),
            chunk(b"data", &[0; 4]),
        ]);
        assert!(matches!(read_wav(&adpcm), Err(AudioError::Unsupported(_))));
        let lie = {
            let mut f = riff(&[chunk(b"fmt ", &fmt16(1, 1, 8000, 16))]);
            f.extend_from_slice(b"data");
            f.extend_from_slice(&0xFFFF_FF00u32.to_le_bytes());
            f.extend_from_slice(&[0; 8]);
            f
        };
        assert_eq!(
            read_wav(&lie).unwrap().samples.len(),
            4,
            "a lying data size reads what is there"
        );
        assert!(matches!(read_wav_with(&ok, 1), Err(AudioError::Limit(_))));
        let mut r = Xs(0xBEEF);
        for _ in 0..3000 {
            let mut f = ok.clone();
            for _ in 0..1 + r.next() % 4 {
                let i = (r.next() % f.len() as u64) as usize;
                f[i] = r.next() as u8;
            }
            let _ = read_wav(&f);
            let n = (r.next() % 200) as usize;
            let junk: Vec<u8> = (0..n).map(|_| r.next() as u8).collect();
            let _ = read_wav(&junk);
        }
        assert!(
            write_wav(
                &Wav {
                    sample_rate: 0,
                    channels: 1,
                    samples: vec![]
                },
                SampleFormat::Pcm16
            )
            .is_err()
        );
        assert!(
            write_wav(
                &Wav {
                    sample_rate: 8000,
                    channels: 2,
                    samples: vec![0.0; 3]
                },
                SampleFormat::Pcm16
            )
            .is_err()
        );
        assert!(
            write_wav(
                &Wav {
                    sample_rate: 8000,
                    channels: 0,
                    samples: vec![]
                },
                SampleFormat::Pcm16
            )
            .is_err()
        );
    }

    #[test]
    fn mono_mixdown_averages_channels() {
        assert_eq!(to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(to_mono(&[1.0, 2.0, 3.0], 1), vec![1.0, 2.0, 3.0]);
        assert_eq!(
            to_mono(&[1.0, 2.0, 3.0], 2),
            vec![1.5],
            "a ragged tail is dropped"
        );
        assert!(to_mono(&[], 2).is_empty());
    }

    // ----- resampling -----

    fn rms_err(a: &[f32], b: &[f32], skip: usize) -> f32 {
        let n = a.len().min(b.len()) - 2 * skip;
        (a[skip..skip + n]
            .iter()
            .zip(&b[skip..skip + n])
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            / n as f32)
            .sqrt()
    }

    #[test]
    fn resampling_keeps_in_band_tones() {
        for (from, to) in [
            (48_000u32, 16_000u32),
            (44_100, 16_000),
            (8_000, 16_000),
            (16_000, 22_050),
        ] {
            let x = sine(440.0, from, 0.5, 0.7);
            let y = resample(&x, from, to).unwrap();
            assert_eq!(
                y.len(),
                (x.len() as u64 * to as u64 + from as u64 / 2) as usize / from as usize
            );
            let want = sine(440.0, to, 0.5, 0.7);
            let e = rms_err(&y, &want, 600);
            assert!(e < 5e-3, "{from}->{to}: rms error {e}");
        }
        let x = sine(100.0, 16_000, 0.2, 0.3);
        assert_eq!(resample(&x, 16_000, 16_000).unwrap(), x);
        assert!(
            resample(&x, 0, 16_000).is_err()
                && resample(&x, 16_000, 0).is_err()
                && resample_with(&x, 8000, 16_000, 0).is_err()
        );
        assert!(resample(&[], 8000, 16_000).unwrap().is_empty());
        assert_eq!(resample(&[0.5], 8000, 16_000).unwrap().len(), 2);
    }

    #[test]
    fn resampling_removes_content_above_the_new_nyquist() {
        let x = sine(10_000.0, 48_000, 0.5, 0.8);
        let y = resample(&x, 48_000, 16_000).unwrap();
        let rms = (y[400..y.len() - 400].iter().map(|v| v * v).sum::<f32>()
            / (y.len() - 800) as f32)
            .sqrt();
        assert!(
            rms < 0.8 * 0.7071 * 0.005,
            "10 kHz is not removed: rms {rms}"
        );
    }

    #[test]
    fn resampling_preserves_dc() {
        let x = vec![0.5f32; 4000];
        let y = resample(&x, 16_000, 24_000).unwrap();
        for v in &y[200..y.len() - 200] {
            assert!((v - 0.5).abs() < 2e-3, "{v}");
        }
    }

    // ----- FFT -----

    fn naive_dft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut s = Complex::default();
                for (j, v) in x.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                    s = s.add(v.mul(Complex::new(a.cos(), a.sin())));
                }
                s
            })
            .collect()
    }

    #[test]
    fn fft_matches_the_naive_dft_for_many_lengths() {
        let mut r = Xs(7);
        for n in [
            1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 25, 30, 64, 97, 100, 121, 127, 360, 400,
            1000, 1024,
        ] {
            let x: Vec<Complex> = (0..n)
                .map(|_| Complex::new(r.unit() as f64, r.unit() as f64))
                .collect();
            let want = naive_dft(&x);
            let mut got = vec![Complex::default(); n];
            Fft::new(n).unwrap().forward(&x, &mut got).unwrap();
            let e = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs()))
                .fold(0.0, f64::max);
            assert!(e < 1e-9 * (n as f64).max(1.0), "n={n}: error {e}");
        }
    }

    #[test]
    fn fft_inverse_and_real_pair() {
        let n = 400;
        let mut r = Xs(11);
        let f = Fft::new(n).unwrap();
        let x: Vec<Complex> = (0..n)
            .map(|_| Complex::new(r.unit() as f64, r.unit() as f64))
            .collect();
        let (mut y, mut z) = (vec![Complex::default(); n], vec![Complex::default(); n]);
        f.forward(&x, &mut y).unwrap();
        f.inverse(&y, &mut z).unwrap();
        assert!(
            x.iter()
                .zip(&z)
                .all(|(a, b)| (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12)
        );
        let a: Vec<f32> = (0..n).map(|_| r.unit()).collect();
        let b: Vec<f32> = (0..n).map(|_| r.unit()).collect();
        let (mut oa, mut ob) = (
            vec![Complex::default(); n / 2 + 1],
            vec![Complex::default(); n / 2 + 1],
        );
        f.forward_real_pair(&a, &b, &mut oa, &mut ob).unwrap();
        for (sig, out) in [(&a, &oa), (&b, &ob)] {
            let c: Vec<Complex> = sig.iter().map(|&v| Complex::new(v as f64, 0.0)).collect();
            let want = naive_dft(&c);
            for k in 0..=n / 2 {
                assert!(
                    (want[k].re - out[k].re).abs() < 1e-9 && (want[k].im - out[k].im).abs() < 1e-9,
                    "bin {k}"
                );
            }
        }
        assert!(f.forward(&x[..3], &mut y).is_err());
        assert!(Fft::new(0).is_err() && Fft::new(usize::MAX).is_err());
    }

    // ----- mel -----

    #[test]
    fn mel_scale_has_the_slaney_landmarks() {
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-12);
        assert!(
            (hz_to_mel(6400.0) - 42.0).abs() < 1e-9,
            "{}",
            hz_to_mel(6400.0)
        );
        assert!((hz_to_mel(200.0) - 3.0).abs() < 1e-12);
        for f in [0.0, 100.0, 999.0, 1000.0, 2500.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-7 * (1.0 + f));
        }
    }

    #[test]
    fn whisper_mel_has_the_expected_geometry() {
        let m = MelExtractor::new(MelConfig::default()).unwrap();
        assert_eq!(m.center_hz().len(), 80);
        assert!(m.center_hz().windows(2).all(|w| w[1] > w[0]));
        assert!(m.center_hz()[0] > 0.0 && *m.center_hz().last().unwrap() < 8000.0);
        let x = sine(500.0, 16_000, 1.0, 0.5);
        let mel = m.log_mel(&x, true).unwrap();
        assert_eq!((mel.n_mels, mel.frames, mel.valid_frames), (80, 3000, 100));
        assert_eq!(mel.data.len(), 80 * 3000);
        assert!(mel.data.iter().all(|v| v.is_finite()));
        let (mn, mx) = mel
            .data
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(mx - mn <= 2.0 + 1e-5, "range {}", mx - mn);
        let m128 = MelExtractor::new(MelConfig::whisper_128()).unwrap();
        assert_eq!(m128.log_mel(&x, true).unwrap().n_mels, 128);
        let free = m.log_mel(&x, false).unwrap();
        assert_eq!(free.frames, 100, "unpadded: len / hop frames");
        let tm = free.time_major();
        assert_eq!(tm[3 * 80 + 5], free.at(5, 3));
    }

    #[test]
    fn pure_tones_peak_in_the_right_mel_bin() {
        let m = MelExtractor::new(MelConfig::default()).unwrap();
        let c = m.center_hz();
        for f in [150.0f32, 300.0, 1000.0, 2000.0, 3000.0, 6000.0] {
            let mel = m.log_mel(&sine(f, 16_000, 1.0, 0.5), false).unwrap();
            let t = 50;
            let best = (0..80)
                .max_by(|&a, &b| mel.at(a, t).total_cmp(&mel.at(b, t)))
                .unwrap();
            let spacing = if best + 1 < 80 {
                c[best + 1] - c[best]
            } else {
                c[best] - c[best - 1]
            };
            assert!(
                (c[best] - f as f64).abs() <= spacing,
                "{f} Hz peaked at bin {best} ({} Hz, spacing {spacing})",
                c[best]
            );
        }
    }

    #[test]
    fn a_chirp_climbs_the_mel_axis() {
        let (f0, f1, secs) = (200.0f32, 4000.0f32, 2.0f32);
        let k = (f1 - f0) / secs;
        let x: Vec<f32> = (0..32_000)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.5 * (2.0 * std::f32::consts::PI * (f0 * t + 0.5 * k * t * t)).sin()
            })
            .collect();
        let m = MelExtractor::new(MelConfig::default()).unwrap();
        let mel = m.log_mel(&x, false).unwrap();
        let c = m.center_hz();
        let mut last = 0;
        for t in [30usize, 60, 100, 140, 170] {
            let best = (0..80)
                .max_by(|&a, &b| mel.at(a, t).total_cmp(&mel.at(b, t)))
                .unwrap();
            assert!(
                best > last,
                "frame {t}: bin {best} did not climb past {last}"
            );
            let want = f0 + k * (t as f32 * 0.01);
            let sp = c[(best + 1).min(79)] - c[best.saturating_sub(1)];
            assert!(
                (c[best] - want as f64).abs() < 1.5 * sp as f64 + 60.0,
                "frame {t}: {} Hz vs {want}",
                c[best]
            );
            last = best;
        }
    }

    #[test]
    fn silence_and_odd_inputs_give_finite_constant_spectra() {
        let m = MelExtractor::new(MelConfig::default()).unwrap();
        let mel = m.log_mel(&[], true).unwrap();
        assert_eq!((mel.frames, mel.valid_frames), (3000, 0));
        assert!(
            mel.data.iter().all(|&v| (v + 1.5).abs() < 1e-6),
            "all-zero input is a constant -1.5"
        );
        assert_eq!(m.log_mel(&[], false).unwrap().frames, 0);
        let bad = [f32::NAN, f32::INFINITY, -f32::INFINITY, 0.3];
        assert!(
            m.log_mel(&bad, true)
                .unwrap()
                .data
                .iter()
                .all(|v| v.is_finite())
        );
        for n in [1usize, 2, 3, 159, 160, 161, 399, 400, 401] {
            let x = noise(n, 0.3, n as u64);
            let mel = m.log_mel(&x, false).unwrap();
            assert_eq!(mel.frames, n / 160, "n={n}");
            assert!(mel.data.iter().all(|v| v.is_finite()));
        }
        let long = noise(16_000 * 40, 0.3, 5);
        assert_eq!(
            m.log_mel(&long, true).unwrap().frames,
            3000,
            "over-long audio is trimmed to the chunk"
        );
    }

    #[test]
    fn mel_config_validation() {
        let d = MelConfig::default();
        for bad in [
            MelConfig {
                sample_rate: 0,
                ..d.clone()
            },
            MelConfig {
                n_fft: 1,
                ..d.clone()
            },
            MelConfig {
                hop: 0,
                ..d.clone()
            },
            MelConfig {
                n_mels: 0,
                ..d.clone()
            },
            MelConfig {
                n_mels: 1000,
                ..d.clone()
            },
            MelConfig {
                f_min: 9000.0,
                ..d.clone()
            },
            MelConfig {
                f_max: 20_000.0,
                ..d.clone()
            },
            MelConfig {
                f_min: f32::NAN,
                ..d.clone()
            },
            MelConfig {
                chunk_seconds: 0.0,
                ..d.clone()
            },
        ] {
            assert!(MelExtractor::new(bad.clone()).is_err(), "{bad:?}");
        }
        assert!(
            MelExtractor::new(MelConfig {
                n_fft: 512,
                hop: 128,
                n_mels: 40,
                f_min: 100.0,
                f_max: 7600.0,
                ..d
            })
            .is_ok()
        );
    }

    // ----- VAD -----

    fn burst(secs: f32, seed: u64) -> Vec<f32> {
        let n = (16_000.0 * secs) as usize;
        let a = sine(200.0, 16_000, secs, 0.25);
        let b = sine(1300.0, 16_000, secs, 0.15);
        let nz = noise(n, 0.01, seed);
        (0..n).map(|i| a[i] + b[i] + nz[i]).collect()
    }

    #[test]
    fn vad_finds_one_burst_in_noise() {
        let mut x = noise(16_000, 0.001, 1);
        x.extend(burst(1.0, 2));
        x.extend(noise(16_000, 0.001, 3));
        let s = detect_speech(&x, &VadConfig::default()).unwrap();
        assert_eq!(s.len(), 1, "{s:?}");
        assert!(
            (13_600..=16_800).contains(&s[0].start),
            "start {}",
            s[0].start
        );
        assert!((31_200..=34_400).contains(&s[0].end), "end {}", s[0].end);
    }

    #[test]
    fn vad_bridges_short_gaps_and_keeps_long_ones() {
        let build = |gap: f32| {
            let mut x = noise(8000, 0.001, 1);
            x.extend(burst(0.8, 2));
            x.extend(noise((16_000.0 * gap) as usize, 0.001, 3));
            x.extend(burst(0.8, 4));
            x.extend(noise(8000, 0.001, 5));
            x
        };
        assert_eq!(
            detect_speech(&build(0.1), &VadConfig::default())
                .unwrap()
                .len(),
            1
        );
        let two = detect_speech(&build(1.0), &VadConfig::default()).unwrap();
        assert_eq!(two.len(), 2, "{two:?}");
        assert!(two[0].end < two[1].start);
    }

    #[test]
    fn vad_rejects_clicks_silence_and_accepts_flat_loud_audio() {
        let mut x = noise(32_000, 0.001, 9);
        for v in &mut x[10_000..10_080] {
            *v = 0.5;
        }
        assert!(
            detect_speech(&x, &VadConfig::default()).unwrap().is_empty(),
            "a 5 ms click is not speech"
        );
        assert!(
            detect_speech(&vec![0.0; 16_000], &VadConfig::default())
                .unwrap()
                .is_empty()
        );
        assert!(
            detect_speech(&[], &VadConfig::default())
                .unwrap()
                .is_empty()
        );
        let loud = sine(300.0, 16_000, 2.0, 0.4);
        assert_eq!(
            detect_speech(&loud, &VadConfig::default()).unwrap(),
            vec![Span {
                start: 0,
                end: loud.len()
            }]
        );
        assert!(
            detect_speech(
                &loud,
                &VadConfig {
                    hop_ms: 0.0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            detect_speech(
                &loud,
                &VadConfig {
                    margin_db: f32::NAN,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(detect_speech(&[f32::NAN; 4000], &VadConfig::default()).is_ok());
    }

    #[test]
    fn vad_splits_long_segments_and_plans_windows() {
        let loud = sine(300.0, 16_000, 10.0, 0.4);
        let s = detect_speech(
            &loud,
            &VadConfig {
                max_segment_s: Some(3.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.len() >= 4, "{s:?}");
        assert_eq!((s[0].start, s.last().unwrap().end), (0, loud.len()));
        assert!(s.windows(2).all(|w| w[0].end == w[1].start));
        assert!(s.iter().all(|p| p.end - p.start <= 3 * 16_000));
        let w = plan_windows(
            &[
                Span { start: 0, end: 10 },
                Span { start: 12, end: 30 },
                Span {
                    start: 100,
                    end: 400,
                },
            ],
            30,
        );
        assert_eq!(
            w,
            vec![
                Span { start: 0, end: 30 },
                Span {
                    start: 100,
                    end: 130
                },
                Span {
                    start: 130,
                    end: 160
                },
                Span {
                    start: 160,
                    end: 190
                },
                Span {
                    start: 190,
                    end: 220
                },
                Span {
                    start: 220,
                    end: 250
                },
                Span {
                    start: 250,
                    end: 280
                },
                Span {
                    start: 280,
                    end: 310
                },
                Span {
                    start: 310,
                    end: 340
                },
                Span {
                    start: 340,
                    end: 370
                },
                Span {
                    start: 370,
                    end: 400
                }
            ]
        );
        assert!(plan_windows(&[], 10).is_empty());
    }

    // ----- numeric helpers -----

    #[test]
    fn erf_and_gelu_match_reference_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.5, 0.999_593_047_982_555),
            (3.0, 0.999_977_909_503_001_4),
            (-1.0, -0.842_700_792_949_714_9),
            (7.0, 1.0),
        ] {
            assert!((erf(x) - want).abs() < 1e-12, "erf({x}) = {}", erf(x));
        }
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6 && (gelu(-1.0) + 0.158_655_3).abs() < 1e-6);
        assert!((gelu(10.0) - 10.0).abs() < 1e-6 && gelu(-10.0).abs() < 1e-6);
    }

    #[test]
    fn sinusoid_table_layout() {
        let p = sinusoids(4, 8);
        assert_eq!(&p[0..4], &[0.0; 4]);
        assert_eq!(&p[4..8], &[1.0; 4]);
        assert!((p[8] - 1.0f32.sin()).abs() < 1e-6 && (p[8 + 4] - 1.0f32.cos()).abs() < 1e-6);
        assert!(p.iter().all(|v| v.abs() <= 1.0));
    }

    // ----- Whisper -----

    struct Mem(HashMap<String, Vec<f32>>);
    impl TensorSource for Mem {
        fn read(&self, name: &str) -> std::result::Result<Vec<f32>, LoadError> {
            self.0
                .get(name)
                .cloned()
                .ok_or_else(|| LoadError::Missing(name.into()))
        }
        fn has(&self, name: &str) -> bool {
            self.0.contains_key(name)
        }
    }

    fn tiny_cfg() -> WhisperConfig {
        WhisperConfig {
            n_mels: 8,
            d_model: 16,
            enc_layers: 2,
            dec_layers: 2,
            enc_heads: 2,
            dec_heads: 2,
            enc_ffn: 32,
            dec_ffn: 32,
            vocab: 40,
            max_source: 32,
            max_target: 24,
        }
    }
    fn tiny_mel_cfg() -> MelConfig {
        MelConfig {
            sample_rate: 16_000,
            n_fft: 64,
            hop: 32,
            n_mels: 8,
            f_min: 0.0,
            f_max: 8000.0,
            chunk_seconds: 0.128,
        }
    }

    fn tiny_tensors(
        c: &WhisperConfig,
        p: &str,
        seed: u64,
        with_enc_pos: bool,
    ) -> HashMap<String, Vec<f32>> {
        let mut r = Xs(seed | 1);
        let mut m = HashMap::new();
        let d = c.d_model;
        let mut w = |m: &mut HashMap<String, Vec<f32>>, n: String, len: usize, s: f32| {
            m.insert(n, (0..len).map(|_| r.unit() * s).collect());
        };
        let ones = |m: &mut HashMap<String, Vec<f32>>, n: String, len: usize, v: f32| {
            m.insert(n, vec![v; len]);
        };
        let norm = |m: &mut HashMap<String, Vec<f32>>, n: &str| {
            m.insert(format!("{p}{n}.weight"), vec![1.0; d]);
            m.insert(format!("{p}{n}.bias"), vec![0.0; d]);
        };
        w(
            &mut m,
            format!("{p}encoder.conv1.weight"),
            d * c.n_mels * 3,
            0.3,
        );
        w(&mut m, format!("{p}encoder.conv1.bias"), d, 0.05);
        w(&mut m, format!("{p}encoder.conv2.weight"), d * d * 3, 0.2);
        w(&mut m, format!("{p}encoder.conv2.bias"), d, 0.05);
        if with_enc_pos {
            m.insert(
                format!("{p}encoder.embed_positions.weight"),
                sinusoids(c.max_source, d),
            );
        }
        let mut attn = |m: &mut HashMap<String, Vec<f32>>, b: &str| {
            for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                w(m, format!("{p}{b}.{n}.weight"), d * d, 0.25);
            }
            for n in ["q_proj", "v_proj", "out_proj"] {
                ones(m, format!("{p}{b}.{n}.bias"), d, 0.01);
            }
        };
        for i in 0..c.enc_layers {
            let b = format!("encoder.layers.{i}");
            attn(&mut m, &format!("{b}.self_attn"));
            norm(&mut m, &format!("{b}.self_attn_layer_norm"));
            norm(&mut m, &format!("{b}.final_layer_norm"));
        }
        for i in 0..c.dec_layers {
            let b = format!("decoder.layers.{i}");
            attn(&mut m, &format!("{b}.self_attn"));
            attn(&mut m, &format!("{b}.encoder_attn"));
            for n in [
                "self_attn_layer_norm",
                "encoder_attn_layer_norm",
                "final_layer_norm",
            ] {
                norm(&mut m, &format!("{b}.{n}"));
            }
        }
        let mut r2 = Xs(seed.wrapping_mul(31) | 1);
        let mut w2 = |m: &mut HashMap<String, Vec<f32>>, n: String, len: usize, s: f32| {
            m.insert(n, (0..len).map(|_| r2.unit() * s).collect());
        };
        for i in 0..c.enc_layers {
            let b = format!("encoder.layers.{i}");
            w2(&mut m, format!("{p}{b}.fc1.weight"), c.enc_ffn * d, 0.25);
            w2(&mut m, format!("{p}{b}.fc1.bias"), c.enc_ffn, 0.05);
            w2(&mut m, format!("{p}{b}.fc2.weight"), d * c.enc_ffn, 0.25);
            w2(&mut m, format!("{p}{b}.fc2.bias"), d, 0.05);
        }
        for i in 0..c.dec_layers {
            let b = format!("decoder.layers.{i}");
            w2(&mut m, format!("{p}{b}.fc1.weight"), c.dec_ffn * d, 0.25);
            w2(&mut m, format!("{p}{b}.fc1.bias"), c.dec_ffn, 0.05);
            w2(&mut m, format!("{p}{b}.fc2.weight"), d * c.dec_ffn, 0.25);
            w2(&mut m, format!("{p}{b}.fc2.bias"), d, 0.05);
        }
        norm(&mut m, "encoder.layer_norm");
        norm(&mut m, "decoder.layer_norm");
        w2(
            &mut m,
            format!("{p}decoder.embed_tokens.weight"),
            c.vocab * d,
            1.0,
        );
        w2(
            &mut m,
            format!("{p}decoder.embed_positions.weight"),
            c.max_target * d,
            0.3,
        );
        m
    }

    fn tiny_model(seed: u64) -> Whisper {
        Whisper::load(
            tiny_cfg(),
            &Mem(tiny_tensors(&tiny_cfg(), "model.", seed, false)),
        )
        .unwrap()
    }

    fn tiny_mel(freq: f32) -> Mel {
        MelExtractor::new(tiny_mel_cfg())
            .unwrap()
            .log_mel(&sine(freq, 16_000, 0.128, 0.5), true)
            .unwrap()
    }

    #[test]
    fn config_json_parses_and_validates() {
        let ok = r#"{"num_mel_bins":80,"d_model":384,"encoder_layers":4,"decoder_layers":4,"encoder_attention_heads":6,"decoder_attention_heads":6,"encoder_ffn_dim":1536,"decoder_ffn_dim":1536,"vocab_size":51865,"max_source_positions":1500,"max_target_positions":448}"#;
        let c = WhisperConfig::from_json(ok).unwrap();
        assert_eq!(
            (c.d_model, c.vocab, c.enc_ffn, c.max_source),
            (384, 51865, 1536, 1500)
        );
        let defaults = WhisperConfig::from_json(r#"{"d_model":64,"encoder_layers":1,"decoder_layers":1,"encoder_attention_heads":2,"decoder_attention_heads":2,"vocab_size":100}"#).unwrap();
        assert_eq!(
            (defaults.n_mels, defaults.enc_ffn, defaults.max_target),
            (80, 256, 448)
        );
        for bad in [
            "",
            "[]",
            "{}",
            r#"{"d_model":65,"encoder_layers":1,"decoder_layers":1,"encoder_attention_heads":2,"decoder_attention_heads":2,"vocab_size":100}"#,
            r#"{"d_model":64,"encoder_layers":1,"decoder_layers":1,"encoder_attention_heads":3,"decoder_attention_heads":2,"vocab_size":100}"#,
            r#"{"d_model":64,"encoder_layers":0,"decoder_layers":1,"encoder_attention_heads":2,"decoder_attention_heads":2,"vocab_size":100}"#,
            r#"{"d_model":4000000000,"encoder_layers":1,"decoder_layers":1,"encoder_attention_heads":2,"decoder_attention_heads":2,"vocab_size":100}"#,
            r#"{"d_model":64,"encoder_layers":1,"decoder_layers":1,"encoder_attention_heads":2,"decoder_attention_heads":2,"vocab_size":0}"#,
        ] {
            assert!(WhisperConfig::from_json(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn loading_checks_names_shapes_and_values() {
        let c = tiny_cfg();
        let full = tiny_tensors(&c, "model.", 3, false);
        assert!(Whisper::load(c.clone(), &Mem(full.clone())).is_ok());
        assert!(
            Whisper::load(c.clone(), &Mem(tiny_tensors(&c, "", 3, true))).is_ok(),
            "no `model.` prefix, explicit positions"
        );
        let mut m = full.clone();
        m.remove("model.decoder.layers.1.encoder_attn.k_proj.weight");
        match Whisper::load(c.clone(), &Mem(m)) {
            Err(AudioError::Missing(n)) => {
                assert!(n.contains("decoder.layers.1.encoder_attn.k_proj"), "{n}")
            }
            Err(e) => panic!("{e}"),
            Ok(_) => panic!("loaded"),
        }
        let mut m = full.clone();
        m.get_mut("model.encoder.conv1.weight").unwrap().pop();
        assert!(matches!(
            Whisper::load(c.clone(), &Mem(m)),
            Err(AudioError::Shape(_))
        ));
        let mut m = full.clone();
        m.get_mut("model.decoder.embed_tokens.weight").unwrap()[5] = f32::NAN;
        assert!(matches!(
            Whisper::load(c.clone(), &Mem(m)),
            Err(AudioError::Format(_))
        ));
        let mut m = full;
        m.remove("model.encoder.layers.0.self_attn.q_proj.bias");
        assert!(
            matches!(Whisper::load(c, &Mem(m)), Err(AudioError::Missing(_))),
            "q_proj needs its bias"
        );
    }

    #[test]
    fn encoder_shapes_are_deterministic_and_checked() {
        let w = tiny_model(1);
        let mel = tiny_mel(1000.0);
        assert_eq!((mel.frames, mel.n_mels), (64, 8));
        let a = w.encode(&mel).unwrap();
        assert_eq!(a.len(), 32 * 16);
        assert!(a.iter().all(|v| v.is_finite()));
        assert_eq!(a, w.encode(&mel).unwrap(), "encoding is deterministic");
        assert_ne!(
            a,
            w.encode(&tiny_mel(3000.0)).unwrap(),
            "different audio, different encoding"
        );
        assert_ne!(a, tiny_model(2).encode(&mel).unwrap());
        let short = Mel {
            n_mels: 8,
            frames: 21,
            data: mel.data[..8 * 21].to_vec(),
            valid_frames: 21,
        };
        let _ = short;
        let mut cut = vec![0.0; 8 * 21];
        for m in 0..8 {
            for t in 0..21 {
                cut[m * 21 + t] = mel.at(m, t);
            }
        }
        assert_eq!(
            w.encode(&Mel {
                n_mels: 8,
                frames: 21,
                data: cut,
                valid_frames: 21
            })
            .unwrap()
            .len(),
            11 * 16
        );
        assert!(matches!(
            w.encode(&Mel {
                n_mels: 7,
                frames: 4,
                data: vec![0.0; 28],
                valid_frames: 4
            }),
            Err(AudioError::Shape(_))
        ));
        assert!(matches!(
            w.encode(&Mel {
                n_mels: 8,
                frames: 70,
                data: vec![0.0; 560],
                valid_frames: 70
            }),
            Err(AudioError::Limit(_))
        ));
        assert!(
            w.encode(&Mel {
                n_mels: 8,
                frames: 0,
                data: vec![],
                valid_frames: 0
            })
            .is_err()
        );
        assert!(
            w.encode(&Mel {
                n_mels: 8,
                frames: 4,
                data: vec![0.0; 3],
                valid_frames: 4
            })
            .is_err()
        );
        let with_pos = Whisper::load(
            tiny_cfg(),
            &Mem(tiny_tensors(&tiny_cfg(), "model.", 1, true)),
        )
        .unwrap();
        assert_eq!(
            with_pos.encode(&mel).unwrap(),
            a,
            "an absent position table falls back to the sinusoids"
        );
    }

    #[test]
    fn cached_steps_equal_one_big_step() {
        let w = tiny_model(4);
        let enc = w.encode(&tiny_mel(700.0)).unwrap();
        let prompt = [1usize, 7, 3, 9, 2, 5];
        let mut a = w.start(&enc).unwrap();
        let all = w.step(&mut a, &prompt).unwrap();
        let mut b = w.start(&enc).unwrap();
        let mut last = vec![];
        for &t in &prompt {
            last = w.step(&mut b, &[t]).unwrap();
        }
        assert_eq!((a.position(), b.position()), (6, 6));
        let d = all
            .iter()
            .zip(&last)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(d < 1e-4, "batched vs one-by-one differ by {d}");
        let mut c = w.start(&enc).unwrap();
        w.step(&mut c, &prompt[..4]).unwrap();
        let mixed = w.step(&mut c, &prompt[4..]).unwrap();
        assert!(all.iter().zip(&mixed).all(|(x, y)| (x - y).abs() < 1e-4));
        let next_a = w.step(&mut a, &[11]).unwrap();
        let next_b = w.step(&mut b, &[11]).unwrap();
        assert!(
            next_a
                .iter()
                .zip(&next_b)
                .all(|(x, y)| (x - y).abs() < 1e-4),
            "the caches stay equivalent"
        );
    }

    #[test]
    fn step_rejects_bad_input_without_changing_the_cache() {
        let w = tiny_model(5);
        let enc = w.encode(&tiny_mel(700.0)).unwrap();
        let mut c = w.start(&enc).unwrap();
        w.step(&mut c, &[1, 2]).unwrap();
        assert!(w.step(&mut c, &[]).is_err());
        assert!(matches!(w.step(&mut c, &[40]), Err(AudioError::Shape(_))));
        assert!(matches!(
            w.step(&mut c, &vec![1; 23]),
            Err(AudioError::Limit(_))
        ));
        assert_eq!(c.position(), 2);
        assert!(w.step(&mut c, &vec![1; 22]).is_ok());
        assert!(w.start(&[]).is_err() && w.start(&[0.0; 5]).is_err());
    }

    fn opts(eot: usize) -> DecodeOptions {
        DecodeOptions {
            max_new_tokens: 12,
            ..DecodeOptions::new(vec![1, 2], eot)
        }
    }

    #[test]
    fn greedy_decoding_is_deterministic_and_scores_match_a_replay() {
        let w = tiny_model(6);
        let enc = w.encode(&tiny_mel(900.0)).unwrap();
        let o = opts(30);
        let a = w.decode(&enc, &o).unwrap();
        assert_eq!(a, w.decode(&enc, &o).unwrap());
        assert!(a.tokens.len() <= 12 && a.tokens.iter().all(|&t| t < 40 && t != 30));
        let mut c = w.start(&enc).unwrap();
        let mut lg = w.step(&mut c, &o.prompt).unwrap();
        let mut sum = 0.0f64;
        let mut seq = a.tokens.clone();
        if a.finished {
            seq.push(30);
        }
        for &t in &seq {
            sum += log_softmax(&lg)[t] as f64;
            lg = w.step(&mut c, &[t]).unwrap_or_default();
        }
        assert!(
            (sum - a.sum_logprob as f64).abs() < 1e-3,
            "{sum} vs {}",
            a.sum_logprob
        );
        assert!((a.avg_logprob as f64 - sum / seq.len().max(1) as f64).abs() < 1e-3);
        let sampled = |seed| {
            w.decode(
                &enc,
                &DecodeOptions {
                    temperature: 1.5,
                    seed,
                    ..opts(30)
                },
            )
            .unwrap()
        };
        assert_eq!(sampled(7), sampled(7));
        assert!(
            (0..8)
                .map(|s| sampled(s).tokens)
                .collect::<std::collections::HashSet<_>>()
                .len()
                > 1,
            "sampling varies with the seed"
        );
    }

    #[test]
    fn suppression_and_budgets_are_honoured() {
        let w = tiny_model(8);
        let enc = w.encode(&tiny_mel(900.0)).unwrap();
        let free = w.decode(&enc, &opts(30)).unwrap();
        assert!(!free.tokens.is_empty());
        let banned: Vec<usize> = free.tokens.iter().copied().take(3).collect();
        let d = w
            .decode(
                &enc,
                &DecodeOptions {
                    suppress: banned.clone(),
                    ..opts(30)
                },
            )
            .unwrap();
        assert!(d.tokens.iter().all(|t| !banned.contains(t)));
        let first = free.tokens[0];
        let d = w
            .decode(
                &enc,
                &DecodeOptions {
                    suppress_at_start: vec![first],
                    ..opts(30)
                },
            )
            .unwrap();
        assert_ne!(d.tokens.first(), Some(&first));
        let d = w
            .decode(
                &enc,
                &DecodeOptions {
                    max_new_tokens: 3,
                    ..opts(39)
                },
            )
            .unwrap();
        assert!(d.tokens.len() <= 3);
        let d = w
            .decode(
                &enc,
                &DecodeOptions {
                    max_new_tokens: 0,
                    ..opts(30)
                },
            )
            .unwrap();
        assert!(d.tokens.is_empty() && !d.finished);
        let long = DecodeOptions {
            max_new_tokens: 500,
            ..DecodeOptions::new(vec![1; 20], 30)
        };
        assert!(
            w.decode(&enc, &long).unwrap().tokens.len() <= 4,
            "the decoder position limit caps the output"
        );
    }

    #[test]
    fn beam_search_of_width_one_equals_greedy_and_wider_beams_are_valid() {
        for seed in [6u64, 9, 12] {
            let w = tiny_model(seed);
            let enc = w.encode(&tiny_mel(500.0 + 300.0 * seed as f32)).unwrap();
            let g = w.decode(&enc, &opts(30)).unwrap();
            let b1 = w.beam(&enc, &opts(30)).unwrap();
            assert_eq!(b1.tokens, g.tokens, "seed {seed}");
            assert!((b1.sum_logprob - g.sum_logprob).abs() < 1e-3);
            for width in [2usize, 4, 8] {
                let o = DecodeOptions {
                    beam_size: width,
                    ..opts(30)
                };
                let a = w.decode(&enc, &o).unwrap();
                assert_eq!(a, w.decode(&enc, &o).unwrap(), "deterministic");
                assert!(a.tokens.len() <= 12 && a.tokens.iter().all(|&t| t < 40 && t != 30));
                assert!(a.sum_logprob <= 0.0);
                let norm = |d: &Decoded| {
                    d.sum_logprob / (d.tokens.len() + usize::from(d.finished)).max(1) as f32
                };
                assert!(
                    norm(&a) + 0.5 >= norm(&g) || a.tokens.len() < 12,
                    "width {width} is much worse than greedy"
                );
            }
        }
        let w = tiny_model(6);
        let enc = w.encode(&tiny_mel(900.0)).unwrap();
        assert!(
            w.decode(
                &enc,
                &DecodeOptions {
                    beam_size: 0,
                    ..opts(30)
                }
            )
            .is_err()
        );
        assert!(
            w.decode(
                &enc,
                &DecodeOptions {
                    beam_size: 65,
                    ..opts(30)
                }
            )
            .is_err()
        );
        assert!(w.decode(&enc, &DecodeOptions::new(vec![], 30)).is_err());
        assert!(w.decode(&enc, &DecodeOptions::new(vec![99], 30)).is_err());
        assert!(
            w.decode(
                &enc,
                &DecodeOptions {
                    temperature: -1.0,
                    ..opts(30)
                }
            )
            .is_err()
        );
    }

    fn ts_opts() -> DecodeOptions {
        DecodeOptions {
            timestamps: Some(TimestampRules {
                begin: 32,
                no_timestamps: 31,
                precision_s: 0.02,
                max_initial_steps: Some(2),
            }),
            max_new_tokens: 16,
            ..DecodeOptions::new(vec![1, 2], 30)
        }
    }

    #[test]
    fn timestamp_rules_shape_the_output() {
        let mut saw_pairs = false;
        for seed in 1..=6u64 {
            let w = tiny_model(seed);
            let enc = w.encode(&tiny_mel(400.0 + 100.0 * seed as f32)).unwrap();
            for beam in [1usize, 3] {
                let toks = w
                    .decode(
                        &enc,
                        &DecodeOptions {
                            beam_size: beam,
                            temperature: if beam == 1 { 1.2 } else { 0.0 },
                            seed,
                            ..ts_opts()
                        },
                    )
                    .unwrap()
                    .tokens;
                if toks.is_empty() {
                    continue;
                }
                assert!(
                    toks[0] >= 32 && toks[0] <= 34,
                    "the first token is an early timestamp: {toks:?}"
                );
                assert!(!toks.contains(&31), "no_timestamps never appears");
                let mut prev_ts: Option<usize> = None;
                for (i, &t) in toks.iter().enumerate() {
                    let is_ts = t >= 32;
                    let last_ts = i > 0 && toks[i - 1] >= 32;
                    let penult_ts = i < 2 || toks[i - 2] >= 32;
                    if last_ts && !penult_ts {
                        assert!(
                            t >= 30,
                            "after a lone timestamp comes a timestamp or the end: {toks:?}"
                        );
                    }
                    if last_ts && penult_ts && i >= 2 {
                        assert!(!is_ts, "after a pair comes text: {toks:?}");
                    }
                    if is_ts {
                        if let Some(p) = prev_ts {
                            assert!(t >= p, "timestamps never go back: {toks:?}");
                        }
                        prev_ts = Some(t);
                    }
                    saw_pairs |= i > 0 && is_ts && last_ts;
                }
            }
        }
        assert!(
            saw_pairs,
            "no timestamp pair was ever produced; the test explored too little"
        );
    }

    #[test]
    fn segments_are_cut_at_timestamp_pairs() {
        let r = TimestampRules {
            begin: 100,
            no_timestamps: 99,
            precision_s: 0.02,
            max_initial_steps: None,
        };
        let s = segments_from(&[100, 5, 6, 110, 110, 7, 120, 130, 8], Some(&r));
        assert_eq!(s.len(), 3);
        assert!(
            s[0].start_s == 0.0 && (s[0].end_s - 0.2).abs() < 1e-6 && s[0].tokens == vec![5, 6]
        );
        assert!(
            (s[1].start_s - 0.2).abs() < 1e-6
                && (s[1].end_s - 0.4).abs() < 1e-6
                && s[1].tokens == vec![7]
        );
        assert_eq!(s[2].tokens, vec![8], "an unterminated tail is kept");
        assert!(segments_from(&[5, 6], Some(&r)).is_empty());
        assert!(
            segments_from(&[100, 110], Some(&r))
                .iter()
                .all(|s| s.tokens.is_empty())
                || segments_from(&[100, 110], Some(&r)).len() == 1
        );
        assert!(segments_from(&[1, 2], None).is_empty());
    }

    #[test]
    fn long_audio_is_windowed_and_timed() {
        let w = tiny_model(10);
        let mel = MelExtractor::new(tiny_mel_cfg()).unwrap();
        let audio = sine(600.0, 16_000, 1.0, 0.4);
        let plain = TranscribeOptions {
            decode: DecodeOptions {
                max_new_tokens: 4,
                ..opts(30)
            },
            vad: None,
        };
        let t = w.transcribe(&audio, &mel, &plain).unwrap();
        assert_eq!(t.windows, 8, "1 s in 0.128 s chunks");
        assert_eq!(t.segments.len(), 8);
        assert!(
            (t.segments[1].start_s - 0.128).abs() < 1e-6
                && (t.segments[7].end_s - 1.0).abs() < 1e-6
        );
        assert_eq!(
            t.tokens.len(),
            t.segments.iter().map(|s| s.tokens.len()).sum::<usize>()
        );
        assert_eq!(t, w.transcribe(&audio, &mel, &plain).unwrap());
        let stamped = TranscribeOptions {
            decode: DecodeOptions {
                max_new_tokens: 6,
                ..ts_opts()
            },
            vad: None,
        };
        let t = w.transcribe(&audio, &mel, &stamped).unwrap();
        assert!(
            t.segments
                .iter()
                .all(|s| s.start_s >= 0.0 && s.end_s >= s.start_s)
        );
        let mut x = noise(4000, 0.001, 1);
        x.extend(burst(0.3, 2));
        x.extend(noise(4000, 0.001, 3));
        let vad = TranscribeOptions {
            decode: plain.decode.clone(),
            vad: Some(VadConfig::default()),
        };
        let t = w.transcribe(&x, &mel, &vad).unwrap();
        assert!((3..=6).contains(&t.windows), "{}", t.windows);
        assert_eq!(w.transcribe(&[], &mel, &plain).unwrap().windows, 0);
        assert_eq!(
            w.transcribe(&noise(8000, 0.001, 4), &mel, &vad)
                .unwrap()
                .windows,
            0,
            "silence has nothing to transcribe"
        );
        let wrong = MelExtractor::new(MelConfig {
            n_mels: 9,
            ..tiny_mel_cfg()
        })
        .unwrap();
        assert!(w.transcribe(&audio, &wrong, &plain).is_err());
    }

    #[test]
    fn audio_embeddings_cover_only_real_audio() {
        let w = tiny_model(11);
        let mut r = Xs(99);
        let mut t = HashMap::new();
        t.insert(
            "proj.linear_1.weight".to_string(),
            (0..24 * 32).map(|_| r.unit() * 0.2).collect::<Vec<f32>>(),
        );
        t.insert("proj.linear_1.bias".to_string(), vec![0.1; 24]);
        t.insert(
            "proj.linear_2.weight".to_string(),
            (0..20 * 24).map(|_| r.unit() * 0.2).collect::<Vec<f32>>(),
        );
        let p = AudioProjector::from_source(&Mem(t.clone()), "proj.", 2, 16, 24, Some(20)).unwrap();
        assert_eq!(p.out_dim(), 20);
        let mel = MelExtractor::new(tiny_mel_cfg())
            .unwrap()
            .log_mel(&sine(800.0, 16_000, 0.0625, 0.5), true)
            .unwrap();
        assert_eq!(mel.valid_frames, 31);
        let e = audio_embeddings(&w, &mel, &p).unwrap();
        assert_eq!((e.rows, e.dim, e.data.len()), (8, 20, 160));
        assert!(e.data.iter().all(|v| v.is_finite()));
        assert_eq!(e, audio_embeddings(&w, &mel, &p).unwrap());
        let none = AudioProjector::from_source(&Mem(t.clone()), "proj.", 2, 16, 24, None).unwrap();
        assert_eq!(audio_embeddings(&w, &mel, &none).unwrap().dim, 24);
        assert!(AudioProjector::from_source(&Mem(t.clone()), "nope.", 2, 16, 24, None).is_err());
        assert!(
            AudioProjector::from_source(&Mem(t.clone()), "proj.", 3, 16, 24, None).is_err(),
            "wrong stack => wrong shape"
        );
        assert!(AudioProjector::new(0, 16, vec![0.0; 16], None, None).is_err());
        assert!(AudioProjector::new(2, 16, vec![f32::NAN; 32], None, None).is_err());
        assert!(p.project(&[0.0; 17], 16).is_err() && p.project(&[0.0; 32], 8).is_err());
        assert_eq!(
            p.project(&[0.5; 48], 16).unwrap().rows,
            1,
            "a trailing unpaired frame is dropped"
        );
    }

    fn write_safetensors(path: &Path, tensors: &HashMap<String, Vec<f32>>) {
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        let (mut header, mut body) = (serde_json::Map::new(), Vec::<u8>::new());
        for n in names {
            let v = &tensors[n];
            let start = body.len();
            for x in v {
                body.extend_from_slice(&x.to_le_bytes());
            }
            header.insert(n.clone(), serde_json::json!({"dtype": "F32", "shape": [v.len()], "data_offsets": [start, body.len()]}));
        }
        let h = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h);
        f.extend(body);
        std::fs::write(path, f).unwrap();
    }

    #[test]
    fn a_checkpoint_directory_loads_like_the_in_memory_model() {
        let dir = std::env::temp_dir().join(format!("audio_whisper_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let c = tiny_cfg();
        let json = format!(
            r#"{{"num_mel_bins":{},"d_model":{},"encoder_layers":{},"decoder_layers":{},"encoder_attention_heads":{},"decoder_attention_heads":{},"encoder_ffn_dim":{},"decoder_ffn_dim":{},"vocab_size":{},"max_source_positions":{},"max_target_positions":{}}}"#,
            c.n_mels,
            c.d_model,
            c.enc_layers,
            c.dec_layers,
            c.enc_heads,
            c.dec_heads,
            c.enc_ffn,
            c.dec_ffn,
            c.vocab,
            c.max_source,
            c.max_target
        );
        std::fs::write(dir.join("config.json"), json).unwrap();
        // The tensors are stored with their 2-D / 3-D shapes flattened to one axis; the loader
        // only reads values, so only the element counts matter here.
        write_safetensors(
            &dir.join("model.safetensors"),
            &tiny_tensors(&c, "model.", 21, false),
        );
        let disk = Whisper::open_dir(&dir).unwrap();
        let mem = tiny_model(21);
        let mel = tiny_mel(1200.0);
        assert_eq!(disk.encode(&mel).unwrap(), mem.encode(&mel).unwrap());
        assert!(Whisper::open_dir(&dir.join("missing")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
