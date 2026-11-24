//! Audio front-end demo: synthesise a chirp with a pause, round-trip it through a 48 kHz WAV,
//! resample to 16 kHz, print the log-mel peak track and the voice-activity spans. With
//! `--model DIR` (a Hugging Face Whisper checkpoint in safetensors form) it also transcribes the
//! audio and prints token ids; nothing here has been run against real Whisper weights.
//!
//! `cargo run -p companion-engine --example audio_demo -- [--model DIR] [file.wav]`
use companion_engine::audio::*;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_dir = args.iter().position(|a| a == "--model").and_then(|i| args.get(i + 1)).cloned();
    let wav_path = args.iter().enumerate().find(|(i, a)| a.ends_with(".wav") && (*i == 0 || args[*i - 1] != "--model")).map(|(_, a)| a.clone());

    let bytes = match wav_path {
        Some(p) => std::fs::read(&p).unwrap_or_else(|e| { eprintln!("{p}: {e}"); std::process::exit(1) }),
        None => {
            let sr = 48_000u32;
            let mut s = vec![0.0f32; sr as usize / 2];
            s.extend((0..sr as usize * 2).map(|i| { let t = i as f32 / sr as f32; 0.4 * (2.0 * std::f32::consts::PI * (200.0 * t + 950.0 * t * t)).sin() }));
            s.extend(vec![0.0; sr as usize / 2]);
            write_wav(&Wav { sample_rate: sr, channels: 1, samples: s }, SampleFormat::Pcm16).expect("write")
        }
    };
    let wav = read_wav(&bytes).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1) });
    println!("WAV: {} Hz, {} channel(s), {:.2} s", wav.sample_rate, wav.channels, wav.duration_s());
    let audio = resample(&wav.to_mono(), wav.sample_rate, 16_000).expect("resample");
    println!("16 kHz mono: {} samples", audio.len());

    let mel = MelExtractor::new(MelConfig::default()).expect("mel");
    let spec = mel.log_mel(&audio, true).expect("log-mel");
    println!("log-mel: {} bins x {} frames ({} from real audio)", spec.n_mels, spec.frames, spec.valid_frames);
    for t in (0..spec.valid_frames).step_by((spec.valid_frames / 8).max(1)) {
        let best = (0..spec.n_mels).max_by(|&a, &b| spec.at(a, t).total_cmp(&spec.at(b, t))).unwrap_or(0);
        println!("  t = {:5.2} s  loudest mel bin {:2} (about {:6.0} Hz)", t as f32 * 0.01, best, mel.center_hz()[best]);
    }
    let spans = detect_speech(&audio, &VadConfig::default()).expect("vad");
    for s in &spans { println!("speech: {:.2} s to {:.2} s", s.start as f32 / 16_000.0, s.end as f32 / 16_000.0); }
    if spans.is_empty() { println!("speech: none found"); }

    if let Some(dir) = model_dir {
        let model = Whisper::open_dir(std::path::Path::new(&dir)).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1) });
        let c = model.config();
        println!("model: d_model {}, {} + {} layers, vocab {}", c.d_model, c.enc_layers, c.dec_layers, c.vocab);
        // Multilingual Whisper ids (start-of-transcript, English, transcribe, no timestamps, end);
        // they are specific to those vocabularies, so pass your own for other checkpoints.
        let opts = TranscribeOptions { decode: DecodeOptions::new(vec![50258, 50259, 50359, 50363], 50257), vad: None };
        let mel = MelExtractor::new(MelConfig { n_mels: c.n_mels, ..MelConfig::default() }).expect("mel");
        match model.transcribe(&audio, &mel, &opts) {
            Ok(t) => for s in t.segments { println!("[{:.2} - {:.2}] token ids {:?}", s.start_s, s.end_s, s.tokens); },
            Err(e) => eprintln!("transcription failed: {e}"),
        }
    }
}
