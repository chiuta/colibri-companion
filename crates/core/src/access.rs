//! Options for people with disabilities: how big and spaced things are, colours that do not rely
//! on telling red from green, reading answers aloud with the computer's own voice, and bundles of
//! settings ("presets") for the most common needs.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Child, Command, Stdio};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Targets {
    #[default]
    Normal,
    Large,
    Huge,
}

impl Targets {
    /// Minimum height of a button or field, in points.
    pub fn height(self) -> f32 {
        match self {
            Targets::Normal => 34.0,
            Targets::Large => 48.0,
            Targets::Huge => 64.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Access {
    /// Blue/orange instead of green/red for good/bad, readable with the common kinds of colour blindness.
    pub colour_blind: bool,
    pub targets: Targets,
    /// No fades or slides, no spinning wheels.
    pub reduce_motion: bool,
    /// A thick outline around whatever the keyboard is on.
    pub strong_focus: bool,
    /// 1.0 = normal; more space between lines and between controls.
    pub spacing: f32,
    /// A typeface designed for people with low vision (Atkinson Hyperlegible).
    pub readable_font: bool,
    pub read_aloud_auto: bool,
    /// Words per minute.
    pub speech_rate: u32,
    /// Flash the taskbar when an answer is finished and the window is in the background.
    pub flash_when_done: bool,
    /// Keep the interface to the essentials: hide the advanced windows from the menu.
    pub simple_menus: bool,
}

impl Default for Access {
    fn default() -> Self {
        Self {
            colour_blind: false,
            targets: Targets::Normal,
            reduce_motion: false,
            strong_focus: true,
            spacing: 1.0,
            readable_font: false,
            read_aloud_auto: false,
            speech_rate: 170,
            flash_when_done: false,
            simple_menus: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    LowVision,
    ScreenReader,
    Motor,
    Dyslexia,
    Cognitive,
    ColourBlind,
    Deaf,
}

impl Preset {
    pub const ALL: [Preset; 7] = [
        Preset::LowVision,
        Preset::ScreenReader,
        Preset::Motor,
        Preset::Dyslexia,
        Preset::Cognitive,
        Preset::ColourBlind,
        Preset::Deaf,
    ];
}

impl Access {
    /// Settings that go with a preset, and the text scale and theme to use with them.
    pub fn preset(p: Preset) -> (Access, f32, Option<crate::Theme>) {
        let base = Access::default();
        match p {
            Preset::LowVision => (
                Access {
                    readable_font: true,
                    targets: Targets::Large,
                    spacing: 1.3,
                    read_aloud_auto: false,
                    ..base
                },
                1.7,
                Some(crate::Theme::HighContrast),
            ),
            Preset::ScreenReader => (
                Access {
                    read_aloud_auto: true,
                    reduce_motion: true,
                    simple_menus: true,
                    ..base
                },
                1.1,
                None,
            ),
            Preset::Motor => (
                Access {
                    targets: Targets::Huge,
                    spacing: 1.3,
                    ..base
                },
                1.2,
                None,
            ),
            Preset::Dyslexia => (
                Access {
                    readable_font: true,
                    spacing: 1.5,
                    reduce_motion: true,
                    ..base
                },
                1.3,
                None,
            ),
            Preset::Cognitive => (
                Access {
                    reduce_motion: true,
                    simple_menus: true,
                    spacing: 1.3,
                    targets: Targets::Large,
                    ..base
                },
                1.3,
                None,
            ),
            Preset::ColourBlind => (
                Access {
                    colour_blind: true,
                    ..base
                },
                1.1,
                None,
            ),
            Preset::Deaf => (
                Access {
                    flash_when_done: true,
                    ..base
                },
                1.1,
                None,
            ),
        }
    }
}

// ------------------------------------------------------------------ contrast

fn lin(c: u8) -> f64 {
    let c = f64::from(c) / 255.0;
    if c <= 0.03928 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn luminance(c: [u8; 3]) -> f64 {
    0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])
}

/// WCAG contrast ratio between two colours, from 1 (same) to 21 (black on white).
pub fn contrast(a: [u8; 3], b: [u8; 3]) -> f64 {
    let (x, y) = (luminance(a), luminance(b));
    let (hi, lo) = if x > y { (x, y) } else { (y, x) };
    (hi + 0.05) / (lo + 0.05)
}

// ------------------------------------------------------------------ reading aloud

/// What to run to speak a text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Speech {
    pub program: String,
    pub args: Vec<String>,
    /// Text sent on standard input, so it can never be taken for an option.
    pub stdin: Option<String>,
    pub env: Vec<(String, String)>,
    /// Command that silences a speech the program handed to a background service.
    pub stop: Option<(String, Vec<String>)>,
}

/// The text without what is not meant to be heard: code blocks, link addresses, markdown marks.
pub fn clean_for_speech(markdown: &str, code_note: &str) -> String {
    let mut out = String::new();
    let mut in_code = false;
    for line in markdown.lines() {
        let t = line.trim_start();
        if t.starts_with("```") {
            if !in_code {
                out.push_str(code_note);
                out.push_str(".\n");
            }
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        let mut l = t.trim_start_matches('#').trim_start();
        l = l.trim_start_matches(['>', '-', '*', '+']).trim_start();
        let mut s = String::with_capacity(l.len());
        let mut rest = l;
        while let Some(i) = rest.find("](") {
            // [text](url) -> text
            if let Some(open) = rest[..i].rfind('[') {
                if let Some(close) = rest[i..].find(')') {
                    s.push_str(&rest[..open]);
                    s.push_str(&rest[open + 1..i]);
                    rest = &rest[i + close + 1..];
                    continue;
                }
            }
            break;
        }
        s.push_str(rest);
        let s: String = s
            .chars()
            .filter(|c| !matches!(c, '*' | '_' | '`' | '|' | '~'))
            .collect();
        let s = s
            .split_whitespace()
            .filter(|w| !w.starts_with("http://") && !w.starts_with("https://"))
            .collect::<Vec<_>>()
            .join(" ");
        if !s.is_empty() {
            out.push_str(&s);
            out.push('\n');
        }
    }
    out.trim().to_string()
}

fn lang_code(lang: crate::Lang) -> &'static str {
    use crate::Lang::*;
    match lang {
        En => "en",
        Ro => "ro",
        Fr => "fr",
        It => "it",
        Es => "es",
        Pt => "pt",
        De => "de",
        Hu => "hu",
    }
}

/// How to speak `text` on this system, or `None` when the computer has no speech program.
/// `os` is `std::env::consts::OS`; `have` says whether a program exists.
pub fn speech_plan(
    text: &str,
    lang: crate::Lang,
    wpm: u32,
    os: &str,
    have: &dyn Fn(&str) -> bool,
) -> Option<Speech> {
    let wpm = wpm.clamp(80, 400);
    let code = lang_code(lang);
    match os {
        "macos" => Some(Speech {
            program: "say".into(),
            args: vec!["-r".into(), wpm.to_string(), "-f".into(), "-".into()],
            stdin: Some(text.to_string()),
            env: vec![],
            stop: None,
        }),
        "windows" => {
            let script = format!(
                "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
                 try {{ $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet, [System.Speech.Synthesis.VoiceAge]::NotSet, 0, [System.Globalization.CultureInfo]::new('{code}')) }} catch {{}}; \
                 $s.Rate = {}; $s.Speak($env:CC_SPEAK)",
                ((i64::from(wpm) - 170) / 20).clamp(-10, 10)
            );
            Some(Speech {
                program: "powershell".into(),
                args: vec!["-NoProfile".into(), "-Command".into(), script],
                stdin: None,
                env: vec![("CC_SPEAK".into(), text.to_string())],
                stop: None,
            })
        }
        _ => {
            for p in ["espeak-ng", "espeak"] {
                if have(p) {
                    return Some(Speech {
                        program: p.into(),
                        args: vec![
                            "--stdin".into(),
                            "-v".into(),
                            code.into(),
                            "-s".into(),
                            wpm.to_string(),
                        ],
                        stdin: Some(text.to_string()),
                        env: vec![],
                        stop: None,
                    });
                }
            }
            if have("spd-say") {
                return Some(Speech {
                    program: "spd-say".into(),
                    args: vec![
                        "-w".into(),
                        "-l".into(),
                        code.into(),
                        "-r".into(),
                        (((i64::from(wpm) - 170) * 100 / 170).clamp(-100, 100)).to_string(),
                        "-e".into(),
                    ],
                    stdin: Some(text.to_string()),
                    env: vec![],
                    stop: Some(("spd-say".into(), vec!["-S".into()])),
                });
            }
            None
        }
    }
}

/// Starts the speech. The caller keeps the child to stop it later.
pub fn start(s: &Speech) -> std::io::Result<Child> {
    let mut cmd = Command::new(&s.program);
    cmd.args(&s.args)
        .stdin(if s.stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    for (k, v) in &s.env {
        cmd.env(k, v);
    }
    crate::proc::hide_window(&mut cmd);
    let mut child = cmd.spawn()?;
    if let (Some(text), Some(mut stdin)) = (s.stdin.clone(), child.stdin.take()) {
        // A separate thread: a long text must not freeze the window while the program reads it.
        std::thread::spawn(move || {
            let _ = stdin.write_all(text.as_bytes());
        });
    }
    Ok(child)
}

pub fn stop(child: &mut Child, s: Option<&Speech>) {
    let _ = child.kill();
    let _ = child.wait();
    if let Some((p, a)) = s.and_then(|s| s.stop.clone()) {
        let _ = Command::new(p)
            .args(a)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lang;

    #[test]
    fn contrast_matches_the_known_extremes() {
        assert!((contrast([0, 0, 0], [255, 255, 255]) - 21.0).abs() < 0.01);
        assert!((contrast([120, 120, 120], [120, 120, 120]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn markdown_is_made_listenable() {
        let md = "# Titlu\n**Bold** and `x` see [the docs](https://a.b/c) now https://x.y\n```rust\nfn main() {}\n```\n- item one\n> quote";
        let t = clean_for_speech(md, "Code skipped");
        assert_eq!(
            t,
            "Titlu\nBold and x see the docs now\nCode skipped.\nitem one\nquote"
        );
        assert!(!t.contains("fn main"));
    }

    #[test]
    fn speech_plans_per_system_and_text_never_becomes_an_option() {
        let none = |_: &str| false;
        assert!(speech_plan("x", Lang::Ro, 170, "linux", &none).is_none());
        let only = |p: &str| p == "espeak-ng";
        let p = speech_plan("-rm -rf", Lang::Ro, 200, "linux", &only).unwrap();
        assert_eq!(p.program, "espeak-ng");
        assert!(p.args.windows(2).any(|w| w == ["-v", "ro"]));
        assert!(!p.args.iter().any(|a| a.contains("rm")));
        assert_eq!(p.stdin.as_deref(), Some("-rm -rf"));
        let spd = |p: &str| p == "spd-say";
        let p = speech_plan("t", Lang::Hu, 170, "linux", &spd).unwrap();
        assert!(p.stop.is_some());
        let m = speech_plan("t", Lang::En, 170, "macos", &none).unwrap();
        assert_eq!(m.program, "say");
        let w = speech_plan("it's \"quoted\"", Lang::Fr, 170, "windows", &none).unwrap();
        // the text travels in the environment, not inside the script
        assert!(!w.args.iter().any(|a| a.contains("quoted")));
        assert_eq!(w.env[0].1, "it's \"quoted\"");
        assert!(w.args[2].contains("'fr'"));
        // the rate is kept in range
        let r = speech_plan("t", Lang::En, 9999, "linux", &only).unwrap();
        assert!(r.args.contains(&"400".to_string()));
    }

    #[test]
    fn presets_change_what_they_promise() {
        let (a, scale, th) = Access::preset(Preset::LowVision);
        assert!(scale >= 1.5 && a.readable_font && th == Some(crate::Theme::HighContrast));
        assert_eq!(Access::preset(Preset::Motor).0.targets, Targets::Huge);
        assert!(Access::preset(Preset::ColourBlind).0.colour_blind);
        assert!(Access::preset(Preset::Deaf).0.flash_when_done);
        assert!(Access::preset(Preset::Cognitive).0.simple_menus);
        assert!(Targets::Huge.height() > Targets::Large.height());
    }
}
