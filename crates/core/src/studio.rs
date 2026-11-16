//! Open models that make pictures, video, speech, music or understand sound, checked online in
//! November 2026. This is a map, not a promise: colibri itself makes pictures only with
//! Qwen-Image-2.1; everything else runs in a separate program (ComfyUI, LocalAI, a Python tool)
//! that this program can be pointed at. Licences and sizes come from the models' own pages and
//! from write-ups that sometimes disagree; where they did, the entry says "check".

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality {
    Image,
    Video,
    Speak,
    Listen,
    Music,
}

impl Modality {
    pub const ALL: [Modality; 5] = [
        Modality::Image,
        Modality::Video,
        Modality::Speak,
        Modality::Listen,
        Modality::Music,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Commercial {
    Yes,
    No,
    /// Terms with conditions, or sources that disagree.
    Check,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Where {
    /// colibri serves it.
    Colibri,
    /// Needs a separate program; this one can be pointed at it.
    Separate,
}

#[derive(Clone, Copy, Debug)]
pub struct Model {
    pub modality: Modality,
    pub name: &'static str,
    pub maker: &'static str,
    pub licence: &'static str,
    pub commercial: Commercial,
    /// What it asks of the computer, as reported.
    pub needs: &'static str,
    pub en: &'static str,
    pub ro: &'static str,
    pub run: Where,
}

macro_rules! m {
    ($mo:ident, $n:expr, $mk:expr, $l:expr, $c:ident, $nd:expr, $w:ident, $en:expr, $ro:expr) => {
        Model {
            modality: Modality::$mo,
            name: $n,
            maker: $mk,
            licence: $l,
            commercial: Commercial::$c,
            needs: $nd,
            en: $en,
            ro: $ro,
            run: Where::$w,
        }
    };
}

pub const ALL: &[Model] = &[
    // ---- pictures
    m!(
        Image,
        "Qwen-Image-2.1",
        "Alibaba (Qwen)",
        "Qwen Research (non-commercial)",
        No,
        "see the model sheet",
        Colibri,
        "Top of the open-weight text-to-image arena in October 2026; one model for making and editing pictures, strong at readable text. The only picture model colibri serves.",
        "Primul în arena deschisă de text-la-imagine în octombrie 2026; un singur model pentru creat și editat, bun la text lizibil. Singurul model de imagini servit de colibri."
    ),
    m!(
        Image,
        "HiDream-O1-Image",
        "HiDream.ai",
        "MIT",
        Yes,
        "high-end GPU",
        Separate,
        "Permissive licence with generation, editing and personalisation in one model.",
        "Licență permisivă, cu generare, editare și personalizare într-un singur model."
    ),
    m!(
        Image,
        "FLUX.2 [dev] / [klein]",
        "Black Forest Labs",
        "dev: non-commercial; klein 4B: Apache 2.0",
        Check,
        "klein runs on consumer GPUs",
        Separate,
        "[dev] is the 32B flagship; the small [klein] models make a picture in under half a second on a consumer GPU. Pictures from the non-commercial model may still be usable commercially, per one source: check.",
        "[dev] e modelul-etalon de 32B; micile [klein] fac o imagine în sub o jumătate de secundă pe un GPU obișnuit. Imaginile din modelul necomercial pot fi totuși folosite comercial, după o sursă: verifică."
    ),
    m!(
        Image,
        "Qwen-Image 2.0 / 2512",
        "Alibaba (Qwen)",
        "Apache 2.0",
        Yes,
        "single GPU",
        Separate,
        "Best at readable text in pictures, infographics and posters; version 2.0 is 7B and makes 2K images.",
        "Cel mai bun la text lizibil în imagini, infografice și afișe; versiunea 2.0 are 7B și face imagini 2K."
    ),
    m!(
        Image,
        "HunyuanImage 3.0",
        "Tencent",
        "Tencent Hunyuan Community",
        Check,
        "data-centre GPUs (80B)",
        Separate,
        "Photorealism and cinematic looks; the largest open image model.",
        "Fotorealism și aspect cinematografic; cel mai mare model deschis de imagini."
    ),
    m!(
        Image,
        "Z-Image (Turbo)",
        "Tongyi / community",
        "Apache 2.0",
        Yes,
        "modest GPU",
        Separate,
        "Fast, permissively licensed.",
        "Rapid, cu licență permisivă."
    ),
    m!(
        Image,
        "Stable Diffusion 3.5 / SDXL",
        "Stability AI",
        "SDXL: OpenRAIL++",
        Check,
        "SDXL: 8–12 GB graphics memory",
        Separate,
        "The biggest ecosystem of styles, add-ons and tutorials, and the lightest to run.",
        "Cel mai mare ecosistem de stiluri, extensii și tutoriale, și cel mai ușor de rulat."
    ),
    m!(
        Image,
        "Sana",
        "NVIDIA",
        "see the model page",
        Check,
        "an 8 GB card",
        Separate,
        "Sharp pictures on small graphics cards.",
        "Imagini clare pe plăci grafice mici."
    ),
    m!(
        Image,
        "Ideogram 4.0 (open)",
        "Ideogram",
        "Ideogram Open Model (non-commercial per one source)",
        Check,
        "high-end GPU",
        Separate,
        "Typography, posters and branding.",
        "Tipografie, afișe și branding."
    ),
    // ---- video
    m!(
        Video,
        "Wan 2.2 / 2.7",
        "Alibaba (Tongyi)",
        "Apache 2.0",
        Yes,
        "5B: about 23 GB on an RTX 4090; A14B: 41–60 GB",
        Separate,
        "The best all-round open video model in several 2026 comparisons, with no licence limits; text-to-video and picture-to-video. About nine minutes for five seconds on a big card.",
        "Cel mai bun model video deschis, în mai multe comparații din 2026, fără limite de licență; text-la-video și imagine-la-video. Cam nouă minute pentru cinci secunde pe o placă mare."
    ),
    m!(
        Video,
        "LTX-2.3 / 2.5",
        "Lightricks",
        "Free; paid tier above $10M revenue",
        Check,
        "from 8 GB (small), 32 GB for the big FP8 setup",
        Separate,
        "Makes video and its sound together in one pass, up to 4K; fast and flexible to adapt.",
        "Face video și sunetul lui într-o singură trecere, până la 4K; rapid și ușor de adaptat."
    ),
    m!(
        Video,
        "HunyuanVideo 1.5",
        "Tencent",
        "Apache 2.0 per one source; excludes EU/UK/South Korea per another",
        Check,
        "24 GB for full quality",
        Separate,
        "Cinematic motion; one render of about 75 seconds on a single RTX 4090 reported.",
        "Mișcare cinematografică; se raportează un randat de circa 75 de secunde pe un singur RTX 4090."
    ),
    m!(
        Video,
        "CogVideoX / Mochi 1",
        "Zhipu / Genmo",
        "see each page",
        Check,
        "16 GB+",
        Separate,
        "Older but still capable open video models.",
        "Modele video deschise mai vechi, dar încă capabile."
    ),
    // ---- speak
    m!(
        Speak,
        "Kokoro-82M",
        "hexgrad",
        "Apache 2.0",
        Yes,
        "2–3 GB, or just the processor",
        Separate,
        "Tiny, fast, natural narration with preset voices; cannot copy a voice.",
        "Mic, rapid, narare naturală cu voci prestabilite; nu poate copia o voce."
    ),
    m!(
        Speak,
        "Chatterbox / Turbo",
        "Resemble AI",
        "MIT",
        Yes,
        "a GPU helps",
        Separate,
        "Best-sounding local voice with cloning from a few seconds of audio; adds an inaudible watermark.",
        "Cea mai bună voce locală, cu clonare dintr-o mostră de câteva secunde; adaugă un filigran inaudibil."
    ),
    m!(
        Speak,
        "CosyVoice 2",
        "Alibaba (FunAudioLLM)",
        "Apache 2.0",
        Yes,
        "GPU",
        Separate,
        "Multilingual, streaming, with voice cloning.",
        "Multilingv, în flux, cu clonare de voce."
    ),
    m!(
        Speak,
        "Dia",
        "Nari Labs",
        "Apache 2.0",
        Yes,
        "GPU",
        Separate,
        "Dialogue between several speakers, with laughs and pauses.",
        "Dialog între mai mulți vorbitori, cu râsete și pauze."
    ),
    m!(
        Speak,
        "Voxtral TTS",
        "Mistral",
        "see the model page",
        Check,
        "GPU",
        Separate,
        "Expressive speech in about nine languages (released March 2026).",
        "Voce expresivă în circa nouă limbi (lansat în martie 2026)."
    ),
    m!(
        Speak,
        "Piper",
        "Open Home Foundation",
        "GPL-3.0 (current fork)",
        Check,
        "a Raspberry Pi is enough",
        Separate,
        "Light voices for small devices and home announcements; the original MIT repository was archived in October 2025.",
        "Voci ușoare pentru dispozitive mici și anunțuri în casă; depozitul MIT original a fost arhivat în octombrie 2025."
    ),
    // ---- listen
    m!(
        Listen,
        "Whisper large-v3 / turbo",
        "OpenAI",
        "MIT",
        Yes,
        "turbo is the fast one",
        Separate,
        "99 languages, punctuation included; the most widely used, now behind newer models on accuracy.",
        "99 de limbi, cu punctuație; cel mai folosit, dar depășit la acuratețe de modele mai noi."
    ),
    m!(
        Listen,
        "Parakeet TDT 0.6B v3",
        "NVIDIA",
        "CC-BY-4.0",
        Yes,
        "runs fast on a plain processor",
        Separate,
        "25 languages; beats Whisper on accuracy at a quarter of the size and far faster.",
        "25 de limbi; mai exact decât Whisper la un sfert din mărime și mult mai rapid."
    ),
    m!(
        Listen,
        "Canary-Qwen 2.5B",
        "NVIDIA",
        "CC-BY-4.0",
        Yes,
        "GPU",
        Separate,
        "High accuracy, with a mode that can also answer questions about the audio.",
        "Acuratețe mare, cu un mod care poate răspunde și la întrebări despre sunet."
    ),
    m!(
        Listen,
        "Qwen3-ASR 1.7B",
        "Alibaba (Qwen)",
        "see the model page",
        Check,
        "modest GPU",
        Separate,
        "Lowest average error among the open models listed in October 2026; very strong on Chinese.",
        "Cea mai mică eroare medie între modelele deschise din octombrie 2026; foarte bun la chineză."
    ),
    m!(
        Listen,
        "Voxtral Small / Mini",
        "Mistral",
        "Apache 2.0",
        Yes,
        "Small: 24B; Mini: 4B",
        Separate,
        "Best open model on the conversational-speech board in October 2026 (Small).",
        "Cel mai bun model deschis la vorbire conversațională în octombrie 2026 (Small)."
    ),
    m!(
        Listen,
        "Moonshine",
        "Useful Sensors",
        "see the model page",
        Check,
        "tiny; phones and boards",
        Separate,
        "Very small and fast, for devices with little power.",
        "Foarte mic și rapid, pentru dispozitive cu putere mică."
    ),
    // ---- music
    m!(
        Music,
        "ACE-Step 1.5 (XL)",
        "ACE Studio and StepFun",
        "MIT",
        Yes,
        "under 4 GB; Mac, NVIDIA, AMD, Intel",
        Separate,
        "The most practical open song model: full tracks with sung lyrics in seconds, with editing tools. The only open music family that cleared every test in one October 2026 review.",
        "Cel mai practic model deschis de cântece: piese întregi cu versuri cântate în câteva secunde, cu unelte de editare. Singura familie deschisă care a trecut toate testele într-o analiză din octombrie 2026."
    ),
    m!(
        Music,
        "YuE / YuE2",
        "HKUST and partners",
        "YuE v1: Apache 2.0",
        Check,
        "a big NVIDIA GPU",
        Separate,
        "Lyrics to full songs with vocals; heavy to run.",
        "Din versuri în cântece întregi cu voce; greu de rulat."
    ),
    m!(
        Music,
        "HeartMuLa",
        "HeartMuLa project",
        "Apache 2.0",
        Yes,
        "GPU",
        Separate,
        "Multilingual local music model; promising but still rough for finished songs.",
        "Model muzical local multilingv; promițător, dar încă brut pentru cântece gata făcute."
    ),
    m!(
        Music,
        "DiffRhythm 2",
        "open project",
        "Apache 2.0",
        Yes,
        "GPU",
        Separate,
        "Sings lyrics; usable commercially with credit.",
        "Cântă versuri; utilizabil comercial, cu menționarea sursei."
    ),
    m!(
        Music,
        "Stable Audio Open / 3",
        "Stability AI",
        "see the model page",
        Check,
        "runs on a typical computer (3 Small)",
        Separate,
        "Samples, loops and sound effects up to about 47 seconds; not for sung songs.",
        "Mostre, bucle și efecte sonore până la circa 47 de secunde; nu pentru cântece cu voce."
    ),
];

/// Programs that run these models and that this program can talk to.
#[derive(Clone, Copy, Debug)]
pub struct Runner {
    pub name: &'static str,
    pub url: &'static str,
    pub en: &'static str,
    pub ro: &'static str,
}

pub const RUNNERS: &[Runner] = &[
    Runner {
        name: "colibri",
        url: "https://github.com/JustVugg/colibri",
        en: "Serves Qwen-Image-2.1 at /v1/images/generations next to the chat models.",
        ro: "Servește Qwen-Image-2.1 la /v1/images/generations, lângă modelele de chat.",
    },
    Runner {
        name: "ComfyUI",
        url: "https://github.com/comfyanonymous/ComfyUI",
        en: "Node-based workflows for pictures, video and audio, with a local REST API on port 8188; wrappers add an OpenAI-style images endpoint.",
        ro: "Fluxuri cu noduri pentru imagini, video și sunet, cu API REST local pe portul 8188; unele adaosuri oferă un punct final de imagini în stil OpenAI.",
    },
    Runner {
        name: "LocalAI",
        url: "https://localai.io",
        en: "A drop-in OpenAI-style server with text, speech, pictures and embeddings; can run without a graphics card.",
        ro: "Server în stil OpenAI cu text, voce, imagini și embeddings; poate rula fără placă grafică.",
    },
    Runner {
        name: "Comfy MCP / comfyui-mcp",
        url: "https://modelcontextprotocol.io",
        en: "Lets an AI assistant drive ComfyUI through the Model Context Protocol (planned here).",
        ro: "Lasă un asistent AI să controleze ComfyUI prin Model Context Protocol (planificat aici).",
    },
];

pub fn of(m: Modality) -> impl Iterator<Item = &'static Model> {
    ALL.iter().filter(move |x| x.modality == m)
}

// ------------------------------------------------------------------ making a picture

/// Decodes standard base64, ignoring whitespace; `None` for anything that is not base64.
pub fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// The body of an OpenAI-style picture request (what `coli serve` and most local servers take).
pub fn image_request(prompt: &str, size: &str, model: &str) -> String {
    serde_json::json!({
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": "b64_json",
    })
    .to_string()
}

/// The picture bytes in the server's answer, or a reason in plain words.
pub fn image_from_response(body: &str) -> Result<Vec<u8>, String> {
    let v: serde_json::Value = serde_json::from_str(body)
        .map_err(|_| "the server did not answer with JSON".to_string())?;
    if let Some(e) = v.pointer("/error/message").and_then(|m| m.as_str()) {
        return Err(e.to_string());
    }
    let b64 = v
        .pointer("/data/0/b64_json")
        .and_then(|b| b.as_str())
        .ok_or_else(|| "the answer holds no picture".to_string())?;
    let bytes = base64_decode(b64).ok_or_else(|| "the picture is not valid base64".to_string())?;
    let known = bytes.starts_with(&[0x89, b'P', b'N', b'G'])
        || bytes.starts_with(&[0xFF, 0xD8])
        || bytes.starts_with(b"RIFF");
    if !known {
        return Err("the answer is not a PNG, JPEG or WebP picture".into());
    }
    Ok(bytes)
}

pub fn extension_of(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        "jpg"
    } else if bytes.starts_with(b"RIFF") {
        "webp"
    } else {
        "png"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_is_complete_and_honest() {
        for m in Modality::ALL {
            assert!(of(m).count() >= 4, "{m:?}");
        }
        let mut names: Vec<&str> = ALL.iter().map(|x| x.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), ALL.len());
        for x in ALL {
            assert!(
                !x.en.is_empty()
                    && !x.ro.is_empty()
                    && !x.licence.is_empty()
                    && !x.needs.is_empty(),
                "{}",
                x.name
            );
        }
        // Only the picture model colibri really serves is marked as served by it.
        let c: Vec<&str> = ALL
            .iter()
            .filter(|x| x.run == Where::Colibri)
            .map(|x| x.name)
            .collect();
        assert_eq!(c, ["Qwen-Image-2.1"]);
        // A non-commercial licence is never shown as free for business use.
        assert!(!ALL.iter().any(|x| {
            x.licence.to_lowercase().contains("non-commercial") && x.commercial == Commercial::Yes
        }));
        assert!(RUNNERS.len() >= 4);
    }

    #[test]
    fn base64_decodes_and_pictures_are_checked() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVs\nbG8=").unwrap(), b"hello");
        assert!(base64_decode("not base64!").is_none());
        // 8 bytes of a PNG signature
        let sig = "iVBORw0KGgo=";
        let ok = format!(r#"{{"data":[{{"b64_json":"{sig}"}}]}}"#);
        let png = image_from_response(&ok).unwrap();
        assert_eq!(extension_of(&png), "png");
        assert!(image_from_response(r#"{"data":[{"b64_json":"aGVsbG8="}]}"#).is_err());
        assert_eq!(
            image_from_response(r#"{"error":{"message":"no such model"}}"#).unwrap_err(),
            "no such model"
        );
        assert!(image_from_response("<html>").is_err());
        assert!(image_from_response(r#"{"data":[]}"#).is_err());
        let b = image_request("a cat", "1024x1024", "m");
        assert!(b.contains("b64_json") && b.contains("a cat"));
    }
}
