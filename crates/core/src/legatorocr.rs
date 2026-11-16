//! Reading scanned pages with LegatorOCR, an independent program (MIT licence) that Colibri
//! Companion only ever *calls* as an external command, never links or copies.
//!
//! The idea is small on purpose: if a `legatorocr` program is already installed, run it on a scan
//! (one image, a scanned PDF or a folder of page images) with `--ocr --txt FILE`, read back the
//! plain text it wrote, and hand that text to the document library ([`crate::rag`]) so that the
//! local model can answer from the scan, with the passages cited by name like any other document.
//!
//! Nothing here touches the network and nothing is started unless the caller asks for it. The
//! recognised text is data, not instructions (the library already tells the model so). The text is
//! only as good as the OCR: it is never presented as a verified transcription.
//!
//! Status: used by the Documents window (`crates/app/src/ocr_ui.rs`); see `docs/legatorocr-integration.md`.

use crate::proc;
use crate::rag::{AddError, Library};
use crate::store::Lang;
use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// Name of the program, without a platform extension.
pub const PROGRAM: &str = "legatorocr";
/// Most page images taken from one folder (a run over more is split by the user).
pub const MAX_PAGES: usize = 2000;
/// Longest text read back from the program; the rest is cut (with the same cap the library has).
pub const MAX_TEXT_BYTES: u64 = 32 * 1024 * 1024;
/// How long one run may take before it is ended. OCR of a book takes minutes, not hours.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(3 * 60 * 60);

#[derive(Debug, PartialEq, Eq)]
pub enum OcrError {
    /// No `legatorocr` program was found.
    NotInstalled,
    /// Nothing to read: no file, or a folder without page images.
    NoInput,
    /// The program ran and failed (its last error lines are kept).
    Failed(String),
    /// The program could not be started, or was ended after the time limit.
    Run(String),
    /// The program finished but wrote no text (blank pages, or no OCR engine for the language).
    NoText,
    /// The user stopped the run.
    Cancelled,
}

/// Image and PDF kinds LegatorOCR reads as scans.
pub fn is_scan_file(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "pdf")
    )
}

/// Tesseract-style language code for an interface language (the codes `--lang` takes).
pub fn ocr_lang(lang: Lang) -> &'static str {
    match lang {
        Lang::Ro => "ron",
        Lang::En => "eng",
        Lang::Fr => "fra",
        Lang::It => "ita",
        Lang::Es => "spa",
        Lang::Pt => "por",
        Lang::De => "deu",
        Lang::Hu => "hun",
    }
}

fn exe_names() -> Vec<String> {
    if cfg!(windows) {
        vec![format!("{PROGRAM}.exe")]
    } else {
        vec![PROGRAM.to_string()]
    }
}

/// Looks for the program in `extra` (for example the folder the user chose), then in the folder
/// of this program, then in every folder of `path`. Only the file system is read.
pub fn locate_in(extra: &[PathBuf], path: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = extra.to_vec();
    if let Ok(me) = std::env::current_exe()
        && let Some(d) = me.parent()
    {
        dirs.push(d.to_path_buf());
    }
    if let Some(p) = path {
        dirs.extend(std::env::split_paths(p));
    }
    let names = exe_names();
    dirs.iter()
        .flat_map(|d| names.iter().map(move |n| d.join(n)))
        .find(|c| c.is_file())
}

/// [`locate_in`] with the real `PATH`.
pub fn locate(extra: &[PathBuf]) -> Option<PathBuf> {
    locate_in(extra, std::env::var_os("PATH").as_deref())
}

/// The version line of the program (`legatorocr 0.2.0-pre.65`), or `None` when it does not answer
/// like LegatorOCR. This is what tells a real install from an unrelated file of the same name.
pub fn version(bin: &Path) -> Option<String> {
    let mut cmd = Command::new(bin);
    cmd.arg("--version");
    proc::hide_window(&mut cmd);
    let out = proc::run_capture(cmd, Duration::from_secs(10)).ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    line.to_ascii_lowercase()
        .starts_with(PROGRAM)
        .then_some(line)
}

/// The page files of a scan: the file itself, or the images and PDFs directly inside a folder,
/// in name order (reading order). Sub-folders are not entered.
pub fn collect_inputs(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return if is_scan_file(path) {
            vec![path.to_path_buf()]
        } else {
            Vec::new()
        };
    }
    let Ok(rd) = fs::read_dir(path) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_scan_file(p))
        .collect();
    v.sort_by_key(|p| p.file_name().map(|n| n.to_ascii_lowercase()));
    v.truncate(MAX_PAGES);
    v
}

/// One OCR run.
#[derive(Debug, Clone)]
pub struct Job {
    pub inputs: Vec<PathBuf>,
    /// Tesseract-style language(s), e.g. `ron` or `ron+eng`.
    pub lang: String,
    /// The text file the program writes.
    pub txt: PathBuf,
    /// Where it may put page images (kept empty: `--format none`).
    pub out_dir: PathBuf,
}

/// A language string that is safe to pass: letters, `_`, `+`, `-` only, at most 40 characters.
pub fn clean_lang(s: &str) -> Option<String> {
    let ok = !s.is_empty()
        && s.len() <= 40
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '_' | '-'))
        && !s.starts_with('-');
    ok.then(|| s.to_string())
}

/// The command line, as separate arguments (no shell). Inputs are made absolute so a name that
/// starts with `-` can never be taken for an option.
pub fn build_command(bin: &Path, job: &Job) -> Command {
    let mut cmd = Command::new(bin);
    let mut args: Vec<OsString> = Vec::new();
    for i in &job.inputs {
        let abs = fs::canonicalize(i).unwrap_or_else(|_| i.clone());
        let abs = if abs.is_absolute() {
            abs
        } else {
            Path::new(".").join(abs)
        };
        args.push(abs.into_os_string());
    }
    args.push("--ocr".into());
    args.push("--lang".into());
    args.push(job.lang.clone().into());
    args.push("--txt".into());
    args.push(job.txt.clone().into());
    args.push("--format".into());
    args.push("none".into());
    args.push("--out-dir".into());
    args.push(job.out_dir.clone().into());
    if job.lang.split('+').any(|l| l == "ron") {
        // comma-below ș ț, words split at a line end joined
        args.push("--clean-text".into());
    }
    cmd.args(args);
    proc::hide_window(&mut cmd);
    cmd
}

/// Last few non-empty lines of what the program printed on its error stream.
fn tail_lines(stderr: &[u8]) -> String {
    let s = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = s.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let from = lines.len().saturating_sub(4);
    lines[from..].join("\n").chars().take(600).collect()
}

fn read_capped(path: &Path) -> Option<String> {
    let f = fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    f.take(MAX_TEXT_BYTES).read_to_end(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Runs the program on `scan` (a file or a folder) and returns the recognised text. `work` is a
/// folder for the program's output; it is created and emptied of what this run wrote.
pub fn read_scan(
    bin: &Path,
    scan: &Path,
    lang: &str,
    work: &Path,
    timeout: Duration,
) -> Result<String, OcrError> {
    read_scan_cancel(bin, scan, lang, work, timeout, &AtomicBool::new(false))
}

/// [`read_scan`] that ends the program as soon as `cancel` becomes true (`OcrError::Cancelled`).
pub fn read_scan_cancel(
    bin: &Path,
    scan: &Path,
    lang: &str,
    work: &Path,
    timeout: Duration,
    cancel: &AtomicBool,
) -> Result<String, OcrError> {
    let lang = clean_lang(lang).ok_or_else(|| OcrError::Run("bad language code".into()))?;
    let inputs = collect_inputs(scan);
    if inputs.is_empty() {
        return Err(OcrError::NoInput);
    }
    fs::create_dir_all(work).map_err(|e| OcrError::Run(e.to_string()))?;
    let txt = work.join("scan.txt");
    let _ = fs::remove_file(&txt);
    let job = Job {
        inputs,
        lang,
        txt: txt.clone(),
        out_dir: work.join("pages"),
    };
    let out = proc::run_capture_cancel(build_command(bin, &job), timeout, cancel).map_err(|e| {
        if e.kind() == std::io::ErrorKind::Interrupted {
            OcrError::Cancelled
        } else {
            OcrError::Run(e.to_string())
        }
    })?;
    if !out.status.success() {
        return Err(OcrError::Failed(tail_lines(&out.stderr)));
    }
    let text = read_capped(&txt).unwrap_or_default();
    if text.trim().is_empty() {
        return Err(OcrError::NoText);
    }
    Ok(text)
}

/// Name a scan gets in the library: its file or folder name, with a marker that the text is OCR.
pub fn doc_name(scan: &Path) -> String {
    let base = scan
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("scan")
        .chars()
        .take(80)
        .collect::<String>();
    format!("{base} (OCR)")
}

#[derive(Debug, PartialEq, Eq)]
pub enum IngestError {
    Ocr(OcrError),
    Library(AddError),
}

/// Reads a scan and adds its text to collection `id` of `lib`. Returns how many passages it made.
pub fn ingest_scan(
    lib: &mut Library,
    id: &str,
    bin: &Path,
    scan: &Path,
    lang: &str,
    work: &Path,
) -> Result<usize, IngestError> {
    let text = read_scan(bin, scan, lang, work, DEFAULT_TIMEOUT).map_err(IngestError::Ocr)?;
    lib.add_document(id, &doc_name(scan), &text)
        .map_err(IngestError::Library)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("cc-lgr-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// A stand-in for the real program: it understands `--version` and `--txt FILE`.
    #[cfg(unix)]
    fn fake_bin(dir: &Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join(PROGRAM);
        fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    const FAKE_OK: &str = r#"
if [ "$1" = "--version" ]; then echo "legatorocr 0.0-test"; exit 0; fi
while [ $# -gt 0 ]; do
  if [ "$1" = "--txt" ]; then shift; out="$1"; fi
  shift
done
printf 'The harbour office opened at nine.\n\nFees were paid in silver coins.\n' > "$out"
"#;

    #[test]
    fn language_codes_follow_the_interface_languages() {
        assert_eq!(ocr_lang(Lang::Ro), "ron");
        assert_eq!(ocr_lang(Lang::Hu), "hun");
        assert_eq!(ocr_lang(Lang::Pt), "por");
    }

    #[test]
    fn only_clean_language_strings_pass() {
        assert_eq!(clean_lang("ron+eng").as_deref(), Some("ron+eng"));
        assert!(clean_lang("").is_none());
        assert!(clean_lang("--version").is_none());
        assert!(clean_lang("eng; rm -rf").is_none());
        assert!(clean_lang(&"a".repeat(41)).is_none());
    }

    #[test]
    fn scans_are_recognised_by_extension() {
        assert!(is_scan_file(Path::new("a/b.TIFF")));
        assert!(is_scan_file(Path::new("book.pdf")));
        assert!(!is_scan_file(Path::new("notes.txt")));
        assert!(!is_scan_file(Path::new("noext")));
    }

    #[test]
    fn a_folder_gives_its_pages_in_name_order_and_skips_other_files() {
        let d = temp("folder");
        for n in ["p10.png", "p2.png", "P1.jpg", "readme.txt"] {
            fs::write(d.join(n), b"x").unwrap();
        }
        fs::create_dir_all(d.join("sub")).unwrap();
        fs::write(d.join("sub/deep.png"), b"x").unwrap();
        let names: Vec<String> = collect_inputs(&d)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["P1.jpg", "p10.png", "p2.png"]);
        assert!(collect_inputs(&d.join("readme.txt")).is_empty());
        assert!(collect_inputs(&d.join("missing")).is_empty());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn the_command_line_is_plain_arguments_with_absolute_inputs() {
        let d = temp("cmd");
        let scan = d.join("-rf.png");
        fs::write(&scan, b"x").unwrap();
        let job = Job {
            inputs: vec![scan],
            lang: "ron".into(),
            txt: d.join("t.txt"),
            out_dir: d.join("o"),
        };
        let cmd = build_command(Path::new("legatorocr"), &job);
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(Path::new(&args[0]).is_absolute(), "{args:?}");
        for needed in [
            "--ocr",
            "--lang",
            "ron",
            "--txt",
            "--format",
            "none",
            "--clean-text",
        ] {
            assert!(args.iter().any(|a| a == needed), "{needed} in {args:?}");
        }
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn locate_finds_an_executable_in_a_given_folder_and_nowhere_else() {
        let d = temp("locate");
        assert_eq!(locate_in(&[], Some(d.as_os_str())), None);
        let exe = d.join(exe_names().remove(0));
        fs::write(&exe, b"x").unwrap();
        assert_eq!(locate_in(&[], Some(d.as_os_str())), Some(exe.clone()));
        assert_eq!(locate_in(std::slice::from_ref(&d), None), Some(exe));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    #[cfg(unix)]
    fn version_tells_a_real_install_from_an_unrelated_file() {
        let d = temp("version");
        let bin = fake_bin(&d, FAKE_OK);
        assert_eq!(version(&bin).as_deref(), Some("legatorocr 0.0-test"));
        let other = fake_bin(&d, "echo hello");
        assert_eq!(version(&other), None);
        let failing = fake_bin(&d, "echo legatorocr x; exit 3");
        assert_eq!(version(&failing), None);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    #[cfg(unix)]
    fn a_scan_becomes_passages_the_library_can_search() {
        let d = temp("ingest");
        let bin = fake_bin(&d, FAKE_OK);
        let scan = d.join("ledger.png");
        fs::write(&scan, b"not really an image").unwrap();
        let mut lib = Library::default();
        let id = lib.create("Scans", 1).unwrap();
        let n = ingest_scan(&mut lib, &id, &bin, &scan, "eng", &d.join("work")).unwrap();
        assert!(n >= 1);
        let col = lib.get(&id).unwrap();
        assert_eq!(col.docs[0].name, "ledger.png (OCR)");
        let s = crate::rag::Searcher::new(col);
        let hits = s.search("silver coins fees", 3);
        assert!(!hits.is_empty());
        assert!(col.chunks[hits[0].chunk].text.contains("silver"));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    #[cfg(unix)]
    fn failures_are_reported_not_swallowed() {
        let d = temp("fail");
        let scan = d.join("a.png");
        fs::write(&scan, b"x").unwrap();
        let work = d.join("w");
        let bad = fake_bin(&d, "echo 'no OCR engine for hun' 1>&2; exit 2");
        assert_eq!(
            read_scan(&bad, &scan, "hun", &work, Duration::from_secs(10)),
            Err(OcrError::Failed("no OCR engine for hun".into()))
        );
        let empty = fake_bin(&d, "exit 0");
        assert_eq!(
            read_scan(&empty, &scan, "eng", &work, Duration::from_secs(10)),
            Err(OcrError::NoText)
        );
        assert_eq!(
            read_scan(
                &empty,
                &d.join("nothing.png"),
                "eng",
                &work,
                Duration::from_secs(10)
            ),
            Err(OcrError::NoInput)
        );
        assert!(matches!(
            read_scan(&empty, &scan, "--x", &work, Duration::from_secs(10)),
            Err(OcrError::Run(_))
        ));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    #[cfg(unix)]
    fn a_program_that_never_ends_is_stopped() {
        let d = temp("slow");
        let scan = d.join("a.png");
        fs::write(&scan, b"x").unwrap();
        let slow = fake_bin(&d, "sleep 30");
        let t = std::time::Instant::now();
        let r = read_scan(
            &slow,
            &scan,
            "eng",
            &d.join("w"),
            Duration::from_millis(500),
        );
        assert!(t.elapsed() < Duration::from_secs(10));
        assert!(r.is_err());
        let _ = fs::remove_dir_all(&d);
    }

    /// Only with a real install: `LEGATOROCR_BIN=/path/to/legatorocr cargo test -p companion-core
    /// real_legatorocr -- --nocapture`. Skipped (passes) otherwise.
    #[test]
    fn real_legatorocr_reads_a_generated_page_when_installed() {
        let Some(bin) = std::env::var_os("LEGATOROCR_BIN").map(PathBuf::from) else {
            return;
        };
        assert!(version(&bin).is_some());
        let Some(page) = std::env::var_os("LEGATOROCR_TEST_PAGE").map(PathBuf::from) else {
            return;
        };
        let d = temp("real");
        let text = read_scan(&bin, &page, "eng", &d, Duration::from_secs(300)).unwrap();
        println!("OCR text: {text}");
        assert!(!text.trim().is_empty());
        let mut lib = Library::default();
        let id = lib.create("Scans", 1).unwrap();
        ingest_scan(&mut lib, &id, &bin, &page, "eng", &d.join("again")).unwrap();
        let col = lib.get(&id).unwrap();
        let hits = crate::rag::Searcher::new(col).search("silver coins", 2);
        assert!(!hits.is_empty(), "the scanned text is searchable");
        let _ = fs::remove_dir_all(&d);
    }
}
