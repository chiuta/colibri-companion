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

/// Can this folder be trusted to mean the same place whatever the working folder is? A relative
/// entry (an empty `PATH` item, `.`, `bin`) is read from the working folder by some shells and
/// loaders, so a program of the same name planted in a folder the person merely opened would be
/// found and run. Only absolute folders are searched.
fn searchable(dir: &Path) -> bool {
    dir.is_absolute()
}

/// A file that can be started: on Unix it has an execute bit (a data file of the same name
/// earlier in the list must not hide the real program later in it).
fn startable(p: &Path) -> bool {
    if !p.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(p).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Looks for the program in `extra` (for example the folder the user chose), then in the folder
/// of this program, then in every folder of `path`. Only the file system is read, and only
/// absolute folders are looked in (see [`searchable`]).
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
        .filter(|d| searchable(d))
        .flat_map(|d| names.iter().map(move |n| d.join(n)))
        .find(|c| startable(c))
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

/// Compares two file names the way a person reads them: digit runs by value and letters without
/// regard to case, so `p2` comes before `p10` (reading order of a scanned book).
pub fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let (mut a, mut b) = (a.chars().peekable(), b.chars().peekable());
    loop {
        match (a.peek().copied(), b.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let run = |it: &mut std::iter::Peekable<std::str::Chars<'_>>| {
                    let mut d = String::new();
                    while let Some(c) = it.next_if(char::is_ascii_digit) {
                        d.push(c);
                    }
                    d.trim_start_matches('0').to_string()
                };
                let (da, db) = (run(&mut a), run(&mut b));
                // more digits (after the zeros) is a bigger number; the same length compares as text
                let o = da.len().cmp(&db.len()).then_with(|| da.cmp(&db));
                if o != Ordering::Equal {
                    return o;
                }
            }
            (Some(x), Some(y)) => {
                let o = x.to_lowercase().cmp(y.to_lowercase());
                if o != Ordering::Equal {
                    return o;
                }
                a.next();
                b.next();
            }
        }
    }
}

/// Most directory entries looked at in one folder: a folder with millions of files cannot make
/// the search use unbounded time or memory.
const MAX_ENTRIES: usize = 200_000;

/// The page files of a scan, in reading order, and whether there were more than [`MAX_PAGES`]
/// (the list is then cut). A file gives itself; a folder gives the images and PDFs directly
/// inside it, sub-folders are not entered.
pub fn scan_pages(path: &Path) -> (Vec<PathBuf>, bool) {
    if path.is_file() {
        let v = if is_scan_file(path) {
            vec![path.to_path_buf()]
        } else {
            Vec::new()
        };
        return (v, false);
    }
    let Ok(rd) = fs::read_dir(path) else {
        return (Vec::new(), false);
    };
    let mut v: Vec<PathBuf> = rd
        .take(MAX_ENTRIES)
        .flatten()
        .map(|e| e.path())
        .filter(|p| is_scan_file(p) && p.is_file())
        .collect();
    v.sort_by(|a, b| {
        let name = |p: &PathBuf| p.file_name().map(|n| n.to_string_lossy().into_owned());
        natural_cmp(&name(a).unwrap_or_default(), &name(b).unwrap_or_default())
            .then_with(|| a.cmp(b))
    });
    let more = v.len() > MAX_PAGES;
    v.truncate(MAX_PAGES);
    (v, more)
}

/// [`scan_pages`] without the "there were more" flag.
pub fn collect_inputs(path: &Path) -> Vec<PathBuf> {
    scan_pages(path).0
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

/// On Windows `canonicalize` gives `\\?\C:\dir\f.png` (or `\\?\UNC\host\share\f.png`), a form
/// that many programs and libraries cannot open. This turns it back into `C:\dir\f.png` (or
/// `\\host\share\f.png`); anything else is returned as it is.
pub fn plain_windows_path(p: &Path) -> PathBuf {
    let Some(s) = p.to_str() else {
        return p.to_path_buf();
    };
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    match s.strip_prefix(r"\\?\") {
        // only a drive path (`C:\...`): other verbatim forms (devices, `Volume{..}`) stay verbatim
        Some(rest)
            if rest.len() >= 3
                && rest.as_bytes()[1] == b':'
                && rest.as_bytes()[0].is_ascii_alphabetic() =>
        {
            PathBuf::from(rest)
        }
        _ => p.to_path_buf(),
    }
}

/// An absolute form of `p` that does not need the file to exist and never starts with `-`.
fn absolute(p: &Path) -> PathBuf {
    let abs = fs::canonicalize(p)
        .or_else(|_| std::path::absolute(p))
        .unwrap_or_else(|_| p.to_path_buf());
    let abs = plain_windows_path(&abs);
    if abs.is_absolute() {
        abs
    } else {
        Path::new(".").join(abs)
    }
}

/// The command line, as separate arguments (no shell). Every path is made absolute so a name
/// that starts with `-` can never be taken for an option (not even as the value of `--txt`).
pub fn build_command(bin: &Path, job: &Job) -> Command {
    let mut cmd = Command::new(bin);
    let mut args: Vec<OsString> = Vec::new();
    for i in &job.inputs {
        args.push(absolute(i).into_os_string());
    }
    args.push("--ocr".into());
    args.push("--lang".into());
    args.push(job.lang.clone().into());
    args.push("--txt".into());
    args.push(absolute(&job.txt).into_os_string());
    args.push("--format".into());
    args.push("none".into());
    args.push("--out-dir".into());
    args.push(absolute(&job.out_dir).into_os_string());
    if job.lang.split('+').any(|l| l == "ron") {
        // comma-below ș ț, words split at a line end joined
        args.push("--clean-text".into());
    }
    cmd.args(args);
    proc::hide_window(&mut cmd);
    cmd
}

/// Last few non-empty lines of what the program printed on its error stream. Only the end of the
/// stream is looked at: a program that prints tens of megabytes must not cost that much again.
fn tail_lines(stderr: &[u8]) -> String {
    const LOOK: usize = 8 * 1024;
    let end = &stderr[stderr.len().saturating_sub(LOOK)..];
    let s = String::from_utf8_lossy(end);
    let lines: Vec<&str> = s.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let from = lines.len().saturating_sub(4);
    lines[from..].join("\n").chars().take(600).collect()
}

/// The text a program wrote, as a string: UTF-8 (a byte-order mark is dropped) or UTF-16 with a
/// byte-order mark (what some Windows programs write); anything else invalid is replaced, never
/// an error. NUL characters are dropped.
fn decode_text(bytes: &[u8]) -> String {
    let utf16 = |b: &[u8], le: bool| {
        let units = b.chunks_exact(2).map(|c| {
            if le {
                u16::from_le_bytes([c[0], c[1]])
            } else {
                u16::from_be_bytes([c[0], c[1]])
            }
        });
        char::decode_utf16(units)
            .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
            .collect::<String>()
    };
    let text = match bytes {
        [0xFF, 0xFE, rest @ ..] => utf16(rest, true),
        [0xFE, 0xFF, rest @ ..] => utf16(rest, false),
        [0xEF, 0xBB, 0xBF, rest @ ..] => String::from_utf8_lossy(rest).into_owned(),
        _ => String::from_utf8_lossy(bytes).into_owned(),
    };
    text.replace('\0', "")
}

fn read_capped(path: &Path) -> Option<String> {
    let f = fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    f.take(MAX_TEXT_BYTES).read_to_end(&mut buf).ok()?;
    let mut text = decode_text(&buf);
    // a cut in the middle of a character leaves one replacement mark at the very end
    if buf.len() as u64 >= MAX_TEXT_BYTES {
        while text.ends_with(char::REPLACEMENT_CHARACTER) {
            text.pop();
        }
    }
    Some(text)
}

/// Runs the program on `scan` (a file or a folder) and returns the recognised text. `work` is a
/// folder for the program's output; it is created, and the text file of an earlier run is removed.
/// Two runs at the same time need two different `work` folders.
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
    let (inputs, too_many) = scan_pages(scan);
    if inputs.is_empty() {
        return Err(OcrError::NoInput);
    }
    if too_many {
        // never read the first part of a book and call it the book
        return Err(OcrError::Run(format!(
            "more than {MAX_PAGES} pages in the folder: split it into smaller folders"
        )));
    }
    let work = absolute(work);
    let work = work.as_path();
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
    // a name that is not valid UTF-8 keeps its readable part instead of becoming a bare "scan"
    let base = scan
        .file_name()
        .map_or_else(|| "scan".into(), |n| n.to_string_lossy().into_owned())
        .chars()
        .filter(|c| !c.is_control())
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
        // reading order: page 2 comes before page 10
        assert_eq!(names, ["P1.jpg", "p2.png", "p10.png"]);
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
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // without the execute bit it is only a file, and is not offered
            assert_eq!(locate_in(&[], Some(d.as_os_str())), None);
            fs::set_permissions(&exe, fs::Permissions::from_mode(0o755)).unwrap();
        }
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

    #[test]
    fn page_names_sort_the_way_a_person_reads_them() {
        use std::cmp::Ordering::{Equal, Greater, Less};
        assert_eq!(natural_cmp("p2.png", "p10.png"), Less);
        assert_eq!(natural_cmp("page_9.jpg", "page_10.jpg"), Less);
        assert_eq!(natural_cmp("P1.png", "p1.png"), Equal);
        assert_eq!(natural_cmp("a.png", "B.png"), Less);
        assert_eq!(natural_cmp("scan007.png", "scan7.png"), Equal);
        assert_eq!(natural_cmp("scan08.png", "scan7.png"), Greater);
        assert_eq!(natural_cmp("", "a"), Less);
        // numbers of any length compare without overflowing
        assert_eq!(
            natural_cmp(
                &format!("x{}", "9".repeat(60)),
                &format!("x1{}", "0".repeat(60))
            ),
            Less
        );
        let d = temp("order");
        for n in 1..=12 {
            fs::write(d.join(format!("scan_{n}.png")), b"x").unwrap();
        }
        let names: Vec<String> = collect_inputs(&d)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        let expected: Vec<String> = (1..=12).map(|n| format!("scan_{n}.png")).collect();
        assert_eq!(names, expected);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    #[cfg(unix)]
    fn a_folder_with_more_pages_than_the_limit_is_refused_not_silently_cut() {
        let d = temp("toomany");
        for n in 0..=MAX_PAGES {
            fs::write(d.join(format!("p{n:05}.png")), b"x").unwrap();
        }
        let (pages, more) = scan_pages(&d);
        assert_eq!((pages.len(), more), (MAX_PAGES, true));
        // the program is never started: the answer would be the first part of the book
        let r = read_scan(
            Path::new("/nonexistent/legatorocr"),
            &d,
            "eng",
            &d.join("w"),
            Duration::from_secs(5),
        );
        assert!(
            matches!(&r, Err(OcrError::Run(m)) if m.contains("split")),
            "{r:?}"
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    #[cfg(unix)]
    fn a_program_in_a_relative_search_folder_or_without_the_execute_bit_is_never_found() {
        use std::os::unix::fs::PermissionsExt;
        let d = temp("plant");
        let cwd = std::env::current_dir().unwrap();
        // the same folder, written relative to the working folder: `../../tmp/cc-lgr-plant-..`
        let up = "../".repeat(cwd.components().count());
        let rel = PathBuf::from(format!("{up}{}", d.strip_prefix("/").unwrap().display()));
        assert!(!rel.is_absolute());
        let planted = fake_bin(&d, FAKE_OK);
        assert!(
            rel.join(PROGRAM).is_file(),
            "the relative form must reach the file"
        );
        for hostile in [
            std::env::join_paths([PathBuf::new()]).unwrap(), // an empty entry means "here"
            std::env::join_paths([Path::new(".")]).unwrap(),
            std::env::join_paths([rel.clone()]).unwrap(),
        ] {
            assert_eq!(locate_in(&[], Some(&hostile)), None, "{hostile:?}");
        }
        assert_eq!(locate_in(&[rel], None), None);
        // the absolute form of the same folder is found
        assert_eq!(
            locate_in(std::slice::from_ref(&d), None),
            Some(planted.clone())
        );
        // a data file of the same name earlier in the list does not hide the real program
        let data = temp("plant-data");
        fs::write(data.join(PROGRAM), b"not a program").unwrap();
        fs::set_permissions(data.join(PROGRAM), fs::Permissions::from_mode(0o644)).unwrap();
        let both = std::env::join_paths([&data, &d]).unwrap();
        assert_eq!(locate_in(&[], Some(&both)), Some(planted));
        let _ = fs::remove_dir_all(&d);
        let _ = fs::remove_dir_all(&data);
    }

    #[test]
    fn windows_verbatim_paths_become_plain_ones() {
        let c = |s: &str| {
            plain_windows_path(Path::new(s))
                .to_string_lossy()
                .into_owned()
        };
        assert_eq!(
            c(r"\\?\C:\Users\Ana\scan 1.png"),
            r"C:\Users\Ana\scan 1.png"
        );
        assert_eq!(c(r"\\?\UNC\server\share\a.png"), r"\\server\share\a.png");
        // already plain, and forms that are not drive paths, stay as they are
        assert_eq!(c(r"C:\a.png"), r"C:\a.png");
        assert_eq!(c("/home/ana/a.png"), "/home/ana/a.png");
        assert_eq!(c(r"\\?\Volume{1234}\a.png"), r"\\?\Volume{1234}\a.png");
        assert_eq!(c(r"\\?\C:"), r"\\?\C:");
    }

    #[test]
    fn every_path_in_the_command_is_absolute_even_the_output_ones() {
        let d = temp("abs");
        let scan = d.join("a.png");
        fs::write(&scan, b"x").unwrap();
        let job = Job {
            inputs: vec![scan],
            lang: "eng".into(),
            txt: PathBuf::from("-out/scan.txt"),
            out_dir: PathBuf::from("-out/pages"),
        };
        let args: Vec<String> = build_command(Path::new("legatorocr"), &job)
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        for (i, a) in args.iter().enumerate() {
            if i == 0 || matches!(args[i - 1].as_str(), "--txt" | "--out-dir") {
                assert!(Path::new(a).is_absolute(), "{a} in {args:?}");
                assert!(!a.starts_with('-'), "{a}");
            }
        }
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_noisy_error_stream_costs_next_to_nothing_and_keeps_the_end() {
        let mut big = String::new();
        for i in 0..2_000_000 {
            big.push_str(&format!("line {i}\n"));
        }
        let t = std::time::Instant::now();
        let r = tail_lines(big.as_bytes());
        assert!(
            t.elapsed() < Duration::from_millis(500),
            "{:?}",
            t.elapsed()
        );
        assert_eq!(r, "line 1999996\nline 1999997\nline 1999998\nline 1999999");
        assert_eq!(tail_lines(b""), "");
        assert_eq!(
            tail_lines(&[0xFF, 0xFE, b'\n', b'o', b'k']),
            "\u{FFFD}\u{FFFD}\nok"
        );
    }

    #[test]
    fn text_in_other_encodings_is_read_not_lost() {
        assert_eq!(decode_text(b"\xEF\xBB\xBFHello \xC8\x99"), "Hello \u{219}");
        let mut le = vec![0xFF, 0xFE];
        le.extend("Ana are mere\n".encode_utf16().flat_map(u16::to_le_bytes));
        assert_eq!(decode_text(&le), "Ana are mere\n");
        let mut be = vec![0xFE, 0xFF];
        be.extend("ț".encode_utf16().flat_map(u16::to_be_bytes));
        assert_eq!(decode_text(&be), "ț");
        assert_eq!(decode_text(b"a\xFFb\0c"), "a\u{FFFD}bc");
        assert_eq!(decode_text(b""), "");
    }

    #[test]
    #[cfg(unix)]
    fn a_scan_whose_name_is_not_utf8_keeps_its_readable_name() {
        use std::os::unix::ffi::OsStrExt;
        let name = std::ffi::OsStr::from_bytes(b"carte-\xFF-1.pdf");
        let n = doc_name(Path::new("/x").join(name).as_path());
        assert!(
            n.starts_with("carte-") && n.ends_with("-1.pdf (OCR)"),
            "{n}"
        );
        assert_eq!(doc_name(Path::new("a\nb.png")), "ab.png (OCR)");
        assert_eq!(doc_name(Path::new("/")), "scan (OCR)");
    }

    #[test]
    #[cfg(unix)]
    fn a_program_that_prints_a_flood_still_gives_its_text() {
        let d = temp("flood");
        let scan = d.join("a.png");
        fs::write(&scan, b"x").unwrap();
        let body = format!(
            "{}\nhead -c 30000000 /dev/zero | tr '\\0' 'x'; head -c 30000000 /dev/zero | tr '\\0' 'e' 1>&2",
            FAKE_OK
        );
        let bin = fake_bin(&d, &body);
        let t = std::time::Instant::now();
        let r = read_scan(&bin, &scan, "eng", &d.join("w"), Duration::from_secs(60)).unwrap();
        assert!(r.contains("harbour"));
        assert!(t.elapsed() < Duration::from_secs(30));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    #[cfg(unix)]
    fn a_text_file_in_utf16_is_read() {
        let d = temp("utf16");
        let scan = d.join("a.png");
        fs::write(&scan, b"x").unwrap();
        // the program writes UTF-16 (with a byte-order mark) to the --txt file
        let bin = fake_bin(
            &d,
            r#"while [ $# -gt 0 ]; do if [ "$1" = "--txt" ]; then shift; out="$1"; fi; shift; done
printf '\377\376H\000i\000 \000t\000h\000e\000r\000e\000' > "$out""#,
        );
        assert_eq!(
            read_scan(&bin, &scan, "eng", &d.join("w"), Duration::from_secs(10)).unwrap(),
            "Hi there"
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    #[cfg(unix)]
    fn a_timeout_or_a_stop_leaves_neither_the_program_nor_its_helpers_running() {
        fn alive(pid: i32) -> bool {
            // a zombie that nobody has collected yet is not running
            std::fs::read_to_string(format!("/proc/{pid}/stat"))
                .map(|s| !s.rsplit(") ").next().unwrap_or("").starts_with('Z'))
                .unwrap_or(false)
        }
        for stop_by in ["timeout", "cancel"] {
            let d = temp(&format!("zombie-{stop_by}"));
            let scan = d.join("a.png");
            fs::write(&scan, b"x").unwrap();
            let pidfile = d.join("helper.pid");
            let bin = fake_bin(
                &d,
                &format!("sleep 60 &\necho $! > '{}'\nwait", pidfile.display()),
            );
            let cancel = AtomicBool::new(false);
            let r = std::thread::scope(|sc| {
                let h = sc.spawn(|| {
                    read_scan_cancel(
                        &bin,
                        &scan,
                        "eng",
                        &d.join("w"),
                        Duration::from_millis(if stop_by == "timeout" { 1500 } else { 60_000 }),
                        &cancel,
                    )
                });
                // wait until the helper exists, then (for the cancel case) press Stop
                let t = std::time::Instant::now();
                while !pidfile.exists() && t.elapsed() < Duration::from_secs(10) {
                    std::thread::sleep(Duration::from_millis(20));
                }
                if stop_by == "cancel" {
                    cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                h.join().unwrap()
            });
            match stop_by {
                "timeout" => assert!(matches!(r, Err(OcrError::Run(_))), "{r:?}"),
                _ => assert_eq!(r, Err(OcrError::Cancelled)),
            }
            let pid: i32 = fs::read_to_string(&pidfile)
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            let t = std::time::Instant::now();
            while alive(pid) && t.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(50));
            }
            assert!(
                !alive(pid),
                "{stop_by}: the helper process {pid} is still running"
            );
            let _ = fs::remove_dir_all(&d);
        }
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
