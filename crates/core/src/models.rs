//! The models already on this computer. colibri keeps each downloaded model in its own
//! folder (`<models root>/<catalog id>`) with a `.colibri-download.json` manifest that
//! says whether every file arrived. Reading that manifest is all that is needed to list
//! them, so no colibri code is involved: switching goes through `coli setup --model`.

use crate::paths::home_dir;
use serde::Deserialize;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const MANIFEST: &str = ".colibri-download.json";

#[derive(Clone, Debug, PartialEq)]
pub struct InstalledModel {
    /// The catalog id (the folder's name).
    pub id: String,
    pub dir: PathBuf,
    /// Every file arrived.
    pub complete: bool,
    /// Size the model has when complete, in bytes.
    pub total_bytes: u64,
    /// Bytes on disk now (whole files plus `.part` files).
    pub present_bytes: u64,
    /// A model folder that colibri did not download (copied by hand, converted from a GGUF,
    /// downloaded by another program). It is started from its folder and never deleted here.
    pub local: bool,
}

impl InstalledModel {
    pub fn fraction(&self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.present_bytes as f64 / self.total_bytes as f64).clamp(0.0, 1.0) as f32
        }
    }
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct Manifest {
    complete: bool,
    files: Vec<ManifestFile>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct ManifestFile {
    path: String,
    size: u64,
}

/// colibri's default folder for downloaded models.
pub fn default_models_root() -> Option<PathBuf> {
    home_dir().map(|h| h.join("colibri-models"))
}

/// A disk the person can put models on.
#[derive(Clone, Debug, PartialEq)]
pub struct Drive {
    pub mount: PathBuf,
    pub free_gb: f64,
    pub total_gb: f64,
}

impl Drive {
    /// The folder colibri would use on this disk.
    pub fn models_folder(&self) -> PathBuf {
        self.mount.join("colibri-models")
    }
}

/// Real disks big enough to matter, biggest free space first.
pub fn drives() -> Vec<Drive> {
    let Ok(disks) = sysinfo::Disks::new_with_refreshed_list() else {
        return Vec::new();
    };
    const GB: f64 = 1_073_741_824.0;
    let mut out: Vec<Drive> = Vec::new();
    for d in disks.iter() {
        let total = d.total_space() as f64 / GB;
        let mount = d.mount_point().to_path_buf();
        let m = mount.to_string_lossy();
        let system_bits = [
            "/boot",
            "/snap",
            "/run",
            "/sys",
            "/proc",
            "/dev",
            "/var/lib/docker",
        ]
        .iter()
        .any(|x| m.starts_with(x));
        if total < 8.0 || system_bits || out.iter().any(|o| o.mount == mount) {
            continue;
        }
        out.push(Drive {
            mount,
            free_gb: d.available_space() as f64 / GB,
            total_gb: total,
        });
    }
    out.sort_by(|a, b| b.free_gb.total_cmp(&a.free_gb));
    out
}

/// A `models` folder next to the running program ("portable" install).
pub fn beside_app() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()?
        .parent()
        .map(|d| d.join("colibri-models"))
}

/// Free space in GB on the disk holding `path` (or its nearest existing parent).
pub fn free_gb_at(path: &Path) -> Option<f64> {
    let mut p = path;
    while !p.exists() {
        p = p.parent()?;
    }
    crate::resources::disk_space(p).0
}

/// Can files be created in `dir` (made if missing)? Checked by really trying.
pub fn writable(dir: &Path) -> bool {
    if fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(format!(".write-test-{}", std::process::id()));
    let ok = fs::write(&probe, b"x").is_ok();
    let _ = fs::remove_file(&probe);
    ok
}

/// Every folder worth looking in, without duplicates: the one chosen in Settings, the one
/// the active model lives in, and colibri's default.
pub fn roots(chosen: Option<&Path>, active_model_dir: Option<&Path>) -> Vec<PathBuf> {
    roots_with(chosen, active_model_dir, &[])
}

/// [`roots`] plus the folders the person added by hand ("Load models from a folder").
pub fn roots_with(
    chosen: Option<&Path>,
    active_model_dir: Option<&Path>,
    extra: &[PathBuf],
) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };
    if let Some(c) = chosen {
        push(c.to_path_buf());
    }
    for e in extra {
        push(e.clone());
    }
    if let Some(parent) = active_model_dir.and_then(Path::parent) {
        push(parent.to_path_buf());
    }
    if let Some(d) = default_models_root() {
        push(d);
    }
    if let Some(d) = beside_app().filter(|d| d.is_dir()) {
        push(d);
    }
    // On Windows colibri puts models on another drive when the home drive is short of room.
    #[cfg(windows)]
    for letter in 'C'..='Z' {
        let p = PathBuf::from(format!("{letter}:\\colibri-models"));
        if p.is_dir() {
            push(p);
        }
    }
    out
}

fn read_one(dir: &Path) -> Option<InstalledModel> {
    let Ok(text) = fs::read_to_string(dir.join(MANIFEST)) else {
        return read_local(dir);
    };
    let m: Manifest = serde_json::from_str(&text).ok()?;
    let id = dir.file_name()?.to_string_lossy().into_owned();
    let mut total = 0u64;
    let mut present = 0u64;
    for f in &m.files {
        if f.path.contains("..") || Path::new(&f.path).is_absolute() {
            continue; // never follow a path out of the model folder
        }
        total += f.size;
        let p = dir.join(&f.path);
        if let Ok(meta) = fs::metadata(&p) {
            if meta.len() >= f.size {
                present += f.size;
                continue;
            }
        }
        let part = dir.join(format!("{}.part", f.path));
        if let Ok(meta) = fs::metadata(&part) {
            present += meta.len().min(f.size);
        }
    }
    Some(InstalledModel {
        id,
        dir: dir.to_path_buf(),
        complete: m.complete && present >= total,
        total_bytes: total,
        present_bytes: present,
        local: false,
    })
}

const WEIGHT_EXT: [&str; 6] = ["safetensors", "gguf", "bin", "npz", "pt", "pth"];
/// Files smaller than this are configuration, tokenizers or tests, not weights.
const MIN_WEIGHT_BYTES: u64 = 1 << 20;
/// Stop looking after this many directory entries in one folder tree (a huge or network disk).
const MAX_ENTRIES: usize = 20_000;

fn ext_of(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

/// A folder holding a model that colibri did not download: a `config.json` and weight files
/// (up to two levels down). Unfinished copies (`.part`, `.tmp`, `.crdownload`) make it not complete.
fn read_local(dir: &Path) -> Option<InstalledModel> {
    if !dir.join("config.json").is_file() {
        return None;
    }
    let mut weights = 0u64;
    let mut unfinished = false;
    let mut seen = 0usize;
    let mut stack = vec![(dir.to_path_buf(), 0u8)];
    while let Some((d, depth)) = stack.pop() {
        let Ok(rd) = fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            seen += 1;
            if seen > MAX_ENTRIES {
                break;
            }
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_symlink() {
                continue; // never follow a link out of the folder (or into a loop)
            }
            let path = e.path();
            if ft.is_dir() {
                if depth < 2 {
                    stack.push((path, depth + 1));
                }
            } else if ft.is_file() {
                let ext = ext_of(&path);
                if ["part", "tmp", "crdownload", "incomplete", "download"].contains(&ext.as_str()) {
                    unfinished = true;
                } else if WEIGHT_EXT.contains(&ext.as_str()) {
                    let n = e.metadata().map(|m| m.len()).unwrap_or(0);
                    if n >= MIN_WEIGHT_BYTES {
                        weights += n;
                    }
                }
            }
        }
    }
    if weights == 0 {
        return None;
    }
    Some(InstalledModel {
        id: dir.file_name()?.to_string_lossy().into_owned(),
        dir: dir.to_path_buf(),
        complete: !unfinished,
        total_bytes: weights,
        present_bytes: weights,
        local: true,
    })
}

/// The models found in `roots`, complete ones first, then by name.
pub fn scan(roots: &[PathBuf]) -> Vec<InstalledModel> {
    let mut out: Vec<InstalledModel> = Vec::new();
    let push = |m: InstalledModel, out: &mut Vec<InstalledModel>| {
        if !out.iter().any(|o| o.dir == m.dir) {
            out.push(m);
        }
    };
    for root in roots {
        // the folder itself may be a model (a folder dragged in from the file manager)
        if let Some(m) = read_local(root) {
            push(m, &mut out);
            continue;
        }
        let Ok(rd) = fs::read_dir(root) else { continue };
        for entry in rd.flatten().take(MAX_ENTRIES) {
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_symlink() || !ft.is_dir() {
                continue;
            }
            let path = entry.path();
            if let Some(m) = read_one(&path) {
                push(m, &mut out);
                continue;
            }
            // one level deeper: `models/gemma/gemma-3-12b`
            let Ok(sub) = fs::read_dir(&path) else {
                continue;
            };
            for e2 in sub.flatten().take(2_000) {
                let Ok(ft2) = e2.file_type() else { continue };
                if ft2.is_symlink() || !ft2.is_dir() {
                    continue;
                }
                if let Some(m) = read_one(&e2.path()) {
                    push(m, &mut out);
                }
            }
        }
    }
    out.sort_by(|a, b| (!a.complete, &a.id).cmp(&(!b.complete, &b.id)));
    out
}

/// A GGUF model file found on disk (the first part, when it is split), not yet converted.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufFile {
    /// What to give `coli import`: the file, or the first part of a split model.
    pub first: PathBuf,
    /// Shown name: the file name without `.gguf` and without the `-00001-of-0000N` part.
    pub name: String,
    pub parts: usize,
    /// Size of every part that is present.
    pub bytes: u64,
    /// Parts named by the first file but not beside it.
    pub missing: Vec<String>,
}

/// The part of a split model's name before `-00001-of-0000N` (the whole name when not split).
fn split_prefix(stem: &str) -> Option<&str> {
    let of = stem.rfind("-of-")?;
    let dash = stem[..of].rfind('-')?;
    let number = &stem[dash + 1..of];
    (!number.is_empty() && number.bytes().all(|b| b.is_ascii_digit())).then(|| &stem[..dash])
}

fn display_name(file: &str) -> String {
    let stem = file
        .strip_suffix(".gguf")
        .or_else(|| file.strip_suffix(".GGUF"))
        .unwrap_or(file);
    split_prefix(stem).unwrap_or(stem).to_string()
}

/// The GGUF models in `roots` and their sub-folders (to `depth` levels). A split model is listed
/// once, by its first part; vision projectors (`mmproj`) and files under 20 MB are not models.
/// Links are not followed and the search stops after [`MAX_ENTRIES`] entries per folder tree.
pub fn find_gguf(roots: &[PathBuf], depth: u8) -> Vec<GgufFile> {
    use crate::coli::{is_first_gguf, missing_gguf_parts};
    let mut out: Vec<GgufFile> = Vec::new();
    for root in roots {
        let mut seen = 0usize;
        let mut stack = vec![(root.clone(), 0u8)];
        while let Some((d, level)) = stack.pop() {
            let Ok(rd) = fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                seen += 1;
                if seen > MAX_ENTRIES {
                    break;
                }
                let Ok(ft) = e.file_type() else { continue };
                if ft.is_symlink() {
                    continue;
                }
                let path = e.path();
                if ft.is_dir() {
                    if level < depth && !path.join(MANIFEST).exists() {
                        stack.push((path, level + 1));
                    }
                    continue;
                }
                let file = e.file_name().to_string_lossy().into_owned();
                let lower = file.to_ascii_lowercase();
                if !lower.ends_with(".gguf") || lower.contains("mmproj") || !is_first_gguf(&path) {
                    continue;
                }
                let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                if size < 20 << 20 || out.iter().any(|o| o.first == path) {
                    continue;
                }
                let missing = missing_gguf_parts(&path);
                let name = display_name(&file);
                // every part that sits beside the first one
                let (mut parts, mut bytes) = (1usize, size);
                let stem = file.get(..file.len().saturating_sub(5)).unwrap_or(&file);
                if let (Some(prefix), Some(dir)) = (split_prefix(stem), path.parent()) {
                    if let Ok(rd2) = fs::read_dir(dir) {
                        for o in rd2.flatten() {
                            let n = o.file_name().to_string_lossy().into_owned();
                            let n_stem = n.get(..n.len().saturating_sub(5)).unwrap_or(&n);
                            if n != file
                                && n.to_ascii_lowercase().ends_with(".gguf")
                                && split_prefix(n_stem) == Some(prefix)
                            {
                                parts += 1;
                                bytes += o.metadata().map(|m| m.len()).unwrap_or(0);
                            }
                        }
                    }
                }
                out.push(GgufFile {
                    first: path,
                    name,
                    parts,
                    bytes,
                    missing,
                });
            }
        }
    }
    out.sort_by_key(|g| g.name.to_lowercase());
    out
}

/// Why a model folder may not be removed (when it may not).
#[derive(Debug, PartialEq, Eq)]
pub enum DeleteRefusal {
    NotAModel,
    OutsideRoots,
    InUse,
}

/// Removes one downloaded model. Only a folder that holds colibri's manifest, sits directly
/// inside one of `roots`, and is not `active_dir` is touched.
pub fn delete(dir: &Path, roots: &[PathBuf], active_dir: Option<&Path>) -> io::Result<()> {
    check_delete(dir, roots, active_dir)
        .map_err(|r| io::Error::new(io::ErrorKind::PermissionDenied, format!("{r:?}")))?;
    fs::remove_dir_all(dir)
}

pub fn check_delete(
    dir: &Path,
    roots: &[PathBuf],
    active_dir: Option<&Path>,
) -> Result<(), DeleteRefusal> {
    if !dir.join(MANIFEST).is_file() {
        return Err(DeleteRefusal::NotAModel);
    }
    let canon = |p: &Path| fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let dir_c = canon(dir);
    let parent = dir_c.parent().map(Path::to_path_buf);
    if !roots.iter().any(|r| Some(canon(r)) == parent) {
        return Err(DeleteRefusal::OutsideRoots);
    }
    if active_dir.is_some_and(|a| canon(a) == dir_c) {
        return Err(DeleteRefusal::InUse);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cc-models-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn model(root: &Path, id: &str, complete: bool, written: usize) {
        let dir = root.join(id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join(MANIFEST),
            format!(
                r#"{{"repo":"a/b","complete":{complete},"files":[{{"path":"w.bin","size":100}},{{"path":"sub/c.json","size":10}}]}}"#
            ),
        )
        .unwrap();
        if complete {
            fs::create_dir_all(dir.join("sub")).unwrap();
            fs::write(dir.join("w.bin"), vec![0u8; 100]).unwrap();
            fs::write(dir.join("sub/c.json"), vec![0u8; 10]).unwrap();
        } else {
            fs::write(dir.join("w.bin.part"), vec![0u8; written]).unwrap();
        }
    }

    #[test]
    fn lists_complete_and_partial_models() {
        let root = temp("scan");
        model(&root, "beta", true, 0);
        model(&root, "alpha", false, 40);
        fs::create_dir_all(root.join("not-a-model")).unwrap();
        let found = scan(std::slice::from_ref(&root));
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].id, "beta");
        assert!(found[0].complete && found[0].total_bytes == 110);
        assert_eq!(found[1].id, "alpha");
        assert!(!found[1].complete);
        assert_eq!(found[1].present_bytes, 40);
        assert!((found[1].fraction() - 40.0 / 110.0).abs() < 1e-6);
        // a manifest that says complete while a file is missing is not complete
        fs::remove_file(root.join("beta/w.bin")).unwrap();
        assert!(!scan(&[root])[0].complete);
    }

    #[test]
    fn models_not_downloaded_by_colibri_are_listed_too() {
        let root = temp("local");
        // a folder copied by hand: config.json + a big weight file
        let g = root.join("gemma-3-12b");
        fs::create_dir_all(g.join("shards")).unwrap();
        fs::write(g.join("config.json"), "{}").unwrap();
        fs::write(g.join("shards/model.safetensors"), vec![0u8; 2 << 20]).unwrap();
        // one level deeper
        let deep = root.join("collection/qwen");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("config.json"), "{}").unwrap();
        fs::write(deep.join("w.bin"), vec![0u8; 3 << 20]).unwrap();
        // an unfinished copy
        let half = root.join("half");
        fs::create_dir_all(&half).unwrap();
        fs::write(half.join("config.json"), "{}").unwrap();
        fs::write(half.join("model.gguf"), vec![0u8; 2 << 20]).unwrap();
        fs::write(half.join("model.gguf.part"), b"x").unwrap();
        // not models: no weights, no config, tiny weights
        for (n, files) in [
            ("docs", vec!["config.json", "notes.txt"]),
            ("noconfig", vec!["w.bin"]),
        ] {
            let d = root.join(n);
            fs::create_dir_all(&d).unwrap();
            for f in files {
                fs::write(
                    d.join(f),
                    vec![0u8; if f == "w.bin" { 2 << 20 } else { 10 }],
                )
                .unwrap();
            }
        }
        model(&root, "beta", true, 0);
        let found = scan(std::slice::from_ref(&root));
        let ids: Vec<(&str, bool, bool)> = found
            .iter()
            .map(|m| (m.id.as_str(), m.local, m.complete))
            .collect();
        assert!(ids.contains(&("beta", false, true)), "{ids:?}");
        assert!(ids.contains(&("gemma-3-12b", true, true)), "{ids:?}");
        assert!(ids.contains(&("qwen", true, true)), "{ids:?}");
        assert!(ids.contains(&("half", true, false)), "{ids:?}");
        assert_eq!(found.len(), 4, "{ids:?}");
        // a local model is never deletable from here
        assert_eq!(
            check_delete(&g, std::slice::from_ref(&root), None),
            Err(DeleteRefusal::NotAModel)
        );
        // the folder itself may be a model
        let me = scan(std::slice::from_ref(&g));
        assert_eq!(me.len(), 1);
        assert!(me[0].local && me[0].id == "gemma-3-12b");
    }

    #[test]
    fn gguf_files_are_found_once_per_model() {
        let root = temp("gguf");
        let d = root.join("a/b");
        fs::create_dir_all(&d).unwrap();
        let big = |n: &str, mb: usize| fs::write(d.join(n), vec![0u8; mb << 20]).unwrap();
        big("Gemma-3-12B-Q4_K_M.gguf", 21);
        big("GLM-Q8_0-00001-of-00003.gguf", 21);
        big("GLM-Q8_0-00002-of-00003.gguf", 21);
        big("mmproj-gemma.gguf", 21);
        big("tiny.gguf", 1);
        big("notes.txt", 1);
        let top = root.join("lone.GGUF");
        fs::write(&top, vec![0u8; 21 << 20]).unwrap();
        let f = find_gguf(std::slice::from_ref(&root), 3);
        let names: Vec<&str> = f.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
            names,
            ["Gemma-3-12B-Q4_K_M", "GLM-Q8_0", "lone"],
            "{names:?}"
        );
        let glm = f.iter().find(|g| g.name == "GLM-Q8_0").unwrap();
        assert_eq!(glm.parts, 2);
        assert_eq!(glm.bytes, 42 << 20);
        assert_eq!(glm.missing.len(), 1, "{:?}", glm.missing);
        // too shallow to reach them
        assert_eq!(find_gguf(std::slice::from_ref(&root), 0).len(), 1);
        // a folder that does not exist, and an empty list, are fine
        assert!(find_gguf(&[PathBuf::from("/no/such/folder")], 3).is_empty());
        assert!(find_gguf(&[], 3).is_empty());
    }

    #[test]
    fn extra_folders_join_the_roots() {
        let r = roots_with(None, None, &[PathBuf::from("/x"), PathBuf::from("/x")]);
        assert_eq!(
            r.iter().filter(|p| p.as_path() == Path::new("/x")).count(),
            1
        );
    }

    #[test]
    fn roots_are_deduplicated() {
        let r = roots(Some(Path::new("/m")), Some(Path::new("/m/qwen")));
        assert_eq!(
            r.iter().filter(|p| p.as_path() == Path::new("/m")).count(),
            1
        );
    }

    #[test]
    fn delete_is_careful() {
        let root = temp("del");
        model(&root, "a", true, 0);
        model(&root, "b", true, 0);
        let roots = vec![root.clone()];
        assert_eq!(
            delete(&root.join("a"), &roots, Some(&root.join("a")))
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
        assert!(root.join("a").exists());
        // a folder without the manifest is never removed
        fs::create_dir_all(root.join("docs")).unwrap();
        assert_eq!(
            check_delete(&root.join("docs"), &roots, None),
            Err(DeleteRefusal::NotAModel)
        );
        // nor one outside the roots
        let other = temp("del-other");
        model(&other, "x", true, 0);
        assert_eq!(
            check_delete(&other.join("x"), &roots, None),
            Err(DeleteRefusal::OutsideRoots)
        );
        // the root itself is not a model folder inside a root
        assert!(check_delete(&root, &roots, None).is_err());
        delete(&root.join("b"), &roots, Some(&root.join("a"))).unwrap();
        assert!(!root.join("b").exists() && root.join("a").exists());
    }
}
