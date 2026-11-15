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
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };
    if let Some(c) = chosen {
        push(c.to_path_buf());
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
    let text = fs::read_to_string(dir.join(MANIFEST)).ok()?;
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
    })
}

/// The models found in `roots`, complete ones first, then by name.
pub fn scan(roots: &[PathBuf]) -> Vec<InstalledModel> {
    let mut out: Vec<InstalledModel> = Vec::new();
    for root in roots {
        let Ok(rd) = fs::read_dir(root) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(m) = read_one(&path) {
                    if !out.iter().any(|o| o.dir == m.dir) {
                        out.push(m);
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| (!a.complete, &a.id).cmp(&(!b.complete, &b.id)));
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
