//! Where this app keeps its own files (settings, conversations, logs).
//! Written by hand from the platform conventions to avoid a dependency.

use std::env;
use std::path::PathBuf;

pub fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        env::var_os("USERPROFILE").map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        env::var_os("HOME").map(PathBuf::from)
    }
}

/// The app's data folder. `COLIBRI_COMPANION_HOME` overrides it (used by tests).
pub fn data_dir() -> PathBuf {
    if let Some(explicit) = env::var_os("COLIBRI_COMPANION_HOME") {
        return PathBuf::from(explicit);
    }
    #[cfg(windows)]
    {
        let base = env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .or_else(|| home_dir().map(|h| h.join("AppData").join("Local")))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("ColibriCompanion")
    }
    #[cfg(target_os = "macos")]
    {
        home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("Library")
            .join("Application Support")
            .join("ColibriCompanion")
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        let base = env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| home_dir().map(|h| h.join(".local").join("share")))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("colibri-companion")
    }
}

pub fn settings_file() -> PathBuf {
    data_dir().join("settings.json")
}

pub fn mcp_file() -> PathBuf {
    data_dir().join("mcp.json")
}

pub fn documents_file() -> PathBuf {
    data_dir().join("documents.json")
}

pub fn conversations_dir() -> PathBuf {
    data_dir().join("conversations")
}

pub fn logs_dir() -> PathBuf {
    data_dir().join("logs")
}

/// Where exported files go: the Downloads folder when there is one, else the app's data folder.
pub fn downloads_dir() -> PathBuf {
    home_dir()
        .map(|h| h.join("Downloads"))
        .filter(|d| d.is_dir())
        .unwrap_or_else(data_dir)
}
