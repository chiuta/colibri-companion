//! `cargo run -p companion-core --example probe -- /path/to/colibri`
//! Checks the wrapper against a real colibri checkout (read-only commands only).
use companion_core::{coli::Coli, locate};
fn main() {
    let root = std::env::args().nth(1).map(std::path::PathBuf::from);
    let root = locate::find_colibri(root.as_deref()).expect("no colibri checkout found");
    let python = locate::find_python().expect("python >= 3.10 not found");
    println!(
        "colibri: {}\npython:  {} {:?}",
        root.display(),
        python.display(),
        python.version
    );
    let coli = Coli::new(root, python);
    println!("version: {:?}", coli.version());
    println!("hardware: {:#?}", coli.hardware(None));
    let mut models = coli.list_models().expect("catalog");
    companion_core::coli::sort_for_display(&mut models);
    for m in &models {
        println!(
            "{:<24} {:>7.1} GB  fits={:<5} rec={:<5} chat={} {}",
            m.id,
            m.download_gb,
            m.fits,
            m.recommended,
            m.is_chat_model(),
            m.status
        );
    }
    println!("status: {:#?}", coli.status());
}
