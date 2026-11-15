//! End to end against a fake `coli` that follows colibri's documented JSON contract:
//! list models, install (with pause and resume), start, wait for the model to load,
//! chat with streaming, stop. No real model and no network involved.

use companion_core::chat::{ChatClient, ChatMessage, ChatRequest, Health, StreamEvent, Thinking};
use companion_core::coli::{Coli, sort_for_display};
use companion_core::http::CancelToken;
use companion_core::locate;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_colibri")
}

fn wait_until(what: &str, secs: u64, mut f: impl FnMut() -> bool) {
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(secs) {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    panic!("timed out waiting for {what}");
}

#[test]
fn install_pause_resume_start_chat_stop() {
    let Some(python) = locate::find_python() else {
        eprintln!("python >= 3.10 not available: skipping");
        return;
    };
    let home = std::env::temp_dir().join(format!("cc-flow-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    // SAFETY: the integration test runs in its own process and sets these before spawning anything.
    unsafe {
        std::env::set_var("COLI_SETUP_HOME", &home);
        std::env::set_var("FAKE_COLI_PORT", "18791");
        std::env::set_var("FAKE_COLI_SPEED", "4e9"); // 20 GB in about 5 s
        std::env::set_var("FAKE_COLI_LOAD_SECONDS", "2");
    }
    assert!(locate::is_colibri_root(&fixture()));
    let coli = Coli::new(fixture(), python);

    // the catalog, as the picker shows it
    let mut models = coli.list_models().unwrap();
    sort_for_display(&mut models);
    assert_eq!(models[0].id, "fake-small");
    assert!(models[0].recommended && models[0].fits);
    assert!(!models.last().unwrap().fits || !models.last().unwrap().is_chat_model());
    let hw = coli.hardware(None).unwrap();
    assert_eq!(hw.cores, 8);

    // nothing configured yet
    let st = coli.status().unwrap();
    assert!(!st.configured && !st.setup_ready());

    // install, pause quickly, resume, finish
    let log = home.join("install.log");
    let mut child = coli.spawn_install("fake-small", None, &log).unwrap();
    wait_until("the download to start", 15, || {
        coli.status()
            .unwrap()
            .install
            .is_some_and(|i| i.running && i.phase == "download")
    });
    let pid = coli.status().unwrap().install.unwrap().pid.unwrap();
    coli.pause_install(pid).unwrap();
    child.wait().unwrap();
    let st = coli.status().unwrap();
    assert_eq!(st.install.as_ref().unwrap().phase, "interrupted");
    assert!(!st.install_running());
    let done_before = st.install.unwrap().done.unwrap_or(0.0);

    let mut child = coli.spawn_install("fake-small", None, &log).unwrap();
    wait_until("the install to finish", 30, || {
        coli.status().unwrap().setup_ready()
    });
    child.wait().unwrap();
    let st = coli.status().unwrap();
    assert!(st.setup_ready());
    assert_eq!(
        st.setup.unwrap().model.unwrap().name.as_deref(),
        Some("Fake Small 7B")
    );
    assert!(done_before >= 0.0);
    let log_text = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        companion_core::coli::find_package_hint(&log_text).as_deref(),
        Some("sudo apt install libvulkan-dev glslc")
    );

    // start: loading first, then ready
    coli.start_server().unwrap();
    let base = coli
        .status()
        .unwrap()
        .server
        .unwrap()
        .urls
        .anthropic_base_url;
    let client = ChatClient::new(&base, None);
    wait_until("the server to answer", 10, || {
        client.health() != Health::Down
    });
    assert!(
        matches!(client.health(), Health::Loading(_))
            || matches!(client.health(), Health::Ready(_))
    );
    wait_until("the model to load", 20, || {
        matches!(client.health(), Health::Ready(_))
    });
    assert_eq!(client.model_ids().unwrap(), vec!["fake-model"]);
    assert_eq!(coli.status().unwrap().server_state(), "ready");

    // chat, streaming
    let req = ChatRequest {
        model: "fake-model".into(),
        messages: vec![ChatMessage::new("user", "salut lume")],
        temperature: 0.7,
        top_p: 0.95,
        max_tokens: None,
        thinking: Thinking::Auto,
    };
    let mut text = String::new();
    let summary = client
        .stream_chat(&req, &CancelToken::new(), |e| {
            if let StreamEvent::Content(t) = e {
                text.push_str(&t)
            }
        })
        .unwrap();
    assert_eq!(text.trim(), "Fake answer to: salut lume");
    assert_eq!(summary.finish_reason.as_deref(), Some("stop"));

    // stop
    coli.stop_server().unwrap();
    wait_until("the server to stop", 10, || client.health() == Health::Down);
    assert_eq!(coli.status().unwrap().server_state(), "stopped");
    let _ = std::fs::remove_dir_all(&home);
}
