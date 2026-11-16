//! The MCP client against a tiny fake server (a Python script speaking JSON-RPC over stdio).

use companion_core::mcp::{Client, ServerConfig};
use serde_json::json;
use std::process::Command;

const SERVER: &str = r#"
import sys, json
def send(o): sys.stdout.write(json.dumps(o)+"\n"); sys.stdout.flush()
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m or "method" not in m: continue
    i, meth = m["id"], m["method"]
    if meth == "initialize":
        # a server may ask the client something first; the client must not hang on it
        send({"jsonrpc":"2.0","id":900,"method":"roots/list"})
        send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"1"}}})
    elif meth == "tools/list":
        send({"jsonrpc":"2.0","id":i,"result":{"tools":[
          {"name":"echo","description":"Echo","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}},
          {"name":"fail","description":"Fails","inputSchema":{}},
          {"name":"die","description":"Exits","inputSchema":{}}]}})
    elif meth == "tools/call":
        n = m["params"]["name"]; a = m["params"]["arguments"]
        if n == "echo":
            send({"jsonrpc":"2.0","id":i,"result":{"content":[{"type":"text","text":"you said: "+a.get("text","")},{"type":"image","data":"x","mimeType":"image/png"}]}})
        elif n == "fail":
            send({"jsonrpc":"2.0","id":i,"result":{"isError":True,"content":[{"type":"text","text":"nope"}]}})
        elif n == "die":
            sys.exit(0)
"#;

fn python() -> Option<String> {
    ["python3", "python"]
        .iter()
        .find(|p| Command::new(p).arg("--version").output().is_ok())
        .map(|s| s.to_string())
}

fn start() -> Option<(Client, Vec<companion_core::mcp::Tool>)> {
    let py = python()?;
    let dir = std::env::temp_dir().join(format!(
        "mcp-test-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("server.py");
    std::fs::write(&script, SERVER).unwrap();
    let cfg = ServerConfig {
        name: "fake".into(),
        command: py,
        args: vec![script.display().to_string()],
    };
    Some(Client::start(&cfg).expect("handshake"))
}

#[test]
fn handshake_lists_tools_and_calls_them() {
    let Some((c, tools)) = start() else { return };
    assert_eq!(
        tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["echo", "fail", "die"]
    );
    let out = c.call("echo", json!({"text": "salut"})).unwrap();
    assert!(out.contains("you said: salut"));
    assert!(out.contains("image content not shown"), "{out}");
}

#[test]
fn a_tool_error_is_an_error_with_its_text() {
    let Some((c, _)) = start() else { return };
    assert_eq!(c.call("fail", json!({})).unwrap_err(), "nope");
}

#[test]
fn a_server_that_dies_mid_call_gives_an_error_not_a_hang() {
    let Some((c, _)) = start() else { return };
    let started = std::time::Instant::now();
    assert!(c.call("die", json!({})).is_err());
    assert!(started.elapsed().as_secs() < 20);
    // and it keeps failing cleanly afterwards
    assert!(c.call("echo", json!({})).is_err());
}

#[test]
fn shutdown_stops_the_server_and_later_calls_fail_fast() {
    let Some((c, _)) = start() else { return };
    c.shutdown();
    let started = std::time::Instant::now();
    assert!(c.call("echo", json!({})).is_err());
    assert!(started.elapsed().as_secs() < 20);
    c.shutdown(); // a second call is harmless
}

#[test]
fn a_program_that_does_not_exist_is_a_clear_error() {
    let cfg = ServerConfig {
        name: "x".into(),
        command: "/definitely/not/a/program".into(),
        args: vec![],
    };
    assert!(Client::start(&cfg).is_err());
}
