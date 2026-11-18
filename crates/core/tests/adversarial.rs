mod hostile_gen;
use companion_core::*;
use hostile_gen::hammer;

const GH: &str = r#"[{"full_name":"a/b","name":"b","description":null,"html_url":"https://x","default_branch":"main","private":false}]"#;
const TREE: &str =
    r#"{"tree":[{"path":"a/b.md","type":"blob","size":12,"sha":"abc"}],"truncated":false}"#;
const SSE: &str = r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#;
const SMI: &str =
    "0, NVIDIA GeForce RTX 4090, 23028, 24564, 45, 12\n1, X, [N/A], [N/A], [Not Supported], 0";
const LSPCI: &str =
    "01:00.0 VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090] (rev a1)";
const NETSTAT: &str =
    "  TCP    127.0.0.1:8080   0.0.0.0:0   LISTENING   1234\n  TCP [::1]:80 [::]:0 LISTENING 4";
const TOOL: &str = "```tool\n{\"name\":\"x\",\"arguments\":{\"a\":1}}\n```";
const MD: &str = "text\n```rust\nfn main(){}\n```\n~~~\nx\n~~~\n```\nunclosed";

#[test]
fn network_and_server_replies() {
    for kind in [
        connect::Kind::Github,
        connect::Kind::Gitlab,
        connect::Kind::Gitea,
        connect::Kind::Generic,
    ] {
        hammer("connect parsers", &[GH, TREE, "[]", "{}"], move |s| {
            let _ = connect::parse_repos(kind, s.as_bytes());
            let _ = connect::parse_tree(kind, s.as_bytes());
            let _ = connect::parse_issues(kind, s.as_bytes());
        });
    }
    hammer(
        "chat parsers",
        &[SSE, "data: [DONE]", "{\"error\":{\"message\":\"x\"}}"],
        |s| {
            let _ = chat::parse_sse_line(s);
            let _ = chat::parse_error_body(401, s);
            let _ = chat::parse_error_body(500, s);
        },
    );
    hammer(
        "http::parse_authority",
        &[
            "http://localhost:8080/x",
            "[::1]:80",
            "a:b:c",
            "http://u:p@h:99999",
        ],
        |s| {
            let _ = http::parse_authority(s);
        },
    );
    hammer(
        "http::read_response",
        &[
            "HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n",
        ],
        |s| {
            if let Ok(r) = http::read_response(std::io::Cursor::new(s.as_bytes().to_vec())) {
                let _ = (r.status, r.headers.len());
                let _ = r.header("content-length");
            }
        },
    );
    hammer(
        "update parsers",
        &[
            r#"{"tag_name":"v1.0.0","prerelease":true,"html_url":"u","assets":[]}"#,
            r#"{"sha":"abc","commit":{"committer":{"date":"2026-01-01T00:00:00Z"}}}"#,
        ],
        |s| {
            let _ = update::parse_latest_release(s);
            let _ = update::parse_commit(s);
        },
    );
    hammer(
        "coli parsers",
        &["[{\"id\":\"a\"}]", "tune: 12.5 tok/s"],
        |s| {
            let _ = coli::parse_catalog(s);
            let _ = coli::parse_tune_log(s);
        },
    );
    hammer("mcp::parse_tool_request", &[TOOL, "{\"name\":1}"], |s| {
        let _ = mcp::parse_tool_request(s);
    });
}

#[test]
fn system_command_outputs() {
    hammer(
        "gpu",
        &[
            SMI,
            LSPCI,
            "[{\"Name\":\"X\",\"AdapterRAM\":4293918720}]",
            "{\"SPDisplaysDataType\":[]}",
        ],
        |s| {
            let _ = gpu::parse_windows(s);
            let _ = gpu::parse_lspci(s);
            let _ = gpu::parse_macos(s);
            let _ = resources::parse_nvidia_smi(s);
        },
    );
    hammer(
        "net",
        &[
            NETSTAT,
            "127.0.0.1:80",
            "tcp 0 0 0.0.0.0:22 0.0.0.0:* LISTEN 1/init",
        ],
        |s| {
            let _ = net::parse_endpoint(s);
            let _ = net::parse_netstat_windows(s, &[1234, 4]);
            let _ = net::parse_lsof(s);
            let _ = net::parse_proc_net(s, "tcp", false);
            let _ = net::parse_proc_net(s, "tcp", true);
        },
    );
    hammer(
        "misc log parsers",
        &["12.5 tokens/s", "downloaded 12/340 MB", "version 1.2.3"],
        |s| {
            let _ = autotune::parse_tok_s(s);
            let _ = fetch::parse_line(s);
            let _ = locate::parse_version(s);
        },
    );
}

#[test]
fn user_files_and_prompts() {
    hammer("blocks::extract", &[MD], |s| {
        let _ = blocks::extract(s);
    });
    hammer(
        "prompt sets and cases",
        &["# a\nbody\n---\n# b\nbody2", "{\"q\":\"x\"}"],
        |s| {
            let _ = duel::parse_prompt_set(s);
            let _ = flow_eval::parse_cases(s);
        },
    );
}

#[test]
fn rag_with_hostile_text() {
    hammer(
        "rag",
        &["The quick brown fox. Jumps over the lazy dog.\n\nSecond paragraph."],
        |s| {
            let _ = rag::chunk_text(s);
            let mut lib = rag::Library::default();
            if let Some(id) = lib.create("c", 1) {
                let _ = lib.add_document(&id, "d.txt", s);
                if let Some(c) = lib.get(&id) {
                    let se = rag::Searcher::new(c);
                    let hits = se.search(s, 5);
                    let _ = rag::build_context(c, &hits);
                    let _ = se.search("fox", 5);
                }
            }
        },
    );
}
