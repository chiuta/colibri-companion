//! `--demo`: a tiny stand-in for colibri's server, so the interface can be tried
//! without downloading a model. It speaks the same wire format as `coli serve`
//! (close-framed server-sent events), streams a canned Markdown answer word by
//! word, and notices when the app closes the connection. Plain `std` only.

use companion_core::Lang;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

pub struct DemoServer {
    pub base: String,
}

pub fn start(lang: Lang) -> io::Result<DemoServer> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let base = format!("http://{}", listener.local_addr()?);
    thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            thread::spawn(move || {
                let _ = handle(conn, lang);
            });
        }
    });
    Ok(DemoServer { base })
}

struct Request {
    method: String,
    path: String,
    body: String,
}

fn read_request(stream: &mut TcpStream) -> io::Result<Request> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte)? == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf.push(byte[0]);
        if buf.len() > 64 * 1024 {
            return Err(io::ErrorKind::InvalidData.into());
        }
    }
    let head = String::from_utf8_lossy(&buf).into_owned();
    let mut lines = head.lines();
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let (method, path) = (
        first.next().unwrap_or("").to_string(),
        first.next().unwrap_or("").to_string(),
    );
    let len = lines
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body)?;
    Ok(Request {
        method,
        path,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn respond_json(stream: &mut TcpStream, status: &str, body: &str) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn handle(mut stream: TcpStream, lang: Lang) -> io::Result<()> {
    let req = read_request(&mut stream)?;
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => respond_json(
            &mut stream,
            "200 OK",
            r#"{"status":"ok","arch":"demo","context_tokens":32768,"max_tokens":4096,"kv_slots":1,"input_modalities":["text"],"system":{"rss_gb":0.1}}"#,
        ),
        ("GET", "/v1/models") => respond_json(
            &mut stream,
            "200 OK",
            r#"{"object":"list","data":[{"id":"demo-model","object":"model"}]}"#,
        ),
        ("POST", "/v1/chat/completions") => chat(&mut stream, &req.body, lang),
        _ => respond_json(
            &mut stream,
            "404 Not Found",
            r#"{"error":{"message":"not found","code":"not_found"}}"#,
        ),
    }
}

fn sse(stream: &mut TcpStream, delta: serde_json::Value) -> io::Result<()> {
    let line = format!(
        "data: {}\n\n",
        serde_json::json!({ "choices": [{ "index": 0, "delta": delta }] })
    );
    stream.write_all(line.as_bytes())?;
    stream.flush()
}

fn chat(stream: &mut TcpStream, body: &str, lang: Lang) -> io::Result<()> {
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let last_user = v["messages"]
        .as_array()
        .and_then(|a| a.iter().rev().find(|m| m["role"] == "user"))
        .and_then(|m| m["content"].as_str())
        .unwrap_or("")
        .to_string();
    if last_user.trim() == "/busy" {
        return respond_json(
            stream,
            "429 Too Many Requests",
            r#"{"error":{"message":"the queue is full","code":"queue_full"}}"#,
        );
    }
    if last_user.trim() == "/long" {
        return respond_json(
            stream,
            "400 Bad Request",
            r#"{"error":{"message":"prompt exceeds the context","code":"context_length_exceeded"}}"#,
        );
    }
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    sse(
        stream,
        serde_json::json!({ "role": "assistant", "content": "" }),
    )?;
    let w = wording(lang);
    let thinking = v["enable_thinking"] != serde_json::Value::Bool(false);
    if thinking {
        for word in w[9].split_inclusive(' ') {
            sse(stream, serde_json::json!({ "reasoning_content": word }))?;
            thread::sleep(Duration::from_millis(25));
        }
    }
    let quoted: String = last_user.chars().take(120).collect();
    let system = v["messages"]
        .as_array()
        .and_then(|a| a.iter().find(|m| m["role"] == "system"))
        .and_then(|m| m["content"].as_str())
        .unwrap_or("");
    let answer = scripted_answer(system, &last_user).unwrap_or_else(|| format!(
        "## {}\n\n{}\n\n{}: *“{quoted}”*\n\n{}:\n\n- {}\n- {}\n- {}\n\n```python\nprint(\"{}\")\n```\n\n> {}\n",
        w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]
    ));
    let delay = if last_user.trim() == "/slow" { 400 } else { 22 };
    let mut tokens = 0u64;
    for w in answer.split_inclusive(' ') {
        sse(stream, serde_json::json!({ "content": w }))?;
        tokens += 1;
        thread::sleep(Duration::from_millis(delay));
    }
    sse(stream, serde_json::json!({}))?;
    let done =
        serde_json::json!({ "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }] });
    stream.write_all(format!("data: {done}\n\n").as_bytes())?;
    let usage = serde_json::json!({ "choices": [], "usage": { "prompt_tokens": last_user.len() / 4 + 8, "completion_tokens": tokens } });
    stream.write_all(format!("data: {usage}\n\ndata: [DONE]\n\n").as_bytes())?;
    stream.flush()
}

/// Answers that show the document and tool features without a real model: the demo quotes the
/// first document passage it was given, asks for the first listed tool when the message says
/// `/tool`, and repeats what a tool returned.
fn scripted_answer(system: &str, last_user: &str) -> Option<String> {
    if let Some(rest) = last_user.strip_prefix("Tool ") {
        let body = rest.split("```").nth(1).unwrap_or("").trim();
        let shown: String = body.chars().take(160).collect();
        return Some(format!("The tool answered: {shown}"));
    }
    if last_user.contains("/tool") {
        if let Some(line) = system
            .lines()
            .find(|l| l.starts_with("- ") && l.contains('/'))
        {
            let name = line[2..].split(':').next().unwrap_or("");
            if let Some((server, tool)) = name.split_once('/') {
                return Some(format!(
                    "I will ask the tool.\n```tool\n{{\"server\": \"{server}\", \"tool\": \"{tool}\", \"arguments\": {{\"text\": \"hello\"}}}}\n```"
                ));
            }
        }
    }
    if system.contains("answer from their documents") {
        let start = system.find("\n[1] ")?;
        let mut lines = system[start + 1..].lines();
        lines.next()?; // "[1] file, part n"
        let passage: String = lines.next()?.chars().take(220).collect();
        return Some(format!("From your documents: {passage} [1]"));
    }
    None
}

/// The demo answer's wording in every interface language.
fn wording(lang: Lang) -> [&'static str; 10] {
    match lang.code() {
        "en" => [
            "Hello!",
            "This is Colibri Companion's **demo answer**. No model runs here, but this is what real answers look like.",
            "You wrote",
            "What you can do next",
            "install a model (the start screen shows what fits your computer);",
            "change the answer style: **precise**, **balanced** or **creative**;",
            "make the text bigger in Settings if it is easier to read.",
            "Everything stays on your computer",
            "Nothing leaves this machine.",
            "The person wrote something short. I'll answer in a friendly way, with a little Markdown, to show how text looks.",
        ],
        "ro" => [
            "Salut!",
            "Sunt **răspunsul de demonstrație** al lui Colibri Companion. Nu rulează niciun model aici, dar aceasta e forma pe care o au răspunsurile reale.",
            "Ai scris",
            "Ce poți face în continuare",
            "instalează un model (ecranul de pornire îți arată ce se potrivește calculatorului tău);",
            "schimbă stilul răspunsului: **precis**, **echilibrat** sau **creativ**;",
            "mărește textul din Setări dacă îți e mai comod.",
            "Diacritice: ă â î ș ț Ă Â Î Ș Ț",
            "Totul rămâne pe calculatorul tău.",
            "Omul mi-a scris ceva scurt. Răspund prietenos, cu un exemplu de Markdown, ca să se vadă cum arată textul.",
        ],
        "fr" => [
            "Bonjour !",
            "Voici la **réponse de démonstration** de Colibri Companion. Aucun modèle ne tourne ici, mais les vraies réponses ont cette forme.",
            "Vous avez écrit",
            "Ce que vous pouvez faire ensuite",
            "installer un modèle (l'écran de démarrage montre ce qui convient à votre ordinateur) ;",
            "changer le style des réponses : **précis**, **équilibré** ou **créatif** ;",
            "agrandir le texte dans Réglages si c'est plus confortable.",
            "Tout reste sur votre ordinateur : é è ê ç à ù",
            "Rien ne quitte cet ordinateur.",
            "La personne a écrit quelque chose de court. Je réponds de façon amicale, avec un peu de Markdown, pour montrer l'aspect du texte.",
        ],
        "it" => [
            "Ciao!",
            "Questa è la **risposta dimostrativa** di Colibri Companion. Qui non gira nessun modello, ma le risposte vere hanno questa forma.",
            "Hai scritto",
            "Cosa puoi fare ora",
            "installa un modello (la schermata iniziale mostra cosa va bene per il tuo computer);",
            "cambia lo stile delle risposte: **preciso**, **equilibrato** o **creativo**;",
            "ingrandisci il testo nelle Impostazioni se ti è più comodo.",
            "Tutto resta sul tuo computer: à è é ì ò ù",
            "Niente lascia questo computer.",
            "La persona ha scritto qualcosa di breve. Rispondo in modo cordiale, con un po' di Markdown, per mostrare l'aspetto del testo.",
        ],
        "es" => [
            "¡Hola!",
            "Esta es la **respuesta de demostración** de Colibri Companion. Aquí no se ejecuta ningún modelo, pero así se ven las respuestas reales.",
            "Has escrito",
            "Qué puedes hacer ahora",
            "instala un modelo (la pantalla de inicio muestra lo que encaja con tu ordenador);",
            "cambia el estilo de las respuestas: **preciso**, **equilibrado** o **creativo**;",
            "aumenta el texto en Ajustes si te resulta más cómodo.",
            "Todo se queda en tu ordenador: á é í ó ú ñ ¿ ¡",
            "Nada sale de este equipo.",
            "La persona escribió algo breve. Responderé de forma amable, con un poco de Markdown, para mostrar cómo se ve el texto.",
        ],
        "pt" => [
            "Olá!",
            "Esta é a **resposta de demonstração** do Colibri Companion. Aqui não corre nenhum modelo, mas as respostas reais têm este aspeto.",
            "Escreveste",
            "O que podes fazer a seguir",
            "instala um modelo (o ecrã inicial mostra o que serve ao teu computador);",
            "muda o estilo das respostas: **preciso**, **equilibrado** ou **criativo**;",
            "aumenta o texto nas Definições se for mais confortável.",
            "Tudo fica no teu computador: á é ã õ ç ê",
            "Nada sai deste computador.",
            "A pessoa escreveu algo curto. Vou responder de forma simpática, com um pouco de Markdown, para mostrar o aspeto do texto.",
        ],
        "de" => [
            "Hallo!",
            "Das ist die **Demo-Antwort** von Colibri Companion. Hier läuft kein Modell, aber so sehen echte Antworten aus.",
            "Du hast geschrieben",
            "Was du als Nächstes tun kannst",
            "ein Modell installieren (der Startbildschirm zeigt, was zu deinem Computer passt);",
            "den Antwortstil ändern: **präzise**, **ausgewogen** oder **kreativ**;",
            "den Text in den Einstellungen vergrößern, wenn er so besser lesbar ist.",
            "Alles bleibt auf deinem Computer: ä ö ü ß",
            "Nichts verlässt diesen Rechner.",
            "Die Person hat etwas Kurzes geschrieben. Ich antworte freundlich, mit etwas Markdown, um zu zeigen, wie Text aussieht.",
        ],
        "hu" => [
            "Szia!",
            "Ez a Colibri Companion **bemutató válasza**. Itt nem fut modell, de a valódi válaszok így néznek ki.",
            "Ezt írtad",
            "Mit tehetsz ezután",
            "telepíts egy modellt (a kezdőképernyő megmutatja, mi illik a számítógépedhez);",
            "változtasd meg a válasz stílusát: **pontos**, **kiegyensúlyozott** vagy **kreatív**;",
            "növeld a szöveg méretét a Beállításokban, ha így kényelmesebb.",
            "Minden a számítógépeden marad: á é í ó ö ő ú ü ű",
            "Semmi nem hagyja el ezt a gépet.",
            "A személy valami rövidet írt. Barátságosan válaszolok, egy kis Markdownnal, hogy látszódjon, hogyan néz ki a szöveg.",
        ],
        _ => unreachable!(),
    }
}
