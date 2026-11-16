//! A catalogue of online platforms a local model can be connected to, by category, with the
//! most it can do there, from the safest (reading what you choose) to the boldest (acting on its
//! own). `Status` says honestly what this program does today; the rest is a map of what is
//! possible and which door to use. Addresses and features were checked in November 2026 and
//! change often: each entry points to the platform's own site.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    /// You pick the content; the model reads it.
    Read = 1,
    /// The model looks things up and browses by itself.
    Search = 2,
    /// The model drafts or changes things, after you approve each change.
    Write = 3,
    /// The model sends messages or alerts.
    Notify = 4,
    /// The model runs actions and chains of steps on its own schedule.
    Act = 5,
}

impl Level {
    pub const ALL: [Level; 5] = [
        Level::Read,
        Level::Search,
        Level::Write,
        Level::Notify,
        Level::Act,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Category {
    Code,
    Docs,
    Projects,
    Chat,
    Mail,
    Web,
    Data,
    Automation,
    Knowledge,
    DevOps,
}

impl Category {
    pub const ALL: [Category; 10] = [
        Category::Code,
        Category::Docs,
        Category::Projects,
        Category::Chat,
        Category::Mail,
        Category::Web,
        Category::Data,
        Category::Automation,
        Category::Knowledge,
        Category::DevOps,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Auth {
    None,
    Token,
    OAuth,
    /// Runs on your own computer or network.
    Local,
    /// A secret address you create on the platform.
    WebhookUrl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Built into this program now.
    Ready,
    /// Works now through a webhook or a plain web address.
    ViaWeb,
    /// Possible, not built yet.
    Planned,
}

#[derive(Clone, Copy, Debug)]
pub struct Platform {
    pub name: &'static str,
    pub category: Category,
    pub url: &'static str,
    pub auth: Auth,
    pub status: Status,
    pub min: Level,
    pub max: Level,
    pub en: &'static str,
    pub ro: &'static str,
}

macro_rules! p {
    ($n:expr, $c:ident, $u:expr, $a:ident, $s:ident, $lo:ident, $hi:ident, $en:expr, $ro:expr) => {
        Platform {
            name: $n,
            category: Category::$c,
            url: $u,
            auth: Auth::$a,
            status: Status::$s,
            min: Level::$lo,
            max: Level::$hi,
            en: $en,
            ro: $ro,
        }
    };
}

pub const ALL: &[Platform] = &[
    // ---- code
    p!(
        "GitHub",
        Code,
        "https://github.com",
        Token,
        Ready,
        Read,
        Search,
        "Browse your repositories, read files and open issues into the chat.",
        "Răsfoiești depozitele tale, citești fișiere și probleme (issues) direct în chat."
    ),
    p!(
        "GitLab",
        Code,
        "https://gitlab.com",
        Token,
        Ready,
        Read,
        Search,
        "Same as GitHub, also on your own GitLab server.",
        "La fel ca GitHub, inclusiv pe serverul tău GitLab."
    ),
    p!(
        "Codeberg / Forgejo / Gitea",
        Code,
        "https://codeberg.org",
        Token,
        Ready,
        Read,
        Search,
        "Free, non-profit hosting and self-hosted forges; read repositories and issues.",
        "Găzduire gratuită, non-profit și forje proprii; citești depozite și probleme."
    ),
    p!(
        "Bitbucket",
        Code,
        "https://bitbucket.org",
        Token,
        Planned,
        Read,
        Search,
        "Read repositories with an app password.",
        "Citești depozite cu o parolă de aplicație."
    ),
    p!(
        "SourceHut",
        Code,
        "https://sr.ht",
        Token,
        Planned,
        Read,
        Search,
        "Read repositories and tickets.",
        "Citești depozite și tichete."
    ),
    p!(
        "Software Heritage",
        Code,
        "https://archive.softwareheritage.org",
        None,
        ViaWeb,
        Read,
        Read,
        "Public archive of source code; read pages by address.",
        "Arhivă publică de cod sursă; citești pagini după adresă."
    ),
    // ---- docs and notes
    p!(
        "Obsidian / Logseq folders",
        Docs,
        "https://obsidian.md",
        Local,
        Ready,
        Read,
        Search,
        "Attach a whole notes folder; the model reads and cites your notes.",
        "Atașezi tot folderul de notițe; modelul le citește și le citează."
    ),
    p!(
        "Nextcloud / WebDAV",
        Docs,
        "https://nextcloud.com",
        Token,
        Planned,
        Read,
        Write,
        "Your own cloud files: read and save documents.",
        "Fișierele din norul tău: citești și salvezi documente."
    ),
    p!(
        "Notion",
        Docs,
        "https://developers.notion.com",
        Token,
        Planned,
        Read,
        Write,
        "Read pages and databases, draft new pages.",
        "Citești pagini și baze de date, redactezi pagini noi."
    ),
    p!(
        "Confluence",
        Docs,
        "https://developer.atlassian.com",
        Token,
        Planned,
        Read,
        Write,
        "Read team wiki pages.",
        "Citești paginile wiki ale echipei."
    ),
    p!(
        "Google Drive / Docs",
        Docs,
        "https://developers.google.com/drive",
        OAuth,
        Planned,
        Read,
        Write,
        "Read and draft documents (needs a Google sign-in).",
        "Citești și redactezi documente (cere autentificare Google)."
    ),
    p!(
        "Wikipedia / Wikimedia",
        Docs,
        "https://www.mediawiki.org/wiki/API",
        None,
        ViaWeb,
        Read,
        Search,
        "Free encyclopedia in every language; read articles by address.",
        "Enciclopedie liberă în toate limbile; citești articole după adresă."
    ),
    // ---- projects
    p!(
        "GitHub / GitLab Issues",
        Projects,
        "https://docs.github.com/rest/issues",
        Token,
        Ready,
        Read,
        Search,
        "Read open issues of a repository and turn them into a plan.",
        "Citești problemele deschise ale unui depozit și le transformi în plan."
    ),
    p!(
        "Jira",
        Projects,
        "https://developer.atlassian.com/cloud/jira",
        Token,
        Planned,
        Read,
        Write,
        "Read tickets, draft comments.",
        "Citești tichete, redactezi comentarii."
    ),
    p!(
        "Linear",
        Projects,
        "https://linear.app/developers",
        Token,
        Planned,
        Read,
        Write,
        "Read and draft issues.",
        "Citești și redactezi sarcini."
    ),
    p!(
        "Trello",
        Projects,
        "https://developer.atlassian.com/cloud/trello",
        Token,
        Planned,
        Read,
        Write,
        "Read boards and cards.",
        "Citești panouri și carduri."
    ),
    p!(
        "Asana",
        Projects,
        "https://developers.asana.com",
        Token,
        Planned,
        Read,
        Write,
        "Read tasks and projects.",
        "Citești sarcini și proiecte."
    ),
    p!(
        "Taiga / OpenProject",
        Projects,
        "https://www.openproject.org",
        Token,
        Planned,
        Read,
        Write,
        "Open-source project tools; read work packages.",
        "Unelte open-source de proiect; citești pachete de lucru."
    ),
    // ---- chat
    p!(
        "Slack",
        Chat,
        "https://api.slack.com/messaging/webhooks",
        WebhookUrl,
        ViaWeb,
        Notify,
        Notify,
        "Post a message to a channel through an incoming webhook.",
        "Trimiți un mesaj pe un canal printr-un webhook de intrare."
    ),
    p!(
        "Discord",
        Chat,
        "https://discord.com/developers/docs/resources/webhook",
        WebhookUrl,
        ViaWeb,
        Notify,
        Notify,
        "Post to a channel through a webhook.",
        "Trimiți pe un canal printr-un webhook."
    ),
    p!(
        "Telegram",
        Chat,
        "https://core.telegram.org/bots/api",
        Token,
        ViaWeb,
        Notify,
        Notify,
        "A bot you create sends you messages and alerts.",
        "Un bot creat de tine îți trimite mesaje și alerte."
    ),
    p!(
        "Matrix / Element",
        Chat,
        "https://matrix.org",
        Token,
        Planned,
        Read,
        Notify,
        "Open, decentralised chat; read and send in rooms.",
        "Chat deschis, descentralizat; citești și trimiți în camere."
    ),
    p!(
        "Mattermost / Rocket.Chat",
        Chat,
        "https://mattermost.com",
        WebhookUrl,
        ViaWeb,
        Notify,
        Notify,
        "Self-hosted team chat; post through a webhook.",
        "Chat de echipă pe serverul tău; trimiți printr-un webhook."
    ),
    p!(
        "ntfy",
        Chat,
        "https://ntfy.sh",
        None,
        ViaWeb,
        Notify,
        Notify,
        "Free push notifications to your phone from any script or model.",
        "Notificări push gratuite pe telefon, de la orice script sau model."
    ),
    p!(
        "Gotify",
        Chat,
        "https://gotify.net",
        Token,
        ViaWeb,
        Notify,
        Notify,
        "Self-hosted push notifications.",
        "Notificări push pe serverul tău."
    ),
    // ---- mail and calendar
    p!(
        "E-mail (IMAP / SMTP)",
        Mail,
        "https://www.rfc-editor.org/rfc/rfc9051",
        Token,
        Planned,
        Read,
        Write,
        "Summarise your inbox, draft replies you approve before sending.",
        "Rezumi căsuța poștală, redactezi răspunsuri pe care le aprobi înainte de trimitere."
    ),
    p!(
        "Calendar (CalDAV / .ics)",
        Mail,
        "https://www.rfc-editor.org/rfc/rfc4791",
        Token,
        Planned,
        Read,
        Write,
        "Read your agenda, propose events.",
        "Citești agenda, propui evenimente."
    ),
    p!(
        "Google Calendar / Gmail",
        Mail,
        "https://developers.google.com/workspace",
        OAuth,
        Planned,
        Read,
        Write,
        "Needs a Google sign-in.",
        "Cere autentificare Google."
    ),
    p!(
        "Microsoft 365 / Outlook",
        Mail,
        "https://learn.microsoft.com/graph",
        OAuth,
        Planned,
        Read,
        Write,
        "Needs a Microsoft sign-in.",
        "Cere autentificare Microsoft."
    ),
    p!(
        "Contacts (CardDAV / .vcf)",
        Mail,
        "https://www.rfc-editor.org/rfc/rfc6352",
        Token,
        Planned,
        Read,
        Read,
        "Read your address book to help write messages.",
        "Citești agenda de contacte ca să ajuți la mesaje."
    ),
    // ---- web and search
    p!(
        "Any web page",
        Web,
        "https://",
        None,
        Ready,
        Read,
        Read,
        "Read a page you give by address, as clean text.",
        "Citești o pagină dată prin adresă, ca text curat."
    ),
    p!(
        "RSS / Atom feeds",
        Web,
        "https://www.rssboard.org/rss-specification",
        None,
        ViaWeb,
        Read,
        Search,
        "Read news and blog feeds by address.",
        "Citești fluxuri de știri și bloguri după adresă."
    ),
    p!(
        "SearXNG (self-hosted search)",
        Web,
        "https://docs.searxng.org",
        None,
        ViaWeb,
        Search,
        Search,
        "A private search engine you run yourself; the model searches the web through it.",
        "Un motor de căutare privat, pornit de tine; modelul caută pe web prin el."
    ),
    p!(
        "Brave Search API",
        Web,
        "https://brave.com/search/api",
        Token,
        Planned,
        Search,
        Search,
        "Web search with an API key.",
        "Căutare pe web cu o cheie API."
    ),
    p!(
        "DuckDuckGo",
        Web,
        "https://duckduckgo.com",
        None,
        Planned,
        Search,
        Search,
        "Quick lookups without an account.",
        "Căutări rapide fără cont."
    ),
    p!(
        "Internet Archive",
        Web,
        "https://archive.org/developers",
        None,
        ViaWeb,
        Read,
        Search,
        "Old copies of pages, books and recordings.",
        "Copii vechi de pagini, cărți și înregistrări."
    ),
    // ---- data and files
    p!(
        "Local folders and files",
        Data,
        "",
        Local,
        Ready,
        Read,
        Read,
        "Drop any folder: text, PDF, Office, HTML, images.",
        "Tragi orice folder: text, PDF, Office, HTML, imagini."
    ),
    p!(
        "SQLite / CSV / spreadsheets",
        Data,
        "https://sqlite.org",
        Local,
        Planned,
        Read,
        Search,
        "Ask questions about your tables; the model writes the queries you approve.",
        "Pui întrebări despre tabelele tale; modelul scrie interogările pe care le aprobi."
    ),
    p!(
        "PostgreSQL / MariaDB",
        Data,
        "https://www.postgresql.org",
        Token,
        Planned,
        Read,
        Search,
        "Read-only questions about a database.",
        "Întrebări doar-citire despre o bază de date."
    ),
    p!(
        "Airtable",
        Data,
        "https://airtable.com/developers",
        Token,
        Planned,
        Read,
        Write,
        "Read and draft table rows.",
        "Citești și redactezi rânduri de tabel."
    ),
    p!(
        "Google Sheets",
        Data,
        "https://developers.google.com/sheets",
        OAuth,
        Planned,
        Read,
        Write,
        "Needs a Google sign-in.",
        "Cere autentificare Google."
    ),
    p!(
        "S3 / MinIO / object storage",
        Data,
        "https://min.io",
        Token,
        Planned,
        Read,
        Write,
        "Read and store files in buckets.",
        "Citești și salvezi fișiere în găleți (buckets)."
    ),
    // ---- automation
    p!(
        "n8n",
        Automation,
        "https://n8n.io",
        WebhookUrl,
        ViaWeb,
        Notify,
        Act,
        "Open-source automation: the model calls a workflow with a webhook, the workflow does the rest.",
        "Automatizare open-source: modelul cheamă un flux printr-un webhook, restul îl face fluxul."
    ),
    p!(
        "Node-RED",
        Automation,
        "https://nodered.org",
        WebhookUrl,
        ViaWeb,
        Notify,
        Act,
        "Wire devices and services; an HTTP-in node receives the model's calls.",
        "Legi dispozitive și servicii; un nod HTTP-in primește apelurile modelului."
    ),
    p!(
        "Zapier",
        Automation,
        "https://zapier.com",
        WebhookUrl,
        ViaWeb,
        Notify,
        Act,
        "Thousands of apps behind one webhook (paid service).",
        "Mii de aplicații în spatele unui webhook (serviciu cu plată)."
    ),
    p!(
        "Make",
        Automation,
        "https://make.com",
        WebhookUrl,
        ViaWeb,
        Notify,
        Act,
        "Visual automations behind a webhook (paid service).",
        "Automatizări vizuale în spatele unui webhook (serviciu cu plată)."
    ),
    p!(
        "IFTTT",
        Automation,
        "https://ifttt.com",
        WebhookUrl,
        ViaWeb,
        Notify,
        Act,
        "Simple triggers and actions through webhooks.",
        "Declanșatori și acțiuni simple prin webhook-uri."
    ),
    p!(
        "Home Assistant",
        Automation,
        "https://www.home-assistant.io",
        Token,
        ViaWeb,
        Read,
        Act,
        "Smart home: a webhook can switch lights, scenes and scripts. Keep the model away from locks and alarms.",
        "Casă inteligentă: un webhook poate porni lumini, scene și scripturi. Ține modelul departe de încuietori și alarme."
    ),
    p!(
        "Cron / system scheduler",
        Automation,
        "",
        Local,
        Ready,
        Act,
        Act,
        "Run saved automations on a timetable while this program is open.",
        "Rulezi automatizările salvate după un orar, cât timp programul e deschis."
    ),
    // ---- knowledge
    p!(
        "arXiv",
        Knowledge,
        "https://info.arxiv.org/help/api",
        None,
        ViaWeb,
        Read,
        Search,
        "Open research papers; read abstracts and PDFs by address.",
        "Lucrări de cercetare deschise; citești rezumate și PDF-uri după adresă."
    ),
    p!(
        "OpenAlex / Crossref",
        Knowledge,
        "https://docs.openalex.org",
        None,
        ViaWeb,
        Search,
        Search,
        "Open catalogues of scholarly works.",
        "Cataloage deschise de lucrări științifice."
    ),
    p!(
        "Zotero",
        Knowledge,
        "https://www.zotero.org/support/dev/web_api/v3/start",
        Token,
        Planned,
        Read,
        Search,
        "Your reference library and notes.",
        "Biblioteca ta de referințe și notițe."
    ),
    p!(
        "Hugging Face",
        Knowledge,
        "https://huggingface.co",
        Token,
        ViaWeb,
        Read,
        Search,
        "Model cards and datasets; read pages by address.",
        "Fișe de modele și seturi de date; citești pagini după adresă."
    ),
    p!(
        "Wikidata",
        Knowledge,
        "https://www.wikidata.org",
        None,
        ViaWeb,
        Search,
        Search,
        "Structured facts about people, places and things.",
        "Fapte structurate despre oameni, locuri și lucruri."
    ),
    p!(
        "data.gov.ro / open data portals",
        Knowledge,
        "https://data.gov.ro",
        None,
        ViaWeb,
        Read,
        Search,
        "Open public data.",
        "Date publice deschise."
    ),
    // ---- dev ops
    p!(
        "Docker / Podman",
        DevOps,
        "https://www.docker.com",
        Local,
        Planned,
        Read,
        Act,
        "Read logs and container status; start or stop only with your approval.",
        "Citești jurnale și starea containerelor; pornești sau oprești doar cu aprobarea ta."
    ),
    p!(
        "CI logs (GitHub Actions, GitLab CI)",
        DevOps,
        "https://docs.github.com/actions",
        Token,
        Planned,
        Read,
        Search,
        "Read failed build logs and explain the failure.",
        "Citești jurnalele build-urilor eșuate și explici eșecul."
    ),
    p!(
        "Sentry / GlitchTip",
        DevOps,
        "https://sentry.io",
        Token,
        Planned,
        Read,
        Search,
        "Read error reports and group them.",
        "Citești rapoarte de erori și le grupezi."
    ),
    p!(
        "Grafana / Prometheus",
        DevOps,
        "https://grafana.com",
        Token,
        Planned,
        Read,
        Search,
        "Read metrics and explain a spike.",
        "Citești metrici și explici un vârf."
    ),
    p!(
        "MCP servers (coli mcp and others)",
        DevOps,
        "https://modelcontextprotocol.io",
        Local,
        Planned,
        Read,
        Act,
        "A standard way for programs to offer tools to a model; colibri has its own.",
        "Un mod standard prin care programele oferă unelte unui model; colibri are propriul server."
    ),
];

pub fn in_category(c: Category) -> impl Iterator<Item = &'static Platform> {
    ALL.iter().filter(move |p| p.category == c)
}

/// What one level means, for the legend.
pub fn level_text(l: Level, romanian: bool) -> &'static str {
    match (l, romanian) {
        (Level::Read, false) => "Read: you pick the content, the model reads it",
        (Level::Read, true) => "Citire: tu alegi conținutul, modelul îl citește",
        (Level::Search, false) => "Search: the model looks things up and browses by itself",
        (Level::Search, true) => "Căutare: modelul caută și răsfoiește singur",
        (Level::Write, false) => "Write: it drafts or changes things; you approve each change",
        (Level::Write, true) => "Scriere: redactează sau schimbă; aprobi fiecare schimbare",
        (Level::Notify, false) => "Notify: it sends messages and alerts",
        (Level::Notify, true) => "Notificare: trimite mesaje și alerte",
        (Level::Act, false) => "Act: it runs actions and chains of steps on a schedule",
        (Level::Act, true) => "Acțiune: rulează acțiuni și lanțuri de pași după un orar",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_is_complete_and_sane() {
        assert!(ALL.len() >= 50);
        let mut names: Vec<&str> = ALL.iter().map(|p| p.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), ALL.len(), "names are unique");
        for p in ALL {
            assert!(p.min <= p.max, "{}: min above max", p.name);
            assert!(!p.en.is_empty() && !p.ro.is_empty(), "{}", p.name);
            assert!(
                p.url.is_empty() || p.url.starts_with("https://"),
                "{}",
                p.name
            );
        }
    }

    #[test]
    fn every_category_has_entries_and_levels_are_ordered() {
        for c in Category::ALL {
            assert!(in_category(c).count() >= 2, "{c:?}");
        }
        assert!(Level::Read < Level::Search && Level::Notify < Level::Act);
        assert!(Level::ALL.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn nothing_that_can_act_on_its_own_is_marked_ready_without_a_reason() {
        // The only built-in entries are the ones this program really does today.
        let ready: Vec<&str> = ALL
            .iter()
            .filter(|p| p.status == Status::Ready)
            .map(|p| p.name)
            .collect();
        for n in [
            "GitHub",
            "GitLab",
            "Any web page",
            "Local folders and files",
        ] {
            assert!(ready.contains(&n), "{n}");
        }
        assert!(
            ALL.iter()
                .filter(|p| p.max == Level::Act && p.status == Status::Ready)
                .count()
                <= 1
        );
    }
}
