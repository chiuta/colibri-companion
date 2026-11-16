# Colibri Companion

[English](README.md) · **Română** · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

**Site: https://colibri-companion.chiuta.ro** — descărcare, tutoriale video, ajutor.

O aplicație desktop simplă pentru [colibri](https://github.com/JustVugg/colibri): alege un model potrivit calculatorului tău, descarcă-l cu un clic, apoi conversează — modelul rulează pe calculatorul tău, fără cont și fără serviciu cloud.

## Pornire

```bash
cargo run --release -p colibri-companion            # aplicația reală
cargo run --release -p colibri-companion -- --demo  # demo, fără colibri
# opțiuni: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### Cei patru pași

(1) găsește Python și colibri (și propune descărcarea colibri dacă lipsește), (2) îți permite să alegi un model (cele care nu încap în RAM sunt ascunse; cel recomandat este marcat), (3) descarcă și instalează cu progres, pauză și reluare, (4) pornește serverul și deschide chatul.

Chat: răspunsuri în flux, Stop instant, Markdown, raționament pliabil, conversații salvate, stil (Precis / Echilibrat / Creativ), copiere și regenerare. Teme: deschisă, închisă, contrast ridicat.

## Cum funcționează

Aplicația nu include colibri. Rulează comenzile JSON ale lui colibri (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) și vorbește cu gateway-ul OpenAI-compatibil de pe `127.0.0.1`. Nu instalează niciodată pachete de sistem: dacă lipsește ceva, îți arată comanda de copiat. Datele (setări, conversații) stau local în folderul de date al utilizatorului; `COLIBRI_COMPANION_HOME` îl poate schimba.

## Licențe

- Proiectul: Apache-2.0 (`LICENSE`, `NOTICE`), la fel ca colibri.
- Dependențe: doar licențe permisive, verificate cu `cargo deny check licenses` (`deny.toml`); lista completă în `THIRD_PARTY_LICENSES.md`. Fonturi: Inter și JetBrains Mono (SIL OFL 1.1, în `licenses/`).
- Modelele au licențe proprii; aplicația le arată și cere confirmarea înainte de descărcare.

## Platforme

Linux (x86-64, ARM64; Wayland și X11), Windows 10/11 și macOS (Intel și Apple Silicon) — un singur cod sursă, randare OpenGL (merge și pe plăci video vechi sau în mașini virtuale). `.github/workflows/build.yml` construiește și testează toate variantele și produce arhive gata de distribuit. Pe Linux, `packaging/linux/install.sh` adaugă aplicația în meniu. Scalare HiDPI, scurtături Ctrl/⌘+N, Ctrl/⌘+, și Esc, suport pentru cititoare de ecran (AccessKit). Construirile și testele automate acoperă toate cele trei sisteme; testarea manuală pe calculatoare reale cu Windows și macOS este încă limitată, iar etichetele pentru cititoarele de ecran (AccessKit) nu au fost încă încercate cu cititoare reale.

Linux mai vechi (Debian 8+, Ubuntu 14.04+, CentOS 7+): versiuni suplimentare `linux-x86_64-glibc2.17` și `linux-aarch64-glibc2.17`, legate de glibc 2.17; CI verifică baza simbolurilor și încarcă programul într-un Ubuntu 16.04. Bibliotecile de ferestre și grafică vin tot de la sistemul tău.

**Documente și unelte:** adaugă fișiere (**Mai mult → Documente**) și pune întrebări despre ele: aplicația le caută pe calculatorul tău (după cuvinte, fără cloud) și arată sub fiecare răspuns pasajele folosite. **Mai mult → Unelte MCP** conectează servere MCP adăugate și pornite de tine; modelul poate solicita o unealtă, iar fiecare apel necesită aprobarea ta (funcție timpurie, depinde de cât de bine urmează modelul formatul).

Modele: păstrează mai multe pe calculator, treci între ele sau adaugă altele dintr-o singură fereastră (**Modele**); primul sugerat este cel mai capabil model pe care îl poate rula calculatorul tău (cele cu puțină memorie citesc experții din SSD). Viteză: CUDA pentru NVIDIA, Vulkan pentru alte cipuri grafice, altfel procesorul, alese automat. **Resurse** arată în direct procesorul, memoria, placa video și discul, plus dovada că nimic nu iese din calculator (toate conexiunile deschise ale aplicației și ale motorului, cu comanda pentru verificare proprie). Dacă lipsește colibri, este descărcat pentru tine (fără git), iar o verificare zilnică (care poate fi oprită) te anunță când colibri sau această aplicație au o versiune nouă.

**Calculatorul tău, analizat:** în **Resurse**, aplicația găsește singură plăcile video și spune pe înțeles dacă modelul încape în memorie și de ce poate fi lent (swap, alte programe, o placă neutilizată). Doar sfătuiește; nu schimbă nicio setare de la sine.

## Limbi

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — alese din Setări, cu `--lang en|ro|fr|it|es|pt|de|hu` sau, la prima pornire, după limba sistemului. Traducerile sunt în `crates/app/src/i18n.rs` (cheia e textul englezesc); un test verifică faptul că fiecare text din interfață are toate traducerile. Au fost făcute cu ajutor AI și nu au fost revizuite de vorbitori nativi.

## Dezvoltare

```bash
cargo test --workspace   # include un test cap-coadă cu un `coli` fals (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Nou: căutare în conversații (Ctrl/⌘+K), redenumire (clic dreapta), editarea ultimului mesaj, fișiere text trase în mesaj și salvarea conversației ca fișier .md.
