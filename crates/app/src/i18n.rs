//! Interface translations for the six languages beyond Romanian and English.
//! English text is the key; Romanian and English live next to the code (`t(ro, en)`).
//! Order of each row: fr, it, es, pt, de, hu. Written with AI help; native review is welcome.

use companion_core::Lang;
use std::collections::HashMap;
use std::sync::OnceLock;

static TABLE: &[(&str, [&str; 6])] = &[
    (
        "That folder doesn't look like colibri (c/coli is missing). Pick the folder that contains «c», «docs» and «README.md».",
        [
            "Ce dossier ne ressemble pas à colibri (c/coli est introuvable). Choisissez le dossier qui contient «c», «docs» et «README.md».",
            "Questa cartella non sembra colibri (manca c/coli). Scegli la cartella che contiene «c», «docs» e «README.md».",
            "Esa carpeta no parece colibri (falta c/coli). Elige la carpeta que contiene «c», «docs» y «README.md».",
            "Esta pasta não parece ser o colibri (falta c/coli). Escolhe a pasta que contém «c», «docs» e «README.md».",
            "Dieser Ordner sieht nicht nach colibri aus (c/coli fehlt). Wähle den Ordner, der «c», «docs» und «README.md» enthält.",
            "Ez a mappa nem tűnik colibrinek (hiányzik a c/coli). Válaszd ki azt a mappát, amelyik a «c», a «docs» és a «README.md» elemeket tartalmazza.",
        ],
    ),
    (
        "The download was paused. You can resume it any time.",
        [
            "Le téléchargement a été mis en pause. Vous pouvez le reprendre à tout moment.",
            "Il download è in pausa. Puoi riprenderlo quando vuoi.",
            "La descarga está en pausa. Puedes reanudarla cuando quieras.",
            "A transferência foi colocada em pausa. Podes retomá-la quando quiseres.",
            "Der Download wurde pausiert. Du kannst ihn jederzeit fortsetzen.",
            "A letöltés szünetel. Bármikor folytathatod.",
        ],
    ),
    (
        "Starting the server…",
        [
            "Démarrage du serveur…",
            "Avvio del server…",
            "Iniciando el servidor…",
            "A iniciar o servidor…",
            "Server wird gestartet…",
            "A szerver indítása…",
        ],
    ),
    (
        "Stopping the server…",
        [
            "Arrêt du serveur…",
            "Arresto del server…",
            "Deteniendo el servidor…",
            "A parar o servidor…",
            "Server wird beendet…",
            "A szerver leállítása…",
        ],
    ),
    (
        "Could not save the conversation",
        [
            "Impossible d'enregistrer la conversation",
            "Impossibile salvare la conversazione",
            "No se pudo guardar la conversación",
            "Não foi possível guardar a conversa",
            "Die Unterhaltung konnte nicht gespeichert werden",
            "Nem sikerült elmenteni a beszélgetést",
        ],
    ),
    (
        "The download failed",
        [
            "Le téléchargement a échoué",
            "Il download non è riuscito",
            "La descarga falló",
            "A transferência falhou",
            "Der Download ist fehlgeschlagen",
            "A letöltés nem sikerült",
        ],
    ),
    (
        "Could not start the setup",
        [
            "Impossible de lancer l'installation",
            "Impossibile avviare l'installazione",
            "No se pudo iniciar la instalación",
            "Não foi possível iniciar a instalação",
            "Die Einrichtung konnte nicht gestartet werden",
            "Nem sikerült elindítani a telepítést",
        ],
    ),
    (
        "Copy conversation (Markdown)",
        [
            "Copier la conversation (Markdown)",
            "Copia la conversazione (Markdown)",
            "Copiar la conversación (Markdown)",
            "Copiar a conversa (Markdown)",
            "Unterhaltung kopieren (Markdown)",
            "Beszélgetés másolása (Markdown)",
        ],
    ),
    ("Me", ["Moi", "Io", "Yo", "Eu", "Ich", "Én"]),
    (
        "Conversation copied",
        [
            "Conversation copiée",
            "Conversazione copiata",
            "Conversación copiada",
            "Conversa copiada",
            "Unterhaltung kopiert",
            "Beszélgetés kimásolva",
        ],
    ),
    (
        "+ New conversation",
        [
            "+ Nouvelle conversation",
            "+ Nuova conversazione",
            "+ Nueva conversación",
            "+ Nova conversa",
            "+ Neue Unterhaltung",
            "+ Új beszélgetés",
        ],
    ),
    (
        "Your conversations",
        [
            "Vos conversations",
            "Le tue conversazioni",
            "Tus conversaciones",
            "As tuas conversas",
            "Deine Unterhaltungen",
            "A beszélgetéseid",
        ],
    ),
    (
        "Saved conversations will show up here.",
        [
            "Les conversations enregistrées apparaîtront ici.",
            "Le conversazioni salvate compariranno qui.",
            "Las conversaciones guardadas aparecerán aquí.",
            "As conversas guardadas aparecerão aqui.",
            "Gespeicherte Unterhaltungen erscheinen hier.",
            "A mentett beszélgetések itt jelennek meg.",
        ],
    ),
    (
        "(untitled)",
        [
            "(sans titre)",
            "(senza titolo)",
            "(sin título)",
            "(sem título)",
            "(ohne Titel)",
            "(cím nélkül)",
        ],
    ),
    (
        "Sure?",
        [
            "Sûr ?",
            "Sicuro?",
            "¿Seguro?",
            "Tens a certeza?",
            "Sicher?",
            "Biztos?",
        ],
    ),
    (
        "Press again to delete",
        [
            "Appuyez encore pour supprimer",
            "Premi ancora per eliminare",
            "Pulsa otra vez para borrar",
            "Carrega outra vez para apagar",
            "Zum Löschen noch einmal drücken",
            "Nyomd meg újra a törléshez",
        ],
    ),
    (
        "Delete conversation",
        [
            "Supprimer la conversation",
            "Elimina la conversazione",
            "Borrar conversación",
            "Apagar conversa",
            "Unterhaltung löschen",
            "Beszélgetés törlése",
        ],
    ),
    (
        "How can I help today?",
        [
            "Comment puis-je vous aider aujourd'hui ?",
            "Come posso aiutarti oggi?",
            "¿En qué puedo ayudarte hoy?",
            "Como posso ajudar-te hoje?",
            "Wobei kann ich dir heute helfen?",
            "Miben segíthetek ma?",
        ],
    ),
    (
        "Type below or pick an idea. Everything stays on your computer.",
        [
            "Écrivez ci-dessous ou choisissez une idée. Tout reste sur votre ordinateur.",
            "Scrivi qui sotto o scegli un'idea. Tutto resta sul tuo computer.",
            "Escribe abajo o elige una idea. Todo se queda en tu ordenador.",
            "Escreve abaixo ou escolhe uma ideia. Tudo fica no teu computador.",
            "Schreib unten oder wähle eine Idee. Alles bleibt auf deinem Computer.",
            "Írj alább, vagy válassz egy ötletet. Minden a saját számítógépeden marad.",
        ],
    ),
    (
        "Explain something simply",
        [
            "Expliquer simplement",
            "Spiegami qualcosa in modo semplice",
            "Explícame algo de forma sencilla",
            "Explica-me algo de forma simples",
            "Erkläre mir etwas einfach",
            "Magyarázz el valamit egyszerűen",
        ],
    ),
    (
        "Explain to me, like I'm 10, how the internet works.",
        [
            "Explique-moi, comme si j'avais 10 ans, comment fonctionne Internet.",
            "Spiegami, come se avessi 10 anni, come funziona internet.",
            "Explícame, como si tuviera 10 años, cómo funciona internet.",
            "Explica-me, como se eu tivesse 10 anos, como funciona a internet.",
            "Erkläre mir, als wäre ich 10, wie das Internet funktioniert.",
            "Magyarázd el úgy, mintha 10 éves lennék, hogyan működik az internet.",
        ],
    ),
    (
        "Help me with an email",
        [
            "M'aider avec un e-mail",
            "Aiutami con un'e-mail",
            "Ayúdame con un correo",
            "Ajuda-me com um e-mail",
            "Hilf mir bei einer E-Mail",
            "Segíts egy e-mailben",
        ],
    ),
    (
        "Help me write a polite email asking for a deadline extension.",
        [
            "Aide-moi à écrire un e-mail poli pour demander un délai supplémentaire.",
            "Aiutami a scrivere un'e-mail cortese per chiedere una proroga.",
            "Ayúdame a escribir un correo educado pidiendo una prórroga de plazo.",
            "Ajuda-me a escrever um e-mail educado a pedir o adiamento de um prazo.",
            "Hilf mir, eine höfliche E-Mail zu schreiben, in der ich um eine Fristverlängerung bitte.",
            "Segíts írni egy udvarias e-mailt, amelyben határidő-hosszabbítást kérek.",
        ],
    ),
    (
        "Summarise and clarify",
        [
            "Résumer et clarifier",
            "Riassumi e chiarisci",
            "Resume y aclara",
            "Resumir e clarificar",
            "Zusammenfassen und klären",
            "Összegzés és tisztázás",
        ],
    ),
    (
        "I'll paste a text; summarise it in 5 points and tell me what's unclear.",
        [
            "Je vais coller un texte ; résume-le en 5 points et dis-moi ce qui n'est pas clair.",
            "Incollerò un testo: riassumilo in 5 punti e dimmi cosa non è chiaro.",
            "Voy a pegar un texto; resúmelo en 5 puntos y dime qué no está claro.",
            "Vou colar um texto; resume-o em 5 pontos e diz-me o que não está claro.",
            "Ich füge gleich einen Text ein; fasse ihn in 5 Punkten zusammen und sag mir, was unklar ist.",
            "Be fogok illeszteni egy szöveget; foglald össze 5 pontban, és mondd meg, mi nem világos.",
        ],
    ),
    (
        "Ideas for an evening",
        [
            "Idées pour une soirée",
            "Idee per una serata",
            "Ideas para una tarde-noche",
            "Ideias para uma noite",
            "Ideen für einen Abend",
            "Ötletek egy estére",
        ],
    ),
    (
        "Give me 5 ideas for quiet evenings at home, without screens.",
        [
            "Donne-moi 5 idées de soirées tranquilles à la maison, sans écrans.",
            "Dammi 5 idee per serate tranquille a casa, senza schermi.",
            "Dame 5 ideas de veladas tranquilas en casa, sin pantallas.",
            "Dá-me 5 ideias para noites tranquilas em casa, sem ecrãs.",
            "Gib mir 5 Ideen für ruhige Abende zu Hause, ohne Bildschirme.",
            "Adj 5 ötletet nyugodt otthoni estékhez, képernyők nélkül.",
        ],
    ),
    (
        "Copied to the clipboard",
        [
            "Copié dans le presse-papiers",
            "Copiato negli appunti",
            "Copiado al portapapeles",
            "Copiado para a área de transferência",
            "In die Zwischenablage kopiert",
            "A vágólapra másolva",
        ],
    ),
    (
        "Try again",
        [
            "Réessayer",
            "Riprova",
            "Reintentar",
            "Tentar de novo",
            "Erneut versuchen",
            "Újra",
        ],
    ),
    (
        "Dismiss",
        [
            "Fermer",
            "Chiudi",
            "Cerrar",
            "Fechar",
            "Schließen",
            "Bezárás",
        ],
    ),
    (
        "The model is loading…",
        [
            "Le modèle se charge…",
            "Il modello si sta caricando…",
            "El modelo se está cargando…",
            "O modelo está a carregar…",
            "Das Modell wird geladen…",
            "A modell betöltődik…",
        ],
    ),
    (
        "The model isn't running, so you can only read your conversations.",
        [
            "Le modèle ne tourne pas : vous pouvez seulement relire vos conversations.",
            "Il modello non è in esecuzione, quindi puoi solo rileggere le conversazioni.",
            "El modelo no está en marcha, así que solo puedes leer tus conversaciones.",
            "O modelo não está a funcionar, por isso só podes ler as tuas conversas.",
            "Das Modell läuft nicht, daher kannst du nur deine Unterhaltungen lesen.",
            "A modell nem fut, így csak a beszélgetéseidet olvashatod.",
        ],
    ),
    (
        "Start the model",
        [
            "Démarrer le modèle",
            "Avvia il modello",
            "Iniciar el modelo",
            "Iniciar o modelo",
            "Modell starten",
            "Modell indítása",
        ],
    ),
    (
        "Back",
        ["Retour", "Indietro", "Atrás", "Voltar", "Zurück", "Vissza"],
    ),
    (
        "Type here…  (Enter sends · Shift+Enter new line)",
        [
            "Écrivez ici…  (Entrée envoie · Maj+Entrée saut de ligne)",
            "Scrivi qui…  (Invio manda · Maiusc+Invio va a capo)",
            "Escribe aquí…  (Intro envía · Mayús+Intro nueva línea)",
            "Escreve aqui…  (Enter envia · Shift+Enter nova linha)",
            "Hier schreiben…  (Enter sendet · Umschalt+Enter neue Zeile)",
            "Írj ide…  (Enter küld · Shift+Enter új sor)",
        ],
    ),
    (
        "Stop",
        ["Arrêter", "Ferma", "Detener", "Parar", "Stopp", "Leállítás"],
    ),
    (
        "Send",
        ["Envoyer", "Invia", "Enviar", "Enviar", "Senden", "Küldés"],
    ),
    (
        "Style:",
        [
            "Style :", "Stile:", "Estilo:", "Estilo:", "Stil:", "Stílus:",
        ],
    ),
    (
        "Thinking:",
        [
            "Réflexion :",
            "Ragionamento:",
            "Razonamiento:",
            "Raciocínio:",
            "Denken:",
            "Gondolkodás:",
        ],
    ),
    (
        "I can't reach the server. Start the model and try again.",
        [
            "Je n'arrive pas à joindre le serveur. Démarrez le modèle et réessayez.",
            "Non riesco a raggiungere il server. Avvia il modello e riprova.",
            "No puedo conectar con el servidor. Inicia el modelo e inténtalo de nuevo.",
            "Não consigo contactar o servidor. Inicia o modelo e tenta de novo.",
            "Ich kann den Server nicht erreichen. Starte das Modell und versuche es erneut.",
            "Nem érem el a szervert. Indítsd el a modellt, és próbáld újra.",
        ],
    ),
    (
        "The server is busy or still loading. Try again in a few seconds.",
        [
            "Le serveur est occupé ou en cours de chargement. Réessayez dans quelques secondes.",
            "Il server è occupato o ancora in caricamento. Riprova tra qualche secondo.",
            "El servidor está ocupado o aún se está cargando. Inténtalo de nuevo en unos segundos.",
            "O servidor está ocupado ou ainda a carregar. Tenta de novo dentro de alguns segundos.",
            "Der Server ist ausgelastet oder lädt noch. Versuche es in ein paar Sekunden erneut.",
            "A szerver foglalt, vagy még töltődik. Próbáld újra néhány másodperc múlva.",
        ],
    ),
    (
        "The server wants an API key. Add it in Settings → Server.",
        [
            "Le serveur demande une clé API. Ajoutez-la dans Réglages → Serveur.",
            "Il server richiede una chiave API. Aggiungila in Impostazioni → Server.",
            "El servidor pide una clave de API. Añádela en Ajustes → Servidor.",
            "O servidor pede uma chave de API. Adiciona-a em Definições → Servidor.",
            "Der Server verlangt einen API-Schlüssel. Trage ihn unter Einstellungen → Server ein.",
            "A szerver API-kulcsot kér. Add meg a Beállítások → Szerver alatt.",
        ],
    ),
    (
        "This conversation got too long for the model. Start a new one (top left).",
        [
            "Cette conversation est trop longue pour le modèle. Commencez-en une nouvelle (en haut à gauche).",
            "Questa conversazione è diventata troppo lunga per il modello. Iniziane una nuova (in alto a sinistra).",
            "Esta conversación se ha hecho demasiado larga para el modelo. Empieza una nueva (arriba a la izquierda).",
            "Esta conversa ficou demasiado longa para o modelo. Começa uma nova (no canto superior esquerdo).",
            "Diese Unterhaltung ist für das Modell zu lang geworden. Beginne eine neue (oben links).",
            "Ez a beszélgetés túl hosszú lett a modellnek. Kezdj újat (bal felül).",
        ],
    ),
    (
        "The server answered with an error.",
        [
            "Le serveur a répondu par une erreur.",
            "Il server ha risposto con un errore.",
            "El servidor respondió con un error.",
            "O servidor respondeu com um erro.",
            "Der Server hat mit einem Fehler geantwortet.",
            "A szerver hibával válaszolt.",
        ],
    ),
    (
        "Step",
        ["Étape", "Passo", "Paso", "Passo", "Schritt", "Lépés"],
    ),
    (
        "Looking at what this computer has…",
        [
            "Analyse de cet ordinateur…",
            "Controllo cosa c'è su questo computer…",
            "Revisando qué tiene este ordenador…",
            "A ver o que este computador tem…",
            "Ich sehe nach, was dieser Computer hat…",
            "Megnézem, mi van ezen a számítógépen…",
        ],
    ),
    (
        "I want to connect to a server that's already running",
        [
            "Je veux me connecter à un serveur déjà en marche",
            "Voglio collegarmi a un server già in esecuzione",
            "Quiero conectarme a un servidor que ya está en marcha",
            "Quero ligar-me a um servidor que já está a funcionar",
            "Ich möchte mich mit einem bereits laufenden Server verbinden",
            "Egy már futó szerverhez szeretnék csatlakozni",
        ],
    ),
    (
        "First, you need Python",
        [
            "D'abord, il vous faut Python",
            "Prima serve Python",
            "Primero necesitas Python",
            "Primeiro, precisas de Python",
            "Zuerst brauchst du Python",
            "Először Pythonra van szükséged",
        ],
    ),
    (
        "colibri uses Python 3.10 or newer to set itself up. I couldn't find it on this computer.",
        [
            "colibri utilise Python 3.10 ou plus récent pour s'installer. Je ne l'ai pas trouvé sur cet ordinateur.",
            "colibri usa Python 3.10 o successivo per configurarsi. Non l'ho trovato su questo computer.",
            "colibri usa Python 3.10 o posterior para configurarse. No lo he encontrado en este ordenador.",
            "O colibri usa Python 3.10 ou superior para se configurar. Não o encontrei neste computador.",
            "colibri benötigt Python 3.10 oder neuer, um sich einzurichten. Ich habe es auf diesem Computer nicht gefunden.",
            "A colibri a beállításához Python 3.10-et vagy újabbat használ. Nem találtam meg ezen a számítógépen.",
        ],
    ),
    (
        "1. Download and install Python (free).",
        [
            "1. Téléchargez et installez Python (gratuit).",
            "1. Scarica e installa Python (gratuito).",
            "1. Descarga e instala Python (gratis).",
            "1. Transfere e instala o Python (grátis).",
            "1. Lade Python herunter und installiere es (kostenlos).",
            "1. Töltsd le és telepítsd a Pythont (ingyenes).",
        ],
    ),
    (
        "2. On Windows, tick “Add python.exe to PATH” in the installer.",
        [
            "2. Sous Windows, cochez « Add python.exe to PATH » dans l'installateur.",
            "2. Su Windows, spunta «Add python.exe to PATH» nel programma di installazione.",
            "2. En Windows, marca «Add python.exe to PATH» en el instalador.",
            "2. No Windows, assinala «Add python.exe to PATH» no instalador.",
            "2. Hake unter Windows im Installer «Add python.exe to PATH» an.",
            "2. Windowson jelöld be a telepítőben az «Add python.exe to PATH» opciót.",
        ],
    ),
    (
        "3. Come back here and press “Check again”.",
        [
            "3. Revenez ici et appuyez sur « Vérifier à nouveau ».",
            "3. Torna qui e premi «Controlla di nuovo».",
            "3. Vuelve aquí y pulsa «Comprobar de nuevo».",
            "3. Volta aqui e carrega em «Verificar de novo».",
            "3. Komm hierher zurück und drücke «Erneut prüfen».",
            "3. Gyere vissza ide, és nyomd meg az «Újra ellenőrzés» gombot.",
        ],
    ),
    (
        "Open the Python page",
        [
            "Ouvrir la page de Python",
            "Apri la pagina di Python",
            "Abrir la página de Python",
            "Abrir a página do Python",
            "Python-Seite öffnen",
            "A Python oldalának megnyitása",
        ],
    ),
    (
        "Check again",
        [
            "Vérifier à nouveau",
            "Controlla di nuovo",
            "Comprobar de nuevo",
            "Verificar de novo",
            "Erneut prüfen",
            "Újra ellenőrzés",
        ],
    ),
    (
        "Python is installed once. You never have to type a command.",
        [
            "Python s'installe une seule fois. Vous n'avez jamais à taper de commande.",
            "Python si installa una volta sola. Non devi mai scrivere comandi.",
            "Python se instala una sola vez. Nunca tienes que escribir comandos.",
            "O Python instala-se uma só vez. Nunca precisas de escrever comandos.",
            "Python wird nur einmal installiert. Du musst nie einen Befehl eintippen.",
            "A Pythont csak egyszer kell telepíteni. Soha nem kell parancsot gépelned.",
        ],
    ),
    (
        "Where is colibri?",
        [
            "Où est colibri ?",
            "Dov'è colibri?",
            "¿Dónde está colibri?",
            "Onde está o colibri?",
            "Wo ist colibri?",
            "Hol van a colibri?",
        ],
    ),
    (
        "colibri is the engine that runs models on your computer. I can download it for you, or you can point me to it.",
        [
            "colibri est le moteur qui fait tourner les modèles sur votre ordinateur. Je peux le télécharger pour vous, ou vous m'indiquez où il se trouve.",
            "colibri è il motore che esegue i modelli sul tuo computer. Posso scaricarlo io, oppure puoi indicarmi dov'è.",
            "colibri es el motor que ejecuta modelos en tu ordenador. Puedo descargarlo por ti, o puedes indicarme dónde está.",
            "O colibri é o motor que executa modelos no teu computador. Posso transferi-lo por ti, ou podes indicar-me onde está.",
            "colibri ist die Engine, die Modelle auf deinem Computer ausführt. Ich kann es für dich herunterladen, oder du zeigst mir, wo es liegt.",
            "A colibri az a motor, amely a modelleket a számítógépeden futtatja. Letölthetem helyetted, vagy megmutathatod, hol van.",
        ],
    ),
    (
        "The easy way",
        [
            "La voie facile",
            "La via facile",
            "La vía fácil",
            "A via fácil",
            "Der einfache Weg",
            "A könnyű út",
        ],
    ),
    (
        "I download colibri from GitHub into your home folder (~/colibri). It's small.",
        [
            "Je télécharge colibri depuis GitHub dans votre dossier personnel (~/colibri). Il est léger.",
            "Scarico colibri da GitHub nella tua cartella personale (~/colibri). È leggero.",
            "Descargo colibri de GitHub en tu carpeta personal (~/colibri). Es pequeño.",
            "Transfiro o colibri do GitHub para a tua pasta pessoal (~/colibri). É pequeno.",
            "Ich lade colibri von GitHub in deinen Home-Ordner (~/colibri) herunter. Es ist klein.",
            "Letöltöm a colibrit a GitHubról a saját mappádba (~/colibri). Kicsi.",
        ],
    ),
    (
        "Download colibri for me",
        [
            "Télécharger colibri pour moi",
            "Scarica colibri per me",
            "Descargar colibri por mí",
            "Transferir o colibri por mim",
            "colibri für mich herunterladen",
            "Töltsd le nekem a colibrit",
        ],
    ),
    (
        "I can't find the “git” program, so grab the ZIP from your browser, unzip it, then drag the folder into this window.",
        [
            "Je ne trouve pas le programme « git » : récupérez le ZIP depuis votre navigateur, décompressez-le, puis glissez le dossier dans cette fenêtre.",
            "Non trovo il programma «git»: scarica lo ZIP dal browser, decomprimilo e trascina la cartella in questa finestra.",
            "No encuentro el programa «git»: descarga el ZIP desde tu navegador, descomprímelo y arrastra la carpeta a esta ventana.",
            "Não encontro o programa «git»: obtém o ZIP no navegador, descompacta-o e arrasta a pasta para esta janela.",
            "Ich finde das Programm «git» nicht. Hole dir die ZIP-Datei im Browser, entpacke sie und ziehe den Ordner in dieses Fenster.",
            "Nem találom a «git» programot, ezért töltsd le a ZIP-et a böngészőből, csomagold ki, és húzd a mappát ebbe az ablakba.",
        ],
    ),
    (
        "Open the ZIP download",
        [
            "Ouvrir le téléchargement ZIP",
            "Apri il download ZIP",
            "Abrir la descarga del ZIP",
            "Abrir a transferência do ZIP",
            "ZIP-Download öffnen",
            "A ZIP letöltésének megnyitása",
        ],
    ),
    (
        "I already have it",
        [
            "Je l'ai déjà",
            "Ce l'ho già",
            "Ya lo tengo",
            "Já o tenho",
            "Ich habe es schon",
            "Már megvan",
        ],
    ),
    (
        "Drag the colibri folder into this window, or paste its path.",
        [
            "Glissez le dossier colibri dans cette fenêtre, ou collez son chemin.",
            "Trascina la cartella colibri in questa finestra oppure incolla il suo percorso.",
            "Arrastra la carpeta de colibri a esta ventana o pega su ruta.",
            "Arrasta a pasta do colibri para esta janela ou cola o respetivo caminho.",
            "Ziehe den colibri-Ordner in dieses Fenster oder füge seinen Pfad ein.",
            "Húzd a colibri mappát ebbe az ablakba, vagy illeszd be az elérési útját.",
        ],
    ),
    (
        "Use this folder",
        [
            "Utiliser ce dossier",
            "Usa questa cartella",
            "Usar esta carpeta",
            "Usar esta pasta",
            "Diesen Ordner verwenden",
            "Ennek a mappának a használata",
        ],
    ),
    (
        "Search again automatically",
        [
            "Chercher à nouveau automatiquement",
            "Cerca di nuovo automaticamente",
            "Buscar de nuevo automáticamente",
            "Procurar de novo automaticamente",
            "Automatisch erneut suchen",
            "Automatikus újrakeresés",
        ],
    ),
    (
        "colibri is a separate project (Apache-2.0) by JustVugg. This app never changes its code: it only runs it and talks to it.",
        [
            "colibri est un projet distinct (Apache-2.0) de JustVugg. Cette application ne modifie jamais son code : elle le lance et dialogue avec lui.",
            "colibri è un progetto separato (Apache-2.0) di JustVugg. Questa app non ne modifica mai il codice: lo esegue e ci dialoga.",
            "colibri es un proyecto aparte (Apache-2.0) de JustVugg. Esta aplicación nunca cambia su código: solo lo ejecuta y se comunica con él.",
            "O colibri é um projeto separado (Apache-2.0) de JustVugg. Esta aplicação nunca altera o seu código: apenas o executa e comunica com ele.",
            "colibri ist ein eigenständiges Projekt (Apache-2.0) von JustVugg. Diese App ändert nie dessen Code: Sie führt es nur aus und spricht mit ihm.",
            "A colibri JustVugg különálló projektje (Apache-2.0). Ez az alkalmazás soha nem módosítja a kódját: csak futtatja, és kommunikál vele.",
        ],
    ),
    (
        "Choose a model",
        [
            "Choisir un modèle",
            "Scegli un modello",
            "Elige un modelo",
            "Escolhe um modelo",
            "Modell wählen",
            "Modell kiválasztása",
        ],
    ),
    (
        "A model is the “brain” that answers. I've marked what suits this computer.",
        [
            "Un modèle est le « cerveau » qui répond. J'ai marqué ce qui convient à cet ordinateur.",
            "Un modello è il «cervello» che risponde. Ho segnato quelli adatti a questo computer.",
            "Un modelo es el «cerebro» que responde. He marcado lo que encaja con este ordenador.",
            "Um modelo é o «cérebro» que responde. Marquei o que se adequa a este computador.",
            "Ein Modell ist das «Gehirn», das antwortet. Ich habe markiert, was zu diesem Computer passt.",
            "A modell az «agy», amely válaszol. Megjelöltem, mi illik ehhez a számítógéphez.",
        ],
    ),
    (
        "no dedicated graphics card",
        [
            "pas de carte graphique dédiée",
            "nessuna scheda grafica dedicata",
            "sin tarjeta gráfica dedicada",
            "sem placa gráfica dedicada",
            "keine dedizierte Grafikkarte",
            "nincs dedikált videókártya",
        ],
    ),
    (
        "Your computer",
        [
            "Votre ordinateur",
            "Il tuo computer",
            "Tu ordenador",
            "O teu computador",
            "Dein Computer",
            "A számítógéped",
        ],
    ),
    ("CPU", ["Processeur", "CPU", "CPU", "CPU", "CPU", "CPU"]),
    (
        "free",
        ["libres", "liberi", "libres", "livres", "frei", "szabad"],
    ),
    (
        "Disk",
        ["Disque", "Disco", "Disco", "Disco", "Festplatte", "Lemez"],
    ),
    (
        "Measuring your computer…",
        [
            "Analyse de votre ordinateur…",
            "Sto misurando il tuo computer…",
            "Midiendo tu ordenador…",
            "A medir o teu computador…",
            "Dein Computer wird vermessen…",
            "A számítógéped felmérése…",
        ],
    ),
    (
        "Reading the model list…",
        [
            "Lecture de la liste des modèles…",
            "Lettura dell'elenco dei modelli…",
            "Leyendo la lista de modelos…",
            "A ler a lista de modelos…",
            "Modellliste wird gelesen…",
            "A modelllista beolvasása…",
        ],
    ),
    (
        "I couldn't read the model list.",
        [
            "Je n'ai pas pu lire la liste des modèles.",
            "Non sono riuscito a leggere l'elenco dei modelli.",
            "No pude leer la lista de modelos.",
            "Não consegui ler a lista de modelos.",
            "Ich konnte die Modellliste nicht lesen.",
            "Nem tudtam beolvasni a modelllistát.",
        ],
    ),
    (
        "No model fits right now.",
        [
            "Aucun modèle ne convient pour l'instant.",
            "Al momento nessun modello è adatto.",
            "Ningún modelo encaja ahora mismo.",
            "Nenhum modelo serve neste momento.",
            "Im Moment passt kein Modell.",
            "Jelenleg egyetlen modell sem fér el.",
        ],
    ),
    (
        "Free some disk space or close memory-hungry programs, then measure again. Below you can see why each one doesn't fit.",
        [
            "Libérez de l'espace disque ou fermez des programmes gourmands en mémoire, puis mesurez à nouveau. Ci-dessous, vous voyez pourquoi chacun ne convient pas.",
            "Libera spazio su disco o chiudi i programmi che usano molta memoria, poi misura di nuovo. Qui sotto vedi perché ciascuno non va bene.",
            "Libera espacio en disco o cierra programas que gasten mucha memoria y vuelve a medir. Abajo ves por qué no encaja cada uno.",
            "Liberta espaço em disco ou fecha programas que gastem muita memória e mede de novo. Abaixo vês porque é que cada um não serve.",
            "Gib Speicherplatz frei oder schließe speicherhungrige Programme und miss dann erneut. Unten siehst du, warum jedes nicht passt.",
            "Szabadíts fel lemezterületet, vagy zárj be memóriaigényes programokat, majd mérj újra. Lent látod, miért nem fér el az egyes modell.",
        ],
    ),
    (
        "Models that don't fit right now",
        [
            "Modèles qui ne conviennent pas pour l'instant",
            "Modelli non adatti al momento",
            "Modelos que no encajan ahora mismo",
            "Modelos que não servem neste momento",
            "Modelle, die gerade nicht passen",
            "Jelenleg nem elférő modellek",
        ],
    ),
    (
        "Measure again",
        [
            "Mesurer à nouveau",
            "Misura di nuovo",
            "Medir de nuevo",
            "Medir de novo",
            "Erneut messen",
            "Mérés újra",
        ],
    ),
    (
        "Back to the installed model",
        [
            "Retour au modèle installé",
            "Torna al modello installato",
            "Volver al modelo instalado",
            "Voltar ao modelo instalado",
            "Zurück zum installierten Modell",
            "Vissza a telepített modellhez",
        ],
    ),
    (
        "Recommended for you",
        [
            "Recommandé pour vous",
            "Consigliato per te",
            "Recomendado para ti",
            "Recomendado para ti",
            "Für dich empfohlen",
            "Neked ajánlott",
        ],
    ),
    (
        "Download:",
        [
            "Téléchargement :",
            "Download:",
            "Descarga:",
            "Transferência:",
            "Download:",
            "Letöltés:",
        ],
    ),
    (
        "min",
        [
            "minimum",
            "minimo",
            "mínimo",
            "mínimo",
            "mindestens",
            "minimum",
        ],
    ),
    (
        "recommended",
        [
            "recommandé",
            "consigliato",
            "recomendado",
            "recomendado",
            "empfohlen",
            "ajánlott",
        ],
    ),
    (
        "Licence:",
        [
            "Licence :",
            "Licenza:",
            "Licencia:",
            "Licença:",
            "Lizenz:",
            "Licenc:",
        ],
    ),
    (
        "This model makes pictures; this app is for conversation.",
        [
            "Ce modèle crée des images ; cette application sert à converser.",
            "Questo modello crea immagini; questa app serve per conversare.",
            "Este modelo crea imágenes; esta aplicación es para conversar.",
            "Este modelo cria imagens; esta aplicação serve para conversar.",
            "Dieses Modell erzeugt Bilder; diese App ist für Unterhaltungen gedacht.",
            "Ez a modell képeket készít; ez az alkalmazás beszélgetésre való.",
        ],
    ),
    (
        "Choose this model",
        [
            "Choisir ce modèle",
            "Scegli questo modello",
            "Elegir este modelo",
            "Escolher este modelo",
            "Dieses Modell wählen",
            "Ennek a modellnek a kiválasztása",
        ],
    ),
    (
        "Will download",
        [
            "Va télécharger",
            "Scaricherà",
            "Descargará",
            "Vai transferir",
            "Lädt herunter:",
            "Letöltendő:",
        ],
    ),
    ("into", ["dans", "in", "en", "para", "nach", "ide:"]),
    (
        "Not enough free disk space for this model.",
        [
            "Pas assez d'espace disque libre pour ce modèle.",
            "Spazio libero su disco insufficiente per questo modello.",
            "No hay espacio libre en disco suficiente para este modelo.",
            "Não há espaço livre em disco suficiente para este modelo.",
            "Nicht genug freier Speicherplatz für dieses Modell.",
            "Nincs elég szabad lemezterület ehhez a modellhez.",
        ],
    ),
    (
        "You can pause and resume at any time; it keeps going even if you close this window.",
        [
            "Vous pouvez mettre en pause et reprendre à tout moment ; le téléchargement continue même si vous fermez cette fenêtre.",
            "Puoi mettere in pausa e riprendere quando vuoi; continua anche se chiudi questa finestra.",
            "Puedes pausar y reanudar cuando quieras; sigue aunque cierres esta ventana.",
            "Podes pausar e retomar quando quiseres; continua mesmo que feches esta janela.",
            "Du kannst jederzeit pausieren und fortsetzen; es läuft weiter, auch wenn du dieses Fenster schließt.",
            "Bármikor szüneteltetheted és folytathatod; az ablak bezárása után is tovább megy.",
        ],
    ),
    (
        "The model comes from Hugging Face and has its own licence.",
        [
            "Le modèle vient de Hugging Face et a sa propre licence.",
            "Il modello proviene da Hugging Face e ha una sua licenza.",
            "El modelo viene de Hugging Face y tiene su propia licencia.",
            "O modelo vem do Hugging Face e tem a sua própria licença.",
            "Das Modell stammt von Hugging Face und hat eine eigene Lizenz.",
            "A modell a Hugging Face-ről származik, és saját licence van.",
        ],
    ),
    (
        "See the model page",
        [
            "Voir la page du modèle",
            "Vedi la pagina del modello",
            "Ver la página del modelo",
            "Ver a página do modelo",
            "Modellseite ansehen",
            "A modell oldalának megtekintése",
        ],
    ),
    (
        "I've seen the model's licence and agree to it",
        [
            "J'ai lu la licence du modèle et je l'accepte",
            "Ho letto la licenza del modello e l'accetto",
            "He visto la licencia del modelo y la acepto",
            "Vi a licença do modelo e aceito-a",
            "Ich habe die Lizenz des Modells gelesen und stimme zu",
            "Láttam a modell licencét, és elfogadom",
        ],
    ),
    (
        "Download and install",
        [
            "Télécharger et installer",
            "Scarica e installa",
            "Descargar e instalar",
            "Transferir e instalar",
            "Herunterladen und installieren",
            "Letöltés és telepítés",
        ],
    ),
    (
        "Cancel",
        [
            "Annuler",
            "Annulla",
            "Cancelar",
            "Cancelar",
            "Abbrechen",
            "Mégse",
        ],
    ),
    (
        "Runs well",
        [
            "Fonctionne bien",
            "Funziona bene",
            "Funciona bien",
            "Funciona bem",
            "Läuft gut",
            "Jól fut",
        ],
    ),
    (
        "Runs, but tight",
        [
            "Fonctionne, mais juste",
            "Funziona, ma al limite",
            "Funciona, pero justo",
            "Funciona, mas ao limite",
            "Läuft, aber knapp",
            "Fut, de épphogy",
        ],
    ),
    (
        "Not enough RAM",
        [
            "Pas assez de RAM",
            "RAM insufficiente",
            "RAM insuficiente",
            "RAM insuficiente",
            "Nicht genug RAM",
            "Nincs elég RAM",
        ],
    ),
    (
        "Not enough disk space",
        [
            "Pas assez d'espace disque",
            "Spazio su disco insufficiente",
            "Espacio en disco insuficiente",
            "Espaço em disco insuficiente",
            "Nicht genug Speicherplatz",
            "Nincs elég lemezterület",
        ],
    ),
    (
        "Not supported here",
        [
            "Non pris en charge ici",
            "Non supportato qui",
            "No compatible aquí",
            "Não suportado aqui",
            "Hier nicht unterstützt",
            "Itt nem támogatott",
        ],
    ),
    (
        "Reads parts of the model from your SSD: slower, but very capable. A fast SSD helps a lot.",
        [
            "Lit des parties du modèle sur votre SSD : plus lent, mais très capable. Un SSD rapide aide beaucoup.",
            "Legge parti del modello dal tuo SSD: più lento, ma molto capace. Un SSD veloce aiuta molto.",
            "Lee partes del modelo desde tu SSD: más lento, pero muy capaz. Un SSD rápido ayuda mucho.",
            "Lê partes do modelo a partir do teu SSD: mais lento, mas muito capaz. Um SSD rápido ajuda bastante.",
            "Liest Teile des Modells von deiner SSD: langsamer, aber sehr leistungsfähig. Eine schnelle SSD hilft sehr.",
            "A modell egyes részeit az SSD-dről olvassa: lassabb, de nagyon képes. Egy gyors SSD sokat segít.",
        ],
    ),
    (
        "Runs from RAM: faster answers.",
        [
            "Fonctionne depuis la RAM : réponses plus rapides.",
            "Funziona dalla RAM: risposte più rapide.",
            "Funciona desde la RAM: respuestas más rápidas.",
            "Funciona a partir da RAM: respostas mais rápidas.",
            "Läuft aus dem RAM: schnellere Antworten.",
            "A RAM-ból fut: gyorsabb válaszok.",
        ],
    ),
    (
        "General conversation with thinking and tools. A good place to start.",
        [
            "Conversation générale avec réflexion et outils. Un bon point de départ.",
            "Conversazione generale con ragionamento e strumenti. Un buon punto di partenza.",
            "Conversación general con razonamiento y herramientas. Un buen punto de partida.",
            "Conversa geral com raciocínio e ferramentas. Um bom ponto de partida.",
            "Allgemeine Unterhaltung mit Denken und Werkzeugen. Ein guter Einstieg.",
            "Általános beszélgetés gondolkodással és eszközökkel. Jó kiindulópont.",
        ],
    ),
    (
        "Specialised in code and tools, without the “thinking” step. Handles everyday questions too.",
        [
            "Spécialisé dans le code et les outils, sans étape de « réflexion ». Répond aussi aux questions courantes.",
            "Specializzato in codice e strumenti, senza il passaggio di «ragionamento». Risponde anche alle domande di tutti i giorni.",
            "Especializado en código y herramientas, sin el paso de «razonamiento». También resuelve preguntas cotidianas.",
            "Especializado em código e ferramentas, sem o passo de «raciocínio». Também responde às perguntas do dia a dia.",
            "Spezialisiert auf Code und Werkzeuge, ohne den «Denk»-Schritt. Beantwortet auch Alltagsfragen.",
            "Kódra és eszközökre specializálva, «gondolkodási» lépés nélkül. A mindennapi kérdésekre is válaszol.",
        ],
    ),
    (
        "Getting everything ready",
        [
            "Tout se prépare",
            "Preparazione in corso",
            "Preparándolo todo",
            "A preparar tudo",
            "Alles wird vorbereitet",
            "Minden előkészítése",
        ],
    ),
    (
        "Preparing the engine for your computer…",
        [
            "Préparation du moteur pour votre ordinateur…",
            "Preparo il motore per il tuo computer…",
            "Preparando el motor para tu ordenador…",
            "A preparar o motor para o teu computador…",
            "Die Engine wird für deinen Computer vorbereitet…",
            "A motor előkészítése a számítógépedhez…",
        ],
    ),
    (
        "Downloading the model…",
        [
            "Téléchargement du modèle…",
            "Download del modello…",
            "Descargando el modelo…",
            "A transferir o modelo…",
            "Modell wird heruntergeladen…",
            "A modell letöltése…",
        ],
    ),
    (
        "Starting…",
        [
            "Démarrage…",
            "Avvio…",
            "Iniciando…",
            "A iniciar…",
            "Startet…",
            "Indítás…",
        ],
    ),
    (
        "Working…",
        [
            "En cours…",
            "In corso…",
            "Trabajando…",
            "A trabalhar…",
            "Wird bearbeitet…",
            "Dolgozom…",
        ],
    ),
    ("of", ["sur", "di", "de", "de", "von", "/"]),
    (
        "Feel free to do something else. If you close this window, the download keeps going in the background.",
        [
            "Faites autre chose si vous voulez. Si vous fermez cette fenêtre, le téléchargement continue en arrière-plan.",
            "Puoi fare altro. Se chiudi questa finestra, il download continua in background.",
            "Haz otra cosa si quieres. Si cierras esta ventana, la descarga sigue en segundo plano.",
            "Fica à vontade para fazer outra coisa. Se fechares esta janela, a transferência continua em segundo plano.",
            "Mach ruhig etwas anderes. Wenn du dieses Fenster schließt, läuft der Download im Hintergrund weiter.",
            "Nyugodtan csinálj mást. Ha bezárod ezt az ablakot, a letöltés a háttérben folytatódik.",
        ],
    ),
    (
        "Pause",
        ["Pause", "Pausa", "Pausar", "Pausar", "Pause", "Szünet"],
    ),
    (
        "An optional package would make this faster",
        [
            "Un paquet facultatif rendrait tout plus rapide",
            "Un pacchetto facoltativo renderebbe tutto più veloce",
            "Un paquete opcional lo haría más rápido",
            "Um pacote opcional tornaria tudo mais rápido",
            "Ein optionales Paket würde alles schneller machen",
            "Egy opcionális csomag mindent gyorsabbá tenne",
        ],
    ),
    (
        "To use your graphics card, run this command in a terminal (I never install system packages myself), then press “Resume”.",
        [
            "Pour utiliser votre carte graphique, exécutez cette commande dans un terminal (je n'installe jamais de paquets système moi-même), puis appuyez sur « Reprendre ».",
            "Per usare la scheda grafica, esegui questo comando in un terminale (non installo mai pacchetti di sistema da solo), poi premi «Riprendi».",
            "Para usar tu tarjeta gráfica, ejecuta este comando en un terminal (yo nunca instalo paquetes del sistema), y luego pulsa «Reanudar».",
            "Para usar a placa gráfica, executa este comando num terminal (eu nunca instalo pacotes do sistema), e depois carrega em «Retomar».",
            "Um deine Grafikkarte zu nutzen, führe diesen Befehl in einem Terminal aus (ich installiere nie selbst Systempakete) und drücke dann «Fortsetzen».",
            "A videókártyád használatához futtasd ezt a parancsot egy terminálban (rendszercsomagokat soha nem telepítek magam), majd nyomd meg a «Folytatás» gombot.",
        ],
    ),
    (
        "Added file «{}»",
        [
            "Fichier «{}» ajouté",
            "File «{}» aggiunto",
            "Archivo «{}» añadido",
            "Ficheiro «{}» adicionado",
            "Datei «{}» hinzugefügt",
            "A(z) «{}» fájl hozzáadva",
        ],
    ),
    (
        "«{}» is not a text file or is too large (max 256 KB)",
        [
            "«{}» n'est pas un fichier texte ou est trop volumineux (max 256 Ko)",
            "«{}» non è un file di testo o è troppo grande (max 256 KB)",
            "«{}» no es un archivo de texto o es demasiado grande (máx. 256 KB)",
            "«{}» não é um ficheiro de texto ou é demasiado grande (máx. 256 KB)",
            "«{}» ist keine Textdatei oder zu groß (max. 256 KB)",
            "A(z) «{}» nem szövegfájl, vagy túl nagy (legfeljebb 256 KB)",
        ],
    ),
    (
        "Saved to {}",
        [
            "Enregistré dans {}",
            "Salvato in {}",
            "Guardado en {}",
            "Guardado em {}",
            "Gespeichert unter {}",
            "Mentve ide: {}",
        ],
    ),
    (
        "Could not save the file",
        [
            "Impossible d'enregistrer le fichier",
            "Impossibile salvare il file",
            "No se pudo guardar el archivo",
            "Não foi possível guardar o ficheiro",
            "Die Datei konnte nicht gespeichert werden",
            "A fájl mentése nem sikerült",
        ],
    ),
    (
        "Save as file",
        [
            "Enregistrer",
            "Salva",
            "Guardar",
            "Guardar",
            "Speichern",
            "Mentés",
        ],
    ),
    (
        "Save the conversation as a .md file in Downloads",
        [
            "Enregistrer la conversation en fichier .md dans Téléchargements",
            "Salva la conversazione come file .md in Download",
            "Guardar la conversación como archivo .md en Descargas",
            "Guardar a conversa como ficheiro .md em Transferências",
            "Unterhaltung als .md-Datei im Downloads-Ordner speichern",
            "A beszélgetés mentése .md fájlként a Letöltések mappába",
        ],
    ),
    (
        "Search… (Ctrl+K)",
        [
            "Rechercher… (Ctrl+K)",
            "Cerca… (Ctrl+K)",
            "Buscar… (Ctrl+K)",
            "Pesquisar… (Ctrl+K)",
            "Suchen… (Strg+K)",
            "Keresés… (Ctrl+K)",
        ],
    ),
    (
        "No conversations match.",
        [
            "Aucune conversation ne correspond.",
            "Nessuna conversazione corrisponde.",
            "Ninguna conversación coincide.",
            "Nenhuma conversa corresponde.",
            "Keine Unterhaltung passt.",
            "Nincs egyező beszélgetés.",
        ],
    ),
    (
        "Rename",
        [
            "Renommer",
            "Rinomina",
            "Renombrar",
            "Mudar o nome",
            "Umbenennen",
            "Átnevezés",
        ],
    ),
    (
        "New name",
        [
            "Nouveau nom",
            "Nuovo nome",
            "Nuevo nombre",
            "Novo nome",
            "Neuer Name",
            "Új név",
        ],
    ),
    (
        "Edit",
        [
            "Modifier",
            "Modifica",
            "Editar",
            "Editar",
            "Bearbeiten",
            "Szerkesztés",
        ],
    ),
    (
        "Copy",
        ["Copier", "Copia", "Copiar", "Copiar", "Kopieren", "Másolás"],
    ),
    (
        "Copied",
        [
            "Copié",
            "Copiato",
            "Copiado",
            "Copiado",
            "Kopiert",
            "Kimásolva",
        ],
    ),
    (
        "Technical details",
        [
            "Détails techniques",
            "Dettagli tecnici",
            "Detalles técnicos",
            "Detalhes técnicos",
            "Technische Details",
            "Műszaki részletek",
        ],
    ),
    (
        "The setup is paused",
        [
            "L'installation est en pause",
            "L'installazione è in pausa",
            "La instalación está en pausa",
            "A instalação está em pausa",
            "Die Einrichtung ist pausiert",
            "A telepítés szünetel",
        ],
    ),
    (
        "The setup stopped",
        [
            "L'installation s'est arrêtée",
            "L'installazione si è fermata",
            "La instalación se detuvo",
            "A instalação parou",
            "Die Einrichtung wurde angehalten",
            "A telepítés megállt",
        ],
    ),
    (
        "Nothing is lost: resuming continues where it stopped.",
        [
            "Rien n'est perdu : la reprise continue là où elle s'est arrêtée.",
            "Non si è perso nulla: riprendendo si continua da dove si era fermata.",
            "No se ha perdido nada: al reanudar continúa donde se quedó.",
            "Nada se perdeu: ao retomar, continua de onde parou.",
            "Nichts geht verloren: Beim Fortsetzen geht es dort weiter, wo es aufgehört hat.",
            "Semmi sem veszett el: a folytatás ott megy tovább, ahol abbamaradt.",
        ],
    ),
    (
        "Resume",
        [
            "Reprendre",
            "Riprendi",
            "Reanudar",
            "Retomar",
            "Fortsetzen",
            "Folytatás",
        ],
    ),
    (
        "Choose another model",
        [
            "Choisir un autre modèle",
            "Scegli un altro modello",
            "Elegir otro modelo",
            "Escolher outro modelo",
            "Anderes Modell wählen",
            "Másik modell kiválasztása",
        ],
    ),
    (
        "Show the log",
        [
            "Afficher le journal",
            "Mostra il registro",
            "Mostrar el registro",
            "Mostrar o registo",
            "Protokoll anzeigen",
            "Napló megjelenítése",
        ],
    ),
    (
        "Everything is installed",
        [
            "Tout est installé",
            "Tutto è installato",
            "Todo está instalado",
            "Está tudo instalado",
            "Alles ist installiert",
            "Minden telepítve van",
        ],
    ),
    (
        "The model has to be loaded into memory before your first chat. Big models can take a few minutes.",
        [
            "Le modèle doit être chargé en mémoire avant votre première conversation. Les gros modèles peuvent prendre quelques minutes.",
            "Il modello va caricato in memoria prima della prima chat. I modelli grandi possono richiedere qualche minuto.",
            "El modelo debe cargarse en memoria antes de tu primera conversación. Los modelos grandes pueden tardar unos minutos.",
            "O modelo tem de ser carregado em memória antes da tua primeira conversa. Os modelos grandes podem demorar alguns minutos.",
            "Das Modell muss vor deinem ersten Chat in den Speicher geladen werden. Große Modelle können einige Minuten brauchen.",
            "A modellt be kell tölteni a memóriába az első csevegés előtt. A nagy modellek néhány percig is eltarthatnak.",
        ],
    ),
    (
        "Start by itself next time",
        [
            "Démarrer tout seul la prochaine fois",
            "Avvia da solo la prossima volta",
            "Iniciar solo la próxima vez",
            "Iniciar sozinho da próxima vez",
            "Nächstes Mal selbst starten",
            "Legközelebb induljon el magától",
        ],
    ),
    (
        "See saved conversations",
        [
            "Voir les conversations enregistrées",
            "Vedi le conversazioni salvate",
            "Ver conversaciones guardadas",
            "Ver conversas guardadas",
            "Gespeicherte Unterhaltungen ansehen",
            "Mentett beszélgetések megtekintése",
        ],
    ),
    (
        "Loading the model",
        [
            "Chargement du modèle",
            "Caricamento del modello",
            "Cargando el modelo",
            "A carregar o modelo",
            "Modell wird geladen",
            "A modell betöltése",
        ],
    ),
    (
        "Loading… for",
        [
            "Chargement depuis",
            "Caricamento da",
            "Cargando desde hace",
            "A carregar há",
            "Lädt seit",
            "Betöltés…",
        ],
    ),
    (
        "The model is being read from disk into memory. The first start can take longer. The chat opens by itself as soon as it's ready.",
        [
            "Le modèle est lu depuis le disque vers la mémoire. Le premier démarrage peut être plus long. La conversation s'ouvre toute seule dès que c'est prêt.",
            "Il modello viene letto dal disco in memoria. Il primo avvio può richiedere più tempo. La chat si apre da sola appena è pronto.",
            "El modelo se está leyendo del disco a la memoria. El primer inicio puede tardar más. El chat se abre solo en cuanto esté listo.",
            "O modelo está a ser lido do disco para a memória. O primeiro arranque pode demorar mais. A conversa abre sozinha assim que estiver pronto.",
            "Das Modell wird von der Festplatte in den Speicher gelesen. Der erste Start kann länger dauern. Der Chat öffnet sich von selbst, sobald es fertig ist.",
            "A modellt a lemezről olvassa be a memóriába. Az első indítás tovább tarthat. A csevegés magától megnyílik, amint kész.",
        ],
    ),
    (
        "Meanwhile, see saved conversations",
        [
            "En attendant, voir les conversations enregistrées",
            "Intanto, vedi le conversazioni salvate",
            "Mientras tanto, ver conversaciones guardadas",
            "Entretanto, ver conversas guardadas",
            "Sieh dir in der Zwischenzeit gespeicherte Unterhaltungen an",
            "Addig nézd meg a mentett beszélgetéseket",
        ],
    ),
    (
        "About",
        [
            "À propos",
            "Informazioni",
            "Acerca de",
            "Acerca de",
            "Über",
            "Névjegy",
        ],
    ),
    (
        "Settings",
        [
            "Réglages",
            "Impostazioni",
            "Ajustes",
            "Definições",
            "Einstellungen",
            "Beállítások",
        ],
    ),
    (
        "Stop the model",
        [
            "Arrêter le modèle",
            "Ferma il modello",
            "Detener el modelo",
            "Parar o modelo",
            "Modell beenden",
            "Modell leállítása",
        ],
    ),
    (
        "Another model",
        [
            "Autre modèle",
            "Altro modello",
            "Otro modelo",
            "Outro modelo",
            "Anderes Modell",
            "Másik modell",
        ],
    ),
    (
        "Demo (no real model)",
        [
            "Démo (sans vrai modèle)",
            "Demo (senza modello reale)",
            "Demo (sin modelo real)",
            "Demonstração (sem modelo real)",
            "Demo (kein echtes Modell)",
            "Demó (valódi modell nélkül)",
        ],
    ),
    (
        "Ready",
        ["Prêt", "Pronto", "Listo", "Pronto", "Bereit", "Kész"],
    ),
    (
        "Loading the model…",
        [
            "Chargement du modèle…",
            "Caricamento del modello…",
            "Cargando el modelo…",
            "A carregar o modelo…",
            "Modell wird geladen…",
            "A modell betöltése…",
        ],
    ),
    (
        "Installing…",
        [
            "Installation…",
            "Installazione…",
            "Instalando…",
            "A instalar…",
            "Wird installiert…",
            "Telepítés…",
        ],
    ),
    (
        "Model not running",
        [
            "Modèle arrêté",
            "Modello non in esecuzione",
            "Modelo detenido",
            "Modelo parado",
            "Modell läuft nicht",
            "A modell nem fut",
        ],
    ),
    (
        "Everything stays on this computer",
        [
            "Tout reste sur cet ordinateur",
            "Tutto resta su questo computer",
            "Todo se queda en este ordenador",
            "Tudo fica neste computador",
            "Alles bleibt auf diesem Computer",
            "Minden ezen a számítógépen marad",
        ],
    ),
    (
        "Remote server",
        [
            "Serveur distant",
            "Server remoto",
            "Servidor remoto",
            "Servidor remoto",
            "Entfernter Server",
            "Távoli szerver",
        ],
    ),
    (
        "context",
        [
            "contexte",
            "contesto",
            "contexto",
            "contexto",
            "Kontext",
            "kontextus",
        ],
    ),
    (
        "tokens",
        ["jetons", "token", "tokens", "tokens", "Tokens", "token"],
    ),
    (
        "Language",
        ["Langue", "Lingua", "Idioma", "Idioma", "Sprache", "Nyelv"],
    ),
    (
        "Appearance",
        [
            "Apparence",
            "Aspetto",
            "Apariencia",
            "Aspeto",
            "Darstellung",
            "Megjelenés",
        ],
    ),
    (
        "Light",
        ["Clair", "Chiaro", "Claro", "Claro", "Hell", "Világos"],
    ),
    (
        "Dark",
        ["Sombre", "Scuro", "Oscuro", "Escuro", "Dunkel", "Sötét"],
    ),
    (
        "High contrast",
        [
            "Contraste élevé",
            "Alto contrasto",
            "Alto contraste",
            "Contraste elevado",
            "Hoher Kontrast",
            "Nagy kontraszt",
        ],
    ),
    (
        "Text size",
        [
            "Taille du texte",
            "Dimensione del testo",
            "Tamaño del texto",
            "Tamanho do texto",
            "Textgröße",
            "Szövegméret",
        ],
    ),
    (
        "Reset",
        [
            "Réinitialiser",
            "Ripristina",
            "Restablecer",
            "Repor",
            "Zurücksetzen",
            "Visszaállítás",
        ],
    ),
    (
        "How the model answers",
        [
            "Comment le modèle répond",
            "Come risponde il modello",
            "Cómo responde el modelo",
            "Como o modelo responde",
            "Wie das Modell antwortet",
            "Hogyan válaszol a modell",
        ],
    ),
    (
        "Show the model's “thinking”, when there is any",
        [
            "Afficher la « réflexion » du modèle, quand il y en a",
            "Mostra il «ragionamento» del modello, quando c'è",
            "Mostrar el «razonamiento» del modelo, cuando lo haya",
            "Mostrar o «raciocínio» do modelo, quando existir",
            "«Denkprozess» des Modells anzeigen, falls vorhanden",
            "A modell «gondolkodásának» megjelenítése, ha van",
        ],
    ),
    (
        "Standing instructions (optional), e.g. “Answer briefly”:",
        [
            "Instructions permanentes (facultatif), p. ex. « Réponds brièvement » :",
            "Istruzioni permanenti (facoltative), ad es. «Rispondi in modo breve»:",
            "Instrucciones permanentes (opcional), p. ej. «Responde brevemente»:",
            "Instruções permanentes (opcional), p. ex. «Responde de forma breve»:",
            "Dauerhafte Anweisungen (optional), z. B. «Antworte kurz»:",
            "Állandó utasítások (nem kötelező), pl. «Válaszolj röviden»:",
        ],
    ),
    (
        "Limit the length of an answer",
        [
            "Limiter la longueur d'une réponse",
            "Limita la lunghezza di una risposta",
            "Limitar la longitud de una respuesta",
            "Limitar o tamanho de uma resposta",
            "Länge einer Antwort begrenzen",
            "A válasz hosszának korlátozása",
        ],
    ),
    (
        "Server",
        [
            "Serveur", "Server", "Servidor", "Servidor", "Server", "Szerver",
        ],
    ),
    (
        "Leave the default to use the colibri installed from here. Change it for any OpenAI-compatible server (http:// only).",
        [
            "Laissez la valeur par défaut pour utiliser le colibri installé ici. Modifiez-la pour tout serveur compatible OpenAI (http:// uniquement).",
            "Lascia il valore predefinito per usare il colibri installato da qui. Cambialo per qualsiasi server compatibile con OpenAI (solo http://).",
            "Deja el valor por defecto para usar el colibri instalado desde aquí. Cámbialo para cualquier servidor compatible con OpenAI (solo http://).",
            "Deixa o valor predefinido para usar o colibri instalado a partir daqui. Altera-o para qualquer servidor compatível com OpenAI (apenas http://).",
            "Lass den Standardwert, um das von hier installierte colibri zu nutzen. Ändere ihn für jeden OpenAI-kompatiblen Server (nur http://).",
            "Hagyd az alapértéket az innen telepített colibri használatához. Módosítsd bármely OpenAI-kompatibilis szerverhez (csak http://).",
        ],
    ),
    (
        "Address",
        [
            "Adresse",
            "Indirizzo",
            "Dirección",
            "Endereço",
            "Adresse",
            "Cím",
        ],
    ),
    (
        "API key",
        [
            "Clé API",
            "Chiave API",
            "Clave de API",
            "Chave de API",
            "API-Schlüssel",
            "API-kulcs",
        ],
    ),
    (
        "(optional)",
        [
            "(facultatif)",
            "(facoltativa)",
            "(opcional)",
            "(opcional)",
            "(optional)",
            "(nem kötelező)",
        ],
    ),
    (
        "✔ Connected",
        [
            "✔ Connecté",
            "✔ Connesso",
            "✔ Conectado",
            "✔ Ligado",
            "✔ Verbunden",
            "✔ Csatlakoztatva",
        ],
    ),
    (
        "The server answers but is still loading",
        [
            "Le serveur répond mais se charge encore",
            "Il server risponde ma si sta ancora caricando",
            "El servidor responde pero aún se está cargando",
            "O servidor responde, mas ainda está a carregar",
            "Der Server antwortet, lädt aber noch",
            "A szerver válaszol, de még töltődik",
        ],
    ),
    (
        "Nothing answers at this address",
        [
            "Rien ne répond à cette adresse",
            "Nessuna risposta a questo indirizzo",
            "Nada responde en esta dirección",
            "Nada responde neste endereço",
            "Unter dieser Adresse antwortet nichts",
            "Ezen a címen semmi nem válaszol",
        ],
    ),
    (
        "Start the model automatically when the app opens",
        [
            "Démarrer le modèle automatiquement à l'ouverture de l'application",
            "Avvia il modello automaticamente all'apertura dell'app",
            "Iniciar el modelo automáticamente al abrir la aplicación",
            "Iniciar o modelo automaticamente ao abrir a aplicação",
            "Modell beim Öffnen der App automatisch starten",
            "A modell automatikus indítása az alkalmazás megnyitásakor",
        ],
    ),
    (
        "Folders",
        [
            "Dossiers", "Cartelle", "Carpetas", "Pastas", "Ordner", "Mappák",
        ],
    ),
    (
        "(found automatically)",
        [
            "(trouvé automatiquement)",
            "(trovata automaticamente)",
            "(encontrada automáticamente)",
            "(encontrada automaticamente)",
            "(automatisch gefunden)",
            "(automatikusan megtalálva)",
        ],
    ),
    (
        "Search again",
        [
            "Chercher à nouveau",
            "Cerca di nuovo",
            "Buscar de nuevo",
            "Procurar de novo",
            "Erneut suchen",
            "Keresés újra",
        ],
    ),
    (
        "Models",
        [
            "Modèles", "Modelli", "Modelos", "Modelos", "Modelle", "Modellek",
        ],
    ),
    (
        "Open the conversations folder",
        [
            "Ouvrir le dossier des conversations",
            "Apri la cartella delle conversazioni",
            "Abrir la carpeta de conversaciones",
            "Abrir a pasta das conversas",
            "Ordner der Unterhaltungen öffnen",
            "A beszélgetések mappájának megnyitása",
        ],
    ),
    (
        "About Colibri Companion",
        [
            "À propos de Colibri Companion",
            "Informazioni su Colibri Companion",
            "Acerca de Colibri Companion",
            "Acerca do Colibri Companion",
            "Über Colibri Companion",
            "A Colibri Companionról",
        ],
    ),
    (
        "A friendly, unofficial interface for the colibri engine. It is not made, supported or endorsed by colibri's authors.",
        [
            "Une interface conviviale et non officielle pour le moteur colibri. Elle n'est ni conçue, ni soutenue, ni approuvée par les auteurs de colibri.",
            "Un'interfaccia amichevole e non ufficiale per il motore colibri. Non è creata, supportata né approvata dagli autori di colibri.",
            "Una interfaz amigable y no oficial para el motor colibri. No está creada, respaldada ni aprobada por los autores de colibri.",
            "Uma interface amigável e não oficial para o motor colibri. Não é criada, apoiada nem aprovada pelos autores do colibri.",
            "Eine freundliche, inoffizielle Oberfläche für die colibri-Engine. Sie wird von den colibri-Autoren weder entwickelt noch unterstützt oder empfohlen.",
            "Barátságos, nem hivatalos felület a colibri motorhoz. A colibri szerzői nem készítették, nem támogatják és nem hagyták jóvá.",
        ],
    ),
    (
        "Licences",
        [
            "Licences",
            "Licenze",
            "Licencias",
            "Licenças",
            "Lizenzen",
            "Licencek",
        ],
    ),
    (
        "• This app: Apache License 2.0 (see LICENSE and NOTICE).\n• colibri (a separate project): Apache License 2.0. This app contains and modifies none of its code; it runs it and talks to its public API.\n• Models carry their own licences, shown before download.\n• The libraries used are listed below.",
        [
            "• Cette application : licence Apache 2.0 (voir LICENSE et NOTICE).\n• colibri (projet distinct) : licence Apache 2.0. Cette application n'en contient ni n'en modifie aucun code ; elle le lance et dialogue avec son API publique.\n• Les modèles ont leurs propres licences, affichées avant le téléchargement.\n• Les bibliothèques utilisées sont listées ci-dessous.",
            "• Questa app: Apache License 2.0 (vedi LICENSE e NOTICE).\n• colibri (progetto separato): Apache License 2.0. Questa app non contiene né modifica alcun suo codice: lo esegue e dialoga con la sua API pubblica.\n• I modelli hanno licenze proprie, mostrate prima del download.\n• Le librerie usate sono elencate qui sotto.",
            "• Esta aplicación: Apache License 2.0 (ver LICENSE y NOTICE).\n• colibri (proyecto aparte): Apache License 2.0. Esta aplicación no contiene ni modifica nada de su código: lo ejecuta y se comunica con su API pública.\n• Los modelos tienen sus propias licencias, que se muestran antes de descargar.\n• Las bibliotecas usadas se enumeran abajo.",
            "• Esta aplicação: Apache License 2.0 (ver LICENSE e NOTICE).\n• colibri (projeto separado): Apache License 2.0. Esta aplicação não contém nem altera nenhum código seu: executa-o e comunica com a sua API pública.\n• Os modelos têm licenças próprias, mostradas antes da transferência.\n• As bibliotecas utilizadas estão listadas abaixo.",
            "• Diese App: Apache License 2.0 (siehe LICENSE und NOTICE).\n• colibri (eigenständiges Projekt): Apache License 2.0. Diese App enthält und verändert keinen seiner Codes: Sie führt es aus und spricht mit seiner öffentlichen API.\n• Modelle haben eigene Lizenzen, die vor dem Download angezeigt werden.\n• Die verwendeten Bibliotheken sind unten aufgelistet.",
            "• Ez az alkalmazás: Apache License 2.0 (lásd: LICENSE és NOTICE).\n• colibri (különálló projekt): Apache License 2.0. Ez az alkalmazás semmit nem tartalmaz és nem módosít a kódjából: futtatja, és a nyilvános API-jával kommunikál.\n• A modelleknek saját licencük van, amelyet letöltés előtt megjelenítünk.\n• A felhasznált könyvtárak lent vannak felsorolva.",
        ],
    ),
    (
        "Third-party libraries",
        [
            "Bibliothèques tierces",
            "Librerie di terze parti",
            "Bibliotecas de terceros",
            "Bibliotecas de terceiros",
            "Bibliotheken von Drittanbietern",
            "Harmadik féltől származó könyvtárak",
        ],
    ),
    (
        "Log",
        [
            "Journal",
            "Registro",
            "Registro",
            "Registo",
            "Protokoll",
            "Napló",
        ],
    ),
    (
        "Precise",
        [
            "Précis", "Preciso", "Preciso", "Preciso", "Präzise", "Pontos",
        ],
    ),
    (
        "Balanced",
        [
            "Équilibré",
            "Equilibrato",
            "Equilibrado",
            "Equilibrado",
            "Ausgewogen",
            "Kiegyensúlyozott",
        ],
    ),
    (
        "Creative",
        [
            "Créatif", "Creativo", "Creativo", "Criativo", "Kreativ", "Kreatív",
        ],
    ),
    (
        "Auto",
        [
            "Auto",
            "Automatico",
            "Automático",
            "Automático",
            "Auto",
            "Automatikus",
        ],
    ),
    (
        "On",
        ["Activée", "Attivo", "Activado", "Ativado", "An", "Be"],
    ),
    (
        "Off",
        [
            "Désactivée",
            "Disattivato",
            "Desactivado",
            "Desativado",
            "Aus",
            "Ki",
        ],
    ),
    (
        "~{} left",
        [
            "encore ~{}",
            "ancora ~{}",
            "faltan ~{}",
            "faltam ~{}",
            "noch ~{}",
            "még ~{}",
        ],
    ),
    (
        "Thinking…",
        [
            "Réflexion…",
            "Sta ragionando…",
            "Pensando…",
            "A pensar…",
            "Denkt nach…",
            "Gondolkodik…",
        ],
    ),
    (
        "How it thought",
        [
            "Comment il a réfléchi",
            "Come ha ragionato",
            "Cómo razonó",
            "Como pensou",
            "Wie es gedacht hat",
            "Hogyan gondolkodott",
        ],
    ),
    (
        "Reading your message… {} s",
        [
            "Lecture de votre message… {} s",
            "Sto leggendo il messaggio… {} s",
            "Leyendo tu mensaje… {} s",
            "A ler a tua mensagem… {} s",
            "Deine Nachricht wird gelesen… {} s",
            "Az üzeneted olvasása… {} s",
        ],
    ),
    (
        "(stopped before the end)",
        [
            "(arrêté avant la fin)",
            "(fermato prima della fine)",
            "(detenido antes del final)",
            "(parado antes do fim)",
            "(vor dem Ende gestoppt)",
            "(a vége előtt leállítva)",
        ],
    ),
    (
        "Regenerate",
        [
            "Régénérer",
            "Rigenera",
            "Regenerar",
            "Regenerar",
            "Neu generieren",
            "Újragenerálás",
        ],
    ),
    (
        "colibri on GitHub",
        [
            "colibri sur GitHub",
            "colibri su GitHub",
            "colibri en GitHub",
            "colibri no GitHub",
            "colibri auf GitHub",
            "colibri a GitHubon",
        ],
    ),
    (
        "Bundled fonts: Inter and JetBrains Mono, under the SIL Open Font License 1.1.",
        [
            "Polices incluses : Inter et JetBrains Mono, sous licence SIL Open Font License 1.1.",
            "Caratteri inclusi: Inter e JetBrains Mono, con licenza SIL Open Font License 1.1.",
            "Fuentes incluidas: Inter y JetBrains Mono, con licencia SIL Open Font License 1.1.",
            "Tipos de letra incluídos: Inter e JetBrains Mono, sob a licença SIL Open Font License 1.1.",
            "Enthaltene Schriften: Inter und JetBrains Mono, unter der SIL Open Font License 1.1.",
            "Beépített betűtípusok: Inter és JetBrains Mono, SIL Open Font License 1.1 alatt.",
        ],
    ),
    (
        "(empty)",
        [
            "(vide)", "(vuoto)", "(vacío)", "(vazio)", "(leer)", "(üres)",
        ],
    ),
];

fn index(lang: Lang) -> Option<usize> {
    match lang {
        Lang::Fr => Some(0),
        Lang::It => Some(1),
        Lang::Es => Some(2),
        Lang::Pt => Some(3),
        Lang::De => Some(4),
        Lang::Hu => Some(5),
        Lang::Ro | Lang::En => None,
    }
}

/// The translation of an English interface text, if there is one for this language.
pub fn lookup(lang: Lang, english: &str) -> Option<&'static str> {
    static MAP: OnceLock<HashMap<&'static str, &'static [&'static str; 6]>> = OnceLock::new();
    let map = MAP.get_or_init(|| TABLE.iter().map(|(k, v)| (*k, v)).collect());
    let i = index(lang)?;
    map.get(english).map(|row| row[i])
}

/// Texts that colibri itself produces in English (model summaries, reasons, warnings).
/// Row order: ro, fr, it, es, pt, de, hu. `{}` stands for a value that is copied over.
static ENGINE: &[(&str, [&str; 7])] = &[
    (
        "no {} engine for {}/{}",
        [
            "niciun motor {} pentru {}/{}",
            "aucun moteur {} pour {}/{}",
            "nessun motore {} per {}/{}",
            "no hay motor {} para {}/{}",
            "não há motor {} para {}/{}",
            "keine {}-Engine für {}/{}",
            "nincs {} motor ehhez: {}/{}",
        ],
    ),
    (
        "needs {} GB of RAM (dense part {} GB plus the expert cache); this machine has {} GB",
        [
            "are nevoie de {} GB de RAM (partea densă {} GB plus memoria-tampon pentru experți); acest calculator are {} GB",
            "nécessite {} GB de RAM (partie dense {} GB plus le cache des experts) ; cet ordinateur en a {} GB",
            "servono {} GB di RAM (parte densa {} GB più la cache degli esperti); questo computer ne ha {} GB",
            "necesita {} GB de RAM (parte densa {} GB más la caché de expertos); este ordenador tiene {} GB",
            "precisa de {} GB de RAM (parte densa {} GB mais a cache dos especialistas); este computador tem {} GB",
            "braucht {} GB RAM (dichter Teil {} GB plus Experten-Cache); dieser Computer hat {} GB",
            "{} GB RAM kell hozzá (sűrű rész: {} GB, plusz a szakértői gyorsítótár); ennek a gépnek {} GB van",
        ],
    ),
    (
        "needs {} GB free for the download; {} GB free there",
        [
            "are nevoie de {} GB liberi pentru descărcare; acolo sunt liberi {} GB",
            "nécessite {} GB libres pour le téléchargement ; {} GB libres à cet endroit",
            "servono {} GB liberi per il download; lì ce ne sono {} GB liberi",
            "necesita {} GB libres para la descarga; ahí hay {} GB libres",
            "precisa de {} GB livres para a transferência; aí há {} GB livres",
            "braucht {} GB freien Platz für den Download; dort sind {} GB frei",
            "{} GB szabad hely kell a letöltéshez; ott {} GB szabad",
        ],
    ),
    (
        "runs with a small expert cache (needs {} GB, runs best with {} GB)",
        [
            "rulează cu o memorie-tampon mică pentru experți (are nevoie de {} GB, merge cel mai bine cu {} GB)",
            "fonctionne avec un petit cache d'experts (nécessite {} GB, fonctionne mieux avec {} GB)",
            "funziona con una piccola cache degli esperti (servono {} GB, va meglio con {} GB)",
            "funciona con una caché de expertos pequeña (necesita {} GB, rinde mejor con {} GB)",
            "funciona com uma cache de especialistas pequena (precisa de {} GB, funciona melhor com {} GB)",
            "läuft mit kleinem Experten-Cache (braucht {} GB, läuft am besten mit {} GB)",
            "kis szakértői gyorsítótárral fut (kell hozzá {} GB, a legjobb {} GB-tal)",
        ],
    ),
    (
        "the whole model fits in RAM ({} GB)",
        [
            "tot modelul încape în RAM ({} GB)",
            "tout le modèle tient en RAM ({} GB)",
            "l'intero modello sta nella RAM ({} GB)",
            "todo el modelo cabe en la RAM ({} GB)",
            "o modelo inteiro cabe na RAM ({} GB)",
            "das ganze Modell passt in den RAM ({} GB)",
            "az egész modell elfér a RAM-ban ({} GB)",
        ],
    ),
    (
        "dense part {} GB stays in RAM, the experts stream from the SSD (a fast NVMe drive makes it faster)",
        [
            "partea densă de {} GB rămâne în RAM, experții se citesc de pe SSD (un disc NVMe rapid ajută)",
            "la partie dense ({} GB) reste en RAM, les experts sont lus depuis le SSD (un NVMe rapide aide)",
            "la parte densa ({} GB) resta in RAM, gli esperti vengono letti dall'SSD (un NVMe veloce aiuta)",
            "la parte densa ({} GB) se queda en la RAM, los expertos se leen del SSD (un NVMe rápido ayuda)",
            "a parte densa ({} GB) fica na RAM, os especialistas são lidos do SSD (um NVMe rápido ajuda)",
            "der dichte Teil ({} GB) bleibt im RAM, die Experten werden von der SSD gelesen (eine schnelle NVMe hilft)",
            "a sűrű rész ({} GB) a RAM-ban marad, a szakértőket az SSD-ről olvassa (egy gyors NVMe segít)",
        ],
    ),
    (
        "{} is a Windows drive seen from WSL: reading it is many times slower than the Linux disk, and the engine streams the model from it. Keep the model in the Linux home (for example ~/colibri-models).",
        [
            "{} este o unitate Windows văzută din WSL: citirea de pe ea este de multe ori mai lentă decât discul Linux, iar motorul citește modelul de acolo. Ține modelul în folderul personal Linux (de exemplu ~/colibri-models).",
            "{} est un lecteur Windows vu depuis WSL : le lire est bien plus lent que le disque Linux, et le moteur lit le modèle depuis cet endroit. Gardez le modèle dans le dossier personnel Linux (par exemple ~/colibri-models).",
            "{} è un'unità Windows vista da WSL: leggerla è molte volte più lento del disco Linux, e il motore legge il modello da lì. Tieni il modello nella home di Linux (ad esempio ~/colibri-models).",
            "{} es una unidad de Windows vista desde WSL: leerla es muchas veces más lento que el disco de Linux, y el motor lee el modelo desde ahí. Guarda el modelo en la carpeta personal de Linux (por ejemplo ~/colibri-models).",
            "{} é uma unidade do Windows vista a partir do WSL: lê-la é muitas vezes mais lento do que o disco do Linux, e o motor lê o modelo a partir daí. Guarda o modelo na pasta pessoal do Linux (por exemplo ~/colibri-models).",
            "{} ist ein Windows-Laufwerk, von WSL aus gesehen: Lesen davon ist um ein Vielfaches langsamer als die Linux-Festplatte, und die Engine liest das Modell von dort. Behalte das Modell im Linux-Home (zum Beispiel ~/colibri-models).",
            "{} egy Windows-meghajtó WSL-ből nézve: olvasása sokszor lassabb a Linux-lemeznél, a motor pedig onnan olvassa a modellt. Tartsd a modellt a Linux saját mappájában (például ~/colibri-models).",
        ],
    ),
    (
        "Qwen Research License: non-commercial use only",
        [
            "Licență Qwen Research: doar uz necomercial",
            "Licence Qwen Research : usage non commercial uniquement",
            "Licenza Qwen Research: solo uso non commerciale",
            "Licencia Qwen Research: solo uso no comercial",
            "Licença Qwen Research: apenas uso não comercial",
            "Qwen-Research-Lizenz: nur nicht kommerzielle Nutzung",
            "Qwen Research licenc: csak nem kereskedelmi felhasználás",
        ],
    ),
    (
        "text to image (pictures, not chat)",
        [
            "text în imagine (imagini, nu conversație)",
            "texte vers image (images, pas de conversation)",
            "da testo a immagine (immagini, non chat)",
            "texto a imagen (imágenes, no chat)",
            "texto para imagem (imagens, não conversa)",
            "Text-zu-Bild (Bilder, kein Chat)",
            "szövegből kép (képek, nem csevegés)",
        ],
    ),
    (
        "pruned DeepSeek V4 Flash (132 of 256 experts), official fp4/fp8 weights",
        [
            "DeepSeek V4 Flash redus (132 din 256 de experți), ponderi oficiale fp4/fp8",
            "DeepSeek V4 Flash élagué (132 experts sur 256), poids officiels fp4/fp8",
            "DeepSeek V4 Flash sfoltito (132 esperti su 256), pesi ufficiali fp4/fp8",
            "DeepSeek V4 Flash podado (132 de 256 expertos), pesos oficiales fp4/fp8",
            "DeepSeek V4 Flash podado (132 de 256 especialistas), pesos oficiais fp4/fp8",
            "beschnittenes DeepSeek V4 Flash (132 von 256 Experten), offizielle fp4/fp8-Gewichte",
            "megtisztított DeepSeek V4 Flash (256 szakértőből 132), hivatalos fp4/fp8 súlyok",
        ],
    ),
    (
        "284B, official checkpoint, no conversion",
        [
            "284B, punct de control oficial, fără conversie",
            "284B, checkpoint officiel, sans conversion",
            "284B, checkpoint ufficiale, senza conversione",
            "284B, checkpoint oficial, sin conversión",
            "284B, checkpoint oficial, sem conversão",
            "284B, offizieller Checkpoint, ohne Konvertierung",
            "284B, hivatalos checkpoint, átalakítás nélkül",
        ],
    ),
    (
        "{} with vision and tools, official checkpoint",
        [
            "{} cu viziune și instrumente, punct de control oficial",
            "{} avec vision et outils, checkpoint officiel",
            "{} con visione e strumenti, checkpoint ufficiale",
            "{} con visión y herramientas, checkpoint oficial",
            "{} com visão e ferramentas, checkpoint oficial",
            "{} mit Bildverständnis und Werkzeugen, offizieller Checkpoint",
            "{} látással és eszközökkel, hivatalos checkpoint",
        ],
    ),
    (
        "125B + 51B n-gram, official FP8 checkpoint (pinned revision)",
        [
            "125B + 51B n-gram, punct de control oficial FP8 (revizie fixată)",
            "125B + 51B n-gram, checkpoint officiel FP8 (révision figée)",
            "125B + 51B n-gram, checkpoint ufficiale FP8 (revisione fissata)",
            "125B + 51B n-gram, checkpoint oficial FP8 (revisión fijada)",
            "125B + 51B n-gram, checkpoint oficial FP8 (revisão fixada)",
            "125B + 51B n-Gramm, offizieller FP8-Checkpoint (festgelegte Revision)",
            "125B + 51B n-gram, hivatalos FP8 checkpoint (rögzített revízió)",
        ],
    ),
    (
        "744B, int4-gs64, no MTP head (no speculative decoding)",
        [
            "744B, int4-gs64, fără cap MTP (fără decodare speculativă)",
            "744B, int4-gs64, sans tête MTP (sans décodage spéculatif)",
            "744B, int4-gs64, senza testa MTP (senza decodifica speculativa)",
            "744B, int4-gs64, sin cabeza MTP (sin decodificación especulativa)",
            "744B, int4-gs64, sem cabeça MTP (sem descodificação especulativa)",
            "744B, int4-gs64, ohne MTP-Kopf (ohne spekulatives Decoding)",
            "744B, int4-gs64, MTP-fej nélkül (spekulatív dekódolás nélkül)",
        ],
    ),
    (
        "744B, the reference model; int4-gs64 with the int8 MTP head",
        [
            "744B, modelul de referință; int4-gs64 cu capul MTP int8",
            "744B, le modèle de référence ; int4-gs64 avec tête MTP int8",
            "744B, il modello di riferimento; int4-gs64 con testa MTP int8",
            "744B, el modelo de referencia; int4-gs64 con cabeza MTP int8",
            "744B, o modelo de referência; int4-gs64 com cabeça MTP int8",
            "744B, das Referenzmodell; int4-gs64 mit int8-MTP-Kopf",
            "744B, a referenciamodell; int4-gs64 int8 MTP-fejjel",
        ],
    ),
    (
        "975B, int4 experts with bf16 dense weights",
        [
            "975B, experți int4 cu ponderi dense bf16",
            "975B, experts int4 avec poids denses bf16",
            "975B, esperti int4 con pesi densi bf16",
            "975B, expertos int4 con pesos densos bf16",
            "975B, especialistas int4 com pesos densos bf16",
            "975B, int4-Experten mit dichten bf16-Gewichten",
            "975B, int4 szakértők bf16 sűrű súlyokkal",
        ],
    ),
    (
        "2.8T, original checkpoint with native MXFP4 experts",
        [
            "2.8T, punct de control original cu experți MXFP4 nativi",
            "2.8T, checkpoint original avec experts MXFP4 natifs",
            "2.8T, checkpoint originale con esperti MXFP4 nativi",
            "2.8T, checkpoint original con expertos MXFP4 nativos",
            "2.8T, checkpoint original com especialistas MXFP4 nativos",
            "2.8T, originaler Checkpoint mit nativen MXFP4-Experten",
            "2.8T, eredeti checkpoint natív MXFP4 szakértőkkel",
        ],
    ),
];

/// Splits `text` along the `{}` slots of `template`; `None` when it does not match.
fn capture<'a>(template: &str, text: &'a str) -> Option<Vec<&'a str>> {
    let parts: Vec<&str> = template.split("{}").collect();
    let mut rest = text.strip_prefix(parts[0])?;
    let mut values = Vec::new();
    for (i, lit) in parts.iter().enumerate().skip(1) {
        let end = if i == parts.len() - 1 {
            if lit.is_empty() {
                rest.len()
            } else {
                rest.strip_suffix(lit).map(|r| r.len())?
            }
        } else {
            rest.find(lit)?
        };
        values.push(&rest[..end]);
        rest = &rest[end + lit.len()..];
    }
    rest.is_empty().then_some(values)
}

/// A text from colibri, in the interface language when it is a known one; otherwise unchanged.
pub fn engine_text(lang: Lang, text: &str) -> String {
    let col = match lang {
        Lang::En => return text.to_owned(),
        Lang::Ro => 0,
        Lang::Fr => 1,
        Lang::It => 2,
        Lang::Es => 3,
        Lang::Pt => 4,
        Lang::De => 5,
        Lang::Hu => 6,
    };
    for (template, rows) in ENGINE {
        if let Some(values) = capture(template, text) {
            let mut out = String::new();
            let mut values = values.into_iter();
            for (i, piece) in rows[col].split("{}").enumerate() {
                if i > 0 {
                    out.push_str(values.next().unwrap_or(""));
                }
                out.push_str(piece);
            }
            return out;
        }
    }
    text.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_complete_and_keep_their_structure() {
        for (key, row) in TABLE {
            for text in row {
                assert!(!text.trim().is_empty(), "empty translation for {key:?}");
                assert_eq!(
                    text.matches("{}").count(),
                    key.matches("{}").count(),
                    "placeholder mismatch in {text:?}"
                );
                assert_eq!(
                    text.matches('\n').count(),
                    key.matches('\n').count(),
                    "line breaks differ in {text:?}"
                );
            }
        }
        let mut seen = std::collections::HashSet::new();
        assert!(TABLE.iter().all(|(k, _)| seen.insert(*k)), "duplicate key");
    }

    #[test]
    fn lookup_works_and_falls_back() {
        assert_eq!(lookup(Lang::De, "Send"), Some("Senden"));
        assert_eq!(lookup(Lang::Hu, "Settings"), Some("Beállítások"));
        assert_eq!(lookup(Lang::En, "Send"), None);
        assert_eq!(lookup(Lang::Fr, "no such text"), None);
    }

    /// Every `t("romanian", "english")` in the sources must have all six translations.
    #[test]
    fn every_interface_text_is_translated() {
        fn literal(s: &str) -> Option<(String, &str)> {
            let s = s.trim_start().strip_prefix('"')?;
            let (mut out, mut chars) = (String::new(), s.char_indices());
            while let Some((i, c)) = chars.next() {
                match c {
                    '"' => return Some((out, &s[i + 1..])),
                    '\\' => match chars.next()?.1 {
                        'n' => out.push('\n'),
                        other => out.push(other),
                    },
                    c => out.push(c),
                }
            }
            None
        }
        let sources = [
            include_str!("app.rs"),
            include_str!("chat_ui.rs"),
            include_str!("screens.rs"),
            include_str!("windows.rs"),
        ];
        let (mut checked, mut missing) = (0, Vec::new());
        for src in sources {
            for (pos, _) in src.match_indices("t(") {
                if src[..pos].ends_with(|c: char| c.is_alphanumeric() || c == '_')
                    && !src[..pos].ends_with("self.")
                {
                    // `format!(`, `print(` and similar: only `t(` and `self.t(` count
                    if !src[..pos].ends_with('.') {
                        continue;
                    }
                }
                let rest = &src[pos + 2..];
                let Some((_ro, rest)) = literal(rest) else {
                    continue;
                };
                let Some(rest) = rest.trim_start().strip_prefix(',') else {
                    continue;
                };
                let Some((en, _)) = literal(rest) else {
                    continue;
                };
                checked += 1;
                for lang in [Lang::Fr, Lang::It, Lang::Es, Lang::Pt, Lang::De, Lang::Hu] {
                    if lookup(lang, &en).is_none() {
                        missing.push(format!("{lang:?}: {en}"));
                    }
                }
            }
        }
        assert!(
            checked > 150,
            "only {checked} texts found; the scanner is broken"
        );
        assert!(
            missing.is_empty(),
            "untranslated texts:\n{}",
            missing.join("\n")
        );
    }

    #[test]
    fn engine_texts_are_translated() {
        let t =
            "needs 16 GB of RAM (dense part 7.9 GB plus the expert cache); this machine has 8.4 GB";
        assert_eq!(
            engine_text(Lang::Fr, t),
            "nécessite 16 GB de RAM (partie dense 7.9 GB plus le cache des experts) ; cet ordinateur en a 8.4 GB"
        );
        assert_eq!(
            engine_text(Lang::De, "1.02T with vision and tools, official checkpoint"),
            "1.02T mit Bildverständnis und Werkzeugen, offizieller Checkpoint"
        );
        assert_eq!(engine_text(Lang::En, t), t);
        assert_eq!(engine_text(Lang::Hu, "something new"), "something new");
        assert!(
            engine_text(Lang::Ro, "the whole model fits in RAM (4 GB)").contains("tot modelul")
        );
    }
    #[test]
    fn every_engine_row_keeps_its_slots() {
        for (key, rows) in ENGINE {
            for r in rows {
                assert_eq!(r.matches("{}").count(), key.matches("{}").count(), "{r}");
            }
        }
    }

    /// The texts colibri 's real catalog produced when this was written all have translations.
    #[test]
    fn real_catalog_texts_are_covered() {
        let texts = [
            "runs with a small expert cache (needs 8 GB, runs best with 18 GB)",
            "needs 10 GB of RAM (dense part 3.5 GB plus the expert cache); this machine has 8.4 GB",
            "needs 120 GB of RAM (dense part 49.4 GB plus the expert cache); this machine has 8.4 GB",
            "needs 23 GB free for the download; 5 GB free there",
            "the whole model fits in RAM (16 GB)",
            "dense part 7.9 GB stays in RAM, the experts stream from the SSD (a fast NVMe drive makes it faster)",
            "Qwen Research License: non-commercial use only",
            "text to image (pictures, not chat)",
            "pruned DeepSeek V4 Flash (132 of 256 experts), official fp4/fp8 weights",
            "284B, official checkpoint, no conversion",
            "309B with vision and tools, official checkpoint",
            "1.02T with vision and tools, official checkpoint",
            "125B + 51B n-gram, official FP8 checkpoint (pinned revision)",
            "744B, int4-gs64, no MTP head (no speculative decoding)",
            "744B, the reference model; int4-gs64 with the int8 MTP head",
            "975B, int4 experts with bf16 dense weights",
            "2.8T, original checkpoint with native MXFP4 experts",
            "/mnt/c/models is a Windows drive seen from WSL: reading it is many times slower than the Linux disk, and the engine streams the model from it. Keep the model in the Linux home (for example ~/colibri-models).",
        ];
        for lang in [
            Lang::Ro,
            Lang::Fr,
            Lang::It,
            Lang::Es,
            Lang::Pt,
            Lang::De,
            Lang::Hu,
        ] {
            for t in texts {
                assert_ne!(engine_text(lang, t), t, "{lang:?}: {t}");
            }
        }
    }
}
