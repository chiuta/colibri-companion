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
    (
        "colibri engine",
        [
            "moteur colibri",
            "motore colibri",
            "motor colibri",
            "motor colibri",
            "colibri-Engine",
            "colibri motor",
        ],
    ),
    (
        "colibri setup",
        [
            "installation colibri",
            "installazione colibri",
            "instalación de colibri",
            "instalação do colibri",
            "colibri-Einrichtung",
            "colibri telepítés",
        ],
    ),
    (
        "colibri is up to date. Check the model list: there may be new models.",
        [
            "colibri est à jour. Regardez la liste des modèles : il peut y en avoir de nouveaux.",
            "colibri è aggiornato. Controlla l'elenco dei modelli: potrebbero essercene di nuovi.",
            "colibri está al día. Revisa la lista de modelos: puede haber modelos nuevos.",
            "O colibri está atualizado. Vê a lista de modelos: pode haver modelos novos.",
            "colibri ist aktuell. Sieh dir die Modellliste an: Es kann neue Modelle geben.",
            "A colibri naprakész. Nézd meg a modellek listáját: lehetnek új modellek.",
        ],
    ),
    (
        "The update failed",
        [
            "La mise à jour a échoué",
            "L'aggiornamento non è riuscito",
            "La actualización falló",
            "A atualização falhou",
            "Das Update ist fehlgeschlagen",
            "A frissítés nem sikerült",
        ],
    ),
    (
        "Deleted “{}”",
        [
            "« {} » supprimé",
            "«{}» eliminato",
            "«{}» eliminado",
            "«{}» eliminado",
            "„{}“ gelöscht",
            "„{}” törölve",
        ],
    ),
    (
        "Could not delete the model",
        [
            "Impossible de supprimer le modèle",
            "Impossibile eliminare il modello",
            "No se pudo eliminar el modelo",
            "Não foi possível eliminar o modelo",
            "Das Modell konnte nicht gelöscht werden",
            "A modellt nem sikerült törölni",
        ],
    ),
    (
        "I download colibri from GitHub into your home folder (~/colibri), no git needed. It's small.",
        [
            "Je télécharge colibri depuis GitHub dans votre dossier personnel (~/colibri), sans avoir besoin de git. C'est léger.",
            "Scarico colibri da GitHub nella tua cartella personale (~/colibri), senza bisogno di git. Occupa poco spazio.",
            "Descargo colibri de GitHub en tu carpeta personal (~/colibri), sin necesitar git. Ocupa poco.",
            "Descarrego o colibri do GitHub para a tua pasta pessoal (~/colibri), sem precisares de git. Ocupa pouco espaço.",
            "Ich lade colibri von GitHub in deinen Home-Ordner (~/colibri), ganz ohne git. Es ist klein.",
            "Letöltöm a colibrit a GitHubról a saját mappádba (~/colibri), git nélkül. Kicsi.",
        ],
    ),
    (
        "Unpacking…",
        [
            "Décompression…",
            "Decompressione…",
            "Descomprimiendo…",
            "A descompactar…",
            "Entpacken…",
            "Kicsomagolás…",
        ],
    ),
    (
        "Not working?",
        [
            "Ça ne marche pas ?",
            "Non funziona?",
            "¿No funciona?",
            "Não funciona?",
            "Klappt es nicht?",
            "Nem működik?",
        ],
    ),
    (
        "Open the ZIP download in your browser",
        [
            "Ouvrir le téléchargement ZIP dans le navigateur",
            "Apri il download ZIP nel browser",
            "Abrir la descarga ZIP en el navegador",
            "Abrir a transferência ZIP no navegador",
            "ZIP-Download im Browser öffnen",
            "A ZIP letöltés megnyitása a böngészőben",
        ],
    ),
    (
        "Resources",
        [
            "Ressources",
            "Risorse",
            "Recursos",
            "Recursos",
            "Ressourcen",
            "Erőforrások",
        ],
    ),
    (
        "See the resources in use and the proof that nothing leaves your computer",
        [
            "Voir les ressources utilisées et la preuve que rien ne quitte votre ordinateur",
            "Vedi le risorse in uso e la prova che nulla esce dal tuo computer",
            "Ver los recursos en uso y la prueba de que nada sale de tu ordenador",
            "Ver os recursos em uso e a prova de que nada sai do teu computador",
            "Genutzte Ressourcen und den Beweis ansehen, dass nichts deinen Computer verlässt",
            "A használt erőforrások és a bizonyíték, hogy semmi sem hagyja el a számítógépedet",
        ],
    ),
    (
        "Check once a day for new versions (one small request to GitHub)",
        [
            "Chercher une fois par jour de nouvelles versions (une petite requête vers GitHub)",
            "Controlla una volta al giorno se ci sono nuove versioni (una piccola richiesta a GitHub)",
            "Buscar una vez al día versiones nuevas (una pequeña petición a GitHub)",
            "Procurar uma vez por dia novas versões (um pequeno pedido ao GitHub)",
            "Einmal täglich nach neuen Versionen suchen (eine kleine Anfrage an GitHub)",
            "Naponta egyszer új verziók keresése (egy kis kérés a GitHubra)",
        ],
    ),
    (
        "Check now",
        [
            "Vérifier maintenant",
            "Controlla ora",
            "Comprobar ahora",
            "Verificar agora",
            "Jetzt prüfen",
            "Ellenőrzés most",
        ],
    ),
    (
        "My models",
        [
            "Mes modèles",
            "I miei modelli",
            "Mis modelos",
            "Os meus modelos",
            "Meine Modelle",
            "Saját modellek",
        ],
    ),
    (
        "You can keep several models on this computer and switch between them any time. Only one runs at a time, so memory isn't overloaded.",
        [
            "Vous pouvez garder plusieurs modèles sur cet ordinateur et passer de l'un à l'autre à tout moment. Un seul fonctionne à la fois, pour ne pas saturer la mémoire.",
            "Puoi tenere più modelli su questo computer e passare dall'uno all'altro quando vuoi. Ne gira uno solo alla volta, così la memoria non si sovraccarica.",
            "Puedes tener varios modelos en este ordenador y cambiar entre ellos cuando quieras. Solo uno funciona a la vez, para no sobrecargar la memoria.",
            "Podes ter vários modelos neste computador e alternar entre eles quando quiseres. Só um funciona de cada vez, para não sobrecarregar a memória.",
            "Du kannst mehrere Modelle auf diesem Computer behalten und jederzeit zwischen ihnen wechseln. Es läuft immer nur eines, damit der Speicher nicht überlastet wird.",
            "Több modellt is tarthatsz ezen a számítógépen, és bármikor válthatsz közöttük. Egyszerre csak egy fut, hogy a memória ne terhelődjön túl.",
        ],
    ),
    (
        "A model is installing right now. Follow its progress in the main window.",
        [
            "Un modèle est en cours d'installation. Suivez sa progression dans la fenêtre principale.",
            "Un modello si sta installando ora. Segui l'avanzamento nella finestra principale.",
            "Se está instalando un modelo ahora mismo. Sigue su progreso en la ventana principal.",
            "Está a ser instalado um modelo neste momento. Acompanha o progresso na janela principal.",
            "Gerade wird ein Modell installiert. Den Fortschritt siehst du im Hauptfenster.",
            "Éppen telepítés zajlik. Kövesd a folyamatot a főablakban.",
        ],
    ),
    (
        "Installed on this computer",
        [
            "Installés sur cet ordinateur",
            "Installati su questo computer",
            "Instalados en este ordenador",
            "Instalados neste computador",
            "Auf diesem Computer installiert",
            "Telepítve ezen a számítógépen",
        ],
    ),
    (
        "I haven't found any downloaded model yet.",
        [
            "Je n'ai encore trouvé aucun modèle téléchargé.",
            "Non ho ancora trovato nessun modello scaricato.",
            "Todavía no he encontrado ningún modelo descargado.",
            "Ainda não encontrei nenhum modelo descarregado.",
            "Ich habe noch kein heruntergeladenes Modell gefunden.",
            "Még nem találtam letöltött modellt.",
        ],
    ),
    (
        "Add another model",
        [
            "Ajouter un autre modèle",
            "Aggiungi un altro modello",
            "Añadir otro modelo",
            "Adicionar outro modelo",
            "Weiteres Modell hinzufügen",
            "Újabb modell hozzáadása",
        ],
    ),
    (
        "You already have every chat model on the list.",
        [
            "Vous avez déjà tous les modèles de conversation de la liste.",
            "Hai già tutti i modelli di conversazione dell'elenco.",
            "Ya tienes todos los modelos de conversación de la lista.",
            "Já tens todos os modelos de conversa da lista.",
            "Du hast bereits alle Chat-Modelle der Liste.",
            "A lista összes csevegőmodellje megvan már.",
        ],
    ),
    (
        "Models come from Hugging Face and each has its own licence.",
        [
            "Les modèles viennent de Hugging Face et chacun a sa propre licence.",
            "I modelli provengono da Hugging Face e ognuno ha la propria licenza.",
            "Los modelos vienen de Hugging Face y cada uno tiene su propia licencia.",
            "Os modelos vêm do Hugging Face e cada um tem a sua própria licença.",
            "Die Modelle stammen von Hugging Face und haben jeweils eine eigene Lizenz.",
            "A modellek a Hugging Face-ről származnak, és mindegyiknek saját licence van.",
        ],
    ),
    (
        "Running now",
        [
            "En cours d'exécution",
            "In esecuzione",
            "En ejecución",
            "Em execução",
            "Läuft gerade",
            "Most fut",
        ],
    ),
    (
        "Selected",
        [
            "Sélectionné",
            "Selezionato",
            "Seleccionado",
            "Selecionado",
            "Ausgewählt",
            "Kiválasztva",
        ],
    ),
    (
        "Unfinished download",
        [
            "Téléchargement inachevé",
            "Download incompleto",
            "Descarga sin terminar",
            "Transferência por terminar",
            "Unvollständiger Download",
            "Befejezetlen letöltés",
        ],
    ),
    (
        "Delete “{}”? This frees {}. You can download it again any time.",
        [
            "Supprimer « {} » ? Cela libère {}. Vous pourrez le télécharger de nouveau à tout moment.",
            "Eliminare «{}»? Libera {}. Potrai scaricarlo di nuovo quando vuoi.",
            "¿Eliminar «{}»? Se liberan {}. Podrás descargarlo de nuevo cuando quieras.",
            "Eliminar «{}»? Liberta {}. Podes descarregá-lo de novo quando quiseres.",
            "„{}“ löschen? Das gibt {} frei. Du kannst es jederzeit erneut herunterladen.",
            "Törlöd a(z) „{}” modellt? Ez {} helyet szabadít fel. Bármikor újra letöltheted.",
        ],
    ),
    (
        "Yes, delete",
        [
            "Oui, supprimer",
            "Sì, elimina",
            "Sí, eliminar",
            "Sim, eliminar",
            "Ja, löschen",
            "Igen, törlöm",
        ],
    ),
    (
        "Resume the download",
        [
            "Reprendre le téléchargement",
            "Riprendi il download",
            "Reanudar la descarga",
            "Retomar a transferência",
            "Download fortsetzen",
            "Letöltés folytatása",
        ],
    ),
    (
        "Start this model",
        [
            "Démarrer ce modèle",
            "Avvia questo modello",
            "Iniciar este modelo",
            "Iniciar este modelo",
            "Dieses Modell starten",
            "A modell indítása",
        ],
    ),
    (
        "Use this model",
        [
            "Utiliser ce modèle",
            "Usa questo modello",
            "Usar este modelo",
            "Usar este modelo",
            "Dieses Modell verwenden",
            "Ennek a modellnek a használata",
        ],
    ),
    (
        "Delete",
        [
            "Supprimer",
            "Elimina",
            "Eliminar",
            "Eliminar",
            "Löschen",
            "Törlés",
        ],
    ),
    (
        "Open folder",
        [
            "Ouvrir le dossier",
            "Apri la cartella",
            "Abrir la carpeta",
            "Abrir a pasta",
            "Ordner öffnen",
            "Mappa megnyitása",
        ],
    ),
    (
        "Install this model",
        [
            "Installer ce modèle",
            "Installa questo modello",
            "Instalar este modelo",
            "Instalar este modelo",
            "Dieses Modell installieren",
            "A modell telepítése",
        ],
    ),
    (
        "The model that is running now will stop while this one downloads. It stays installed and you can switch back any time.",
        [
            "Le modèle en cours s'arrêtera pendant le téléchargement de celui-ci. Il reste installé et vous pourrez y revenir à tout moment.",
            "Il modello in esecuzione si fermerà mentre questo viene scaricato. Resta installato e potrai tornarci in qualsiasi momento.",
            "El modelo que está en marcha se detendrá mientras se descarga este. Sigue instalado y podrás volver a él cuando quieras.",
            "O modelo em execução vai parar enquanto este é descarregado. Continua instalado e podes voltar a ele quando quiseres.",
            "Das laufende Modell wird angehalten, solange dieses heruntergeladen wird. Es bleibt installiert, und du kannst jederzeit zurückwechseln.",
            "A most futó modell leáll, amíg ez letöltődik. Telepítve marad, és bármikor visszaválthatsz rá.",
        ],
    ),
    (
        "The model has its own licence.",
        [
            "Le modèle a sa propre licence.",
            "Il modello ha una propria licenza.",
            "El modelo tiene su propia licencia.",
            "O modelo tem a sua própria licença.",
            "Das Modell hat eine eigene Lizenz.",
            "A modellnek saját licence van.",
        ],
    ),
    (
        "Download and use",
        [
            "Télécharger et utiliser",
            "Scarica e usa",
            "Descargar y usar",
            "Descarregar e usar",
            "Herunterladen und verwenden",
            "Letöltés és használat",
        ],
    ),
    (
        "NVIDIA graphics card found:",
        [
            "Carte graphique NVIDIA trouvée :",
            "Scheda grafica NVIDIA trovata:",
            "Tarjeta gráfica NVIDIA encontrada:",
            "Placa gráfica NVIDIA encontrada:",
            "NVIDIA-Grafikkarte gefunden:",
            "NVIDIA videokártya található:",
        ],
    ),
    (
        "I use it to the fullest with CUDA",
        [
            "je l'utilise au maximum avec CUDA",
            "la sfrutto al massimo con CUDA",
            "la aprovecho al máximo con CUDA",
            "uso-a ao máximo com CUDA",
            "ich nutze sie mit CUDA voll aus",
            "CUDA-val teljes mértékben kihasználom",
        ],
    ),
    (
        "Graphics chip found:",
        [
            "Puce graphique trouvée :",
            "Chip grafico trovato:",
            "Chip gráfico encontrado:",
            "Chip gráfico encontrado:",
            "Grafikchip gefunden:",
            "Grafikus chip található:",
        ],
    ),
    (
        "I use it through Vulkan for acceleration",
        [
            "je l'utilise via Vulkan pour accélérer",
            "lo uso tramite Vulkan per accelerare",
            "lo uso mediante Vulkan para acelerar",
            "uso-o através do Vulkan para acelerar",
            "ich nutze ihn über Vulkan zur Beschleunigung",
            "Vulkanon keresztül használom a gyorsításhoz",
        ],
    ),
    (
        "No graphics card with compute support found: the model runs on the processor. That works, just more slowly.",
        [
            "Aucune carte graphique compatible avec le calcul : le modèle s'exécute sur le processeur. Cela fonctionne, mais plus lentement.",
            "Nessuna scheda grafica con supporto al calcolo: il modello gira sul processore. Funziona, solo più lentamente.",
            "No se encontró ninguna tarjeta gráfica con soporte de cálculo: el modelo se ejecuta en el procesador. Funciona, solo que más lento.",
            "Não foi encontrada nenhuma placa gráfica com suporte de cálculo: o modelo corre no processador. Funciona, só que mais devagar.",
            "Keine Grafikkarte mit Rechenunterstützung gefunden: Das Modell läuft auf dem Prozessor. Das funktioniert, nur langsamer.",
            "Nem található számításra képes videokártya: a modell a processzoron fut. Működik, csak lassabban.",
        ],
    ),
    (
        "Speed and acceleration",
        [
            "Vitesse et accélération",
            "Velocità e accelerazione",
            "Velocidad y aceleración",
            "Velocidade e aceleração",
            "Geschwindigkeit und Beschleunigung",
            "Sebesség és gyorsítás",
        ],
    ),
    (
        "How the model runs:",
        [
            "Comment le modèle s'exécute :",
            "Come gira il modello:",
            "Cómo se ejecuta el modelo:",
            "Como o modelo corre:",
            "Wie das Modell läuft:",
            "Hogyan fut a modell:",
        ],
    ),
    (
        "Automatic (best available)",
        [
            "Automatique (le meilleur disponible)",
            "Automatico (il migliore disponibile)",
            "Automático (el mejor disponible)",
            "Automático (o melhor disponível)",
            "Automatisch (das Beste, was verfügbar ist)",
            "Automatikus (a legjobb elérhető)",
        ],
    ),
    (
        "NVIDIA CUDA",
        [
            "NVIDIA CUDA",
            "NVIDIA CUDA",
            "NVIDIA CUDA",
            "NVIDIA CUDA",
            "NVIDIA CUDA",
            "NVIDIA CUDA",
        ],
    ),
    (
        "Vulkan (AMD, Intel, NVIDIA)",
        [
            "Vulkan (AMD, Intel, NVIDIA)",
            "Vulkan (AMD, Intel, NVIDIA)",
            "Vulkan (AMD, Intel, NVIDIA)",
            "Vulkan (AMD, Intel, NVIDIA)",
            "Vulkan (AMD, Intel, NVIDIA)",
            "Vulkan (AMD, Intel, NVIDIA)",
        ],
    ),
    (
        "Processor only",
        [
            "Processeur uniquement",
            "Solo processore",
            "Solo procesador",
            "Só processador",
            "Nur Prozessor",
            "Csak processzor",
        ],
    ),
    (
        "“Automatic” picks CUDA for an NVIDIA card, otherwise Vulkan, otherwise the processor, and rebuilds the engine if you add a graphics card later. A change applies the next time you install or switch a model.",
        [
            "« Automatique » choisit CUDA pour une carte NVIDIA, sinon Vulkan, sinon le processeur, et reconstruit le moteur si vous ajoutez une carte graphique plus tard. Un changement s'applique à la prochaine installation ou au prochain changement de modèle.",
            "«Automatico» sceglie CUDA per una scheda NVIDIA, altrimenti Vulkan, altrimenti il processore, e ricostruisce il motore se aggiungi una scheda grafica in seguito. Una modifica vale dalla prossima installazione o dal prossimo cambio di modello.",
            "«Automático» elige CUDA para una tarjeta NVIDIA, si no Vulkan, si no el procesador, y reconstruye el motor si añades una tarjeta gráfica más adelante. Un cambio se aplica la próxima vez que instales o cambies de modelo.",
            "«Automático» escolhe CUDA para uma placa NVIDIA, senão Vulkan, senão o processador, e reconstrói o motor se adicionares uma placa gráfica mais tarde. Uma alteração aplica-se na próxima instalação ou troca de modelo.",
            "„Automatisch“ wählt CUDA für eine NVIDIA-Karte, sonst Vulkan, sonst den Prozessor und baut die Engine neu, wenn du später eine Grafikkarte einbaust. Eine Änderung gilt bei der nächsten Installation oder dem nächsten Modellwechsel.",
            "Az „Automatikus” NVIDIA kártyához CUDA-t, egyébként Vulkant, egyébként a processzort választja, és újraépíti a motort, ha később videokártyát építesz be. A módosítás a következő telepítéskor vagy modellváltáskor lép életbe.",
        ],
    ),
    (
        "Dedicated AI chips (NPUs) are only experimental in colibri and stay off; this app doesn't switch them on behind your back.",
        [
            "Les puces d'IA dédiées (NPU) ne sont qu'expérimentales dans colibri et restent désactivées ; cette application ne les active pas en cachette.",
            "I chip dedicati all'IA (NPU) in colibri sono solo sperimentali e restano spenti; questa app non li attiva di nascosto.",
            "Los chips de IA dedicados (NPU) son solo experimentales en colibri y permanecen apagados; esta aplicación no los activa a escondidas.",
            "Os chips de IA dedicados (NPU) são apenas experimentais no colibri e ficam desligados; esta aplicação não os liga às escondidas.",
            "Dedizierte KI-Chips (NPUs) sind in colibri nur experimentell und bleiben aus; diese App schaltet sie nicht heimlich ein.",
            "A dedikált MI-chipek (NPU) a colibriban csak kísérleti jellegűek, és kikapcsolva maradnak; ez az alkalmazás nem kapcsolja be őket a hátad mögött.",
        ],
    ),
    (
        "Apply now to the current model",
        [
            "Appliquer maintenant au modèle actuel",
            "Applica ora al modello attuale",
            "Aplicar ahora al modelo actual",
            "Aplicar agora ao modelo atual",
            "Jetzt auf das aktuelle Modell anwenden",
            "Alkalmazás most az aktuális modellre",
        ],
    ),
    (
        "★ The most capable model for this computer",
        [
            "★ Le modèle le plus capable pour cet ordinateur",
            "★ Il modello più capace per questo computer",
            "★ El modelo más capaz para este ordenador",
            "★ O modelo mais capaz para este computador",
            "★ Das leistungsfähigste Modell für diesen Computer",
            "★ A legkapacitásosabb modell ehhez a számítógéphez",
        ],
    ),
    (
        "Your computer has little memory, yet this model still runs: the small, dense part stays in RAM and the rest is read from the SSD as needed. It answers more slowly than a small model, but it is the most capable one you can use. You need a fast SSD (NVMe) and free disk space.",
        [
            "Votre ordinateur a peu de mémoire, mais ce modèle fonctionne quand même : la petite partie dense reste en RAM et le reste est lu depuis le SSD au fur et à mesure. Il répond plus lentement qu'un petit modèle, mais c'est le plus capable que vous puissiez utiliser. Il faut un SSD rapide (NVMe) et de l'espace disque libre.",
            "Il tuo computer ha poca memoria, ma questo modello gira lo stesso: la parte piccola e densa resta nella RAM e il resto viene letto dall'SSD quando serve. Risponde più lentamente di un modello piccolo, ma è il più capace che puoi usare. Servono un SSD veloce (NVMe) e spazio libero sul disco.",
            "Tu ordenador tiene poca memoria, pero este modelo funciona igualmente: la parte pequeña y densa se queda en la RAM y el resto se lee del SSD según hace falta. Responde más despacio que un modelo pequeño, pero es el más capaz que puedes usar. Necesitas un SSD rápido (NVMe) y espacio libre en disco.",
            "O teu computador tem pouca memória, mas este modelo funciona na mesma: a parte pequena e densa fica na RAM e o resto é lido do SSD conforme é preciso. Responde mais devagar do que um modelo pequeno, mas é o mais capaz que podes usar. Precisas de um SSD rápido (NVMe) e de espaço livre em disco.",
            "Dein Computer hat wenig Speicher, doch dieses Modell läuft trotzdem: Der kleine, dichte Teil bleibt im RAM, der Rest wird bei Bedarf von der SSD gelesen. Es antwortet langsamer als ein kleines Modell, ist aber das leistungsfähigste, das du nutzen kannst. Du brauchst eine schnelle SSD (NVMe) und freien Speicherplatz.",
            "A számítógéped memóriája kevés, mégis fut ez a modell: a kicsi, sűrű rész a RAM-ban marad, a többit szükség szerint az SSD-ről olvassa. Lassabban válaszol, mint egy kis modell, de ez a legképesebb, amit használhatsz. Gyors SSD-re (NVMe) és szabad lemezterületre van szükség.",
        ],
    ),
    (
        "This model keeps its dense part in RAM and reads the rest from the SSD as needed; a fast NVMe drive helps a lot.",
        [
            "Ce modèle garde sa partie dense en RAM et lit le reste depuis le SSD au besoin ; un SSD NVMe rapide aide beaucoup.",
            "Questo modello tiene la parte densa nella RAM e legge il resto dall'SSD quando serve; un NVMe veloce aiuta molto.",
            "Este modelo mantiene su parte densa en la RAM y lee el resto del SSD según hace falta; un NVMe rápido ayuda mucho.",
            "Este modelo mantém a parte densa na RAM e lê o resto do SSD conforme é preciso; um NVMe rápido ajuda muito.",
            "Dieses Modell hält den dichten Teil im RAM und liest den Rest bei Bedarf von der SSD; eine schnelle NVMe-SSD hilft sehr.",
            "Ez a modell a sűrű részét a RAM-ban tartja, a többit szükség szerint az SSD-ről olvassa; egy gyors NVMe sokat segít.",
        ],
    ),
    (
        "It fits in this computer's memory, so it will answer quickly.",
        [
            "Il tient dans la mémoire de cet ordinateur, donc il répondra vite.",
            "Sta nella memoria di questo computer, quindi risponderà in fretta.",
            "Cabe en la memoria de este ordenador, así que responderá rápido.",
            "Cabe na memória deste computador, por isso responderá depressa.",
            "Es passt in den Speicher dieses Computers und antwortet daher schnell.",
            "Elfér a számítógép memóriájában, így gyorsan válaszol.",
        ],
    ),
    (
        "Resources and privacy",
        [
            "Ressources et confidentialité",
            "Risorse e privacy",
            "Recursos y privacidad",
            "Recursos e privacidade",
            "Ressourcen und Datenschutz",
            "Erőforrások és adatvédelem",
        ],
    ),
    (
        "Measuring…",
        [
            "Mesure en cours…",
            "Misurazione…",
            "Midiendo…",
            "A medir…",
            "Messe…",
            "Mérés…",
        ],
    ),
    (
        "What your computer is using now",
        [
            "Ce que votre ordinateur utilise en ce moment",
            "Cosa sta usando ora il tuo computer",
            "Qué está usando ahora tu ordenador",
            "O que o teu computador está a usar agora",
            "Was dein Computer gerade nutzt",
            "Mit használ most a számítógéped",
        ],
    ),
    (
        "Processor",
        [
            "Processeur",
            "Processore",
            "Procesador",
            "Processador",
            "Prozessor",
            "Processzor",
        ],
    ),
    (
        "cores",
        ["cœurs", "core", "núcleos", "núcleos", "Kerne", "mag"],
    ),
    (
        "Graphics card: its load can't be read (no NVIDIA card or no nvidia-smi). The model may still use a graphics chip through Vulkan.",
        [
            "Carte graphique : sa charge n'est pas lisible (pas de carte NVIDIA ou pas de nvidia-smi). Le modèle peut quand même utiliser une puce graphique via Vulkan.",
            "Scheda grafica: non riesco a leggerne il carico (nessuna scheda NVIDIA o nessun nvidia-smi). Il modello può comunque usare un chip grafico tramite Vulkan.",
            "Tarjeta gráfica: no puedo leer su carga (sin tarjeta NVIDIA o sin nvidia-smi). El modelo aún puede usar un chip gráfico mediante Vulkan.",
            "Placa gráfica: não consigo ler a sua carga (sem placa NVIDIA ou sem nvidia-smi). O modelo pode na mesma usar um chip gráfico através do Vulkan.",
            "Grafikkarte: Die Auslastung lässt sich nicht lesen (keine NVIDIA-Karte oder kein nvidia-smi). Das Modell kann trotzdem einen Grafikchip über Vulkan nutzen.",
            "Videokártya: a terhelése nem olvasható (nincs NVIDIA kártya vagy nvidia-smi). A modell ettől még használhat grafikus chipet Vulkanon keresztül.",
        ],
    ),
    (
        "Disk (models): free",
        [
            "Disque (modèles) : libre",
            "Disco (modelli): libero",
            "Disco (modelos): libre",
            "Disco (modelos): livre",
            "Festplatte (Modelle): frei",
            "Lemez (modellek): szabad",
        ],
    ),
    (
        "Who is using it",
        [
            "Qui les utilise",
            "Chi le usa",
            "Quién las usa",
            "Quem as usa",
            "Wer sie nutzt",
            "Ki használja",
        ],
    ),
    (
        "of a core",
        [
            "d'un cœur",
            "di un core",
            "de un núcleo",
            "de um núcleo",
            "eines Kerns",
            "egy magból",
        ],
    ),
    (
        "processes",
        [
            "processus",
            "processi",
            "procesos",
            "processos",
            "Prozesse",
            "folyamat",
        ],
    ),
    (
        "The colibri engine isn't running now, so it uses nothing.",
        [
            "Le moteur colibri ne tourne pas en ce moment, il ne consomme donc rien.",
            "Il motore colibri non è in esecuzione ora, quindi non consuma nulla.",
            "El motor colibri no está en marcha ahora, así que no consume nada.",
            "O motor colibri não está em execução agora, por isso não consome nada.",
            "Die colibri-Engine läuft gerade nicht und verbraucht daher nichts.",
            "A colibri motor most nem fut, ezért nem fogyaszt semmit.",
        ],
    ),
    (
        "I can't read the system's connections right now.",
        [
            "Je ne peux pas lire les connexions du système pour le moment.",
            "Al momento non riesco a leggere le connessioni del sistema.",
            "Ahora mismo no puedo leer las conexiones del sistema.",
            "Neste momento não consigo ler as ligações do sistema.",
            "Ich kann die Verbindungen des Systems gerade nicht lesen.",
            "Most nem tudom kiolvasni a rendszer kapcsolatait.",
        ],
    ),
    (
        "✔ No connection leaves this computer right now",
        [
            "✔ Aucune connexion ne quitte cet ordinateur en ce moment",
            "✔ Nessuna connessione esce da questo computer in questo momento",
            "✔ Ninguna conexión sale de este ordenador ahora mismo",
            "✔ Nenhuma ligação sai deste computador neste momento",
            "✔ Im Moment verlässt keine Verbindung diesen Computer",
            "✔ Jelenleg egyetlen kapcsolat sem hagyja el ezt a számítógépet",
        ],
    ),
    (
        "The app and the colibri engine only talk to each other, inside your computer. Your conversations are not sent anywhere.",
        [
            "L'application et le moteur colibri ne communiquent qu'entre eux, à l'intérieur de votre ordinateur. Vos conversations ne sont envoyées nulle part.",
            "L'app e il motore colibri parlano solo tra loro, dentro il tuo computer. Le tue conversazioni non vengono inviate da nessuna parte.",
            "La aplicación y el motor colibri solo hablan entre sí, dentro de tu ordenador. Tus conversaciones no se envían a ninguna parte.",
            "A aplicação e o motor colibri só falam entre si, dentro do teu computador. As tuas conversas não são enviadas para lado nenhum.",
            "Die App und die colibri-Engine sprechen nur miteinander, innerhalb deines Computers. Deine Unterhaltungen werden nirgendwohin gesendet.",
            "Az alkalmazás és a colibri motor csak egymással beszél, a számítógéped belsejében. A beszélgetéseid sehová sem kerülnek elküldésre.",
        ],
    ),
    (
        "You are using a remote server",
        [
            "Vous utilisez un serveur distant",
            "Stai usando un server remoto",
            "Estás usando un servidor remoto",
            "Estás a usar um servidor remoto",
            "Du verwendest einen entfernten Server",
            "Távoli szervert használsz",
        ],
    ),
    (
        "Your messages are sent to:",
        [
            "Vos messages sont envoyés à :",
            "I tuoi messaggi vengono inviati a:",
            "Tus mensajes se envían a:",
            "As tuas mensagens são enviadas para:",
            "Deine Nachrichten werden gesendet an:",
            "Az üzeneteid ide mennek:",
        ],
    ),
    (
        "You chose this in Settings. Go back to the local address (http://127.0.0.1:8000) to keep everything on this computer.",
        [
            "Vous l'avez choisi dans les Réglages. Revenez à l'adresse locale (http://127.0.0.1:8000) pour que tout reste sur cet ordinateur.",
            "L'hai scelto nelle Impostazioni. Torna all'indirizzo locale (http://127.0.0.1:8000) per tenere tutto su questo computer.",
            "Lo elegiste en Ajustes. Vuelve a la dirección local (http://127.0.0.1:8000) para que todo se quede en este ordenador.",
            "Escolheste isto nas Definições. Volta ao endereço local (http://127.0.0.1:8000) para que tudo fique neste computador.",
            "Das hast du in den Einstellungen gewählt. Kehre zur lokalen Adresse (http://127.0.0.1:8000) zurück, damit alles auf diesem Computer bleibt.",
            "Ezt a Beállításokban választottad. Térj vissza a helyi címre (http://127.0.0.1:8000), hogy minden ezen a számítógépen maradjon.",
        ],
    ),
    (
        "Something is downloading, because you asked for it",
        [
            "Un téléchargement est en cours, car vous l'avez demandé",
            "Un download è in corso, perché l'hai richiesto",
            "Hay una descarga en curso, porque la pediste",
            "Há uma transferência em curso, porque a pediste",
            "Etwas wird heruntergeladen, weil du es angefordert hast",
            "Valami letöltődik, mert te kérted",
        ],
    ),
    (
        "Heads up: a connection to the outside",
        [
            "Attention : une connexion vers l'extérieur",
            "Attenzione: una connessione verso l'esterno",
            "Atención: una conexión hacia el exterior",
            "Atenção: uma ligação para o exterior",
            "Achtung: eine Verbindung nach außen",
            "Figyelem: kapcsolat a külvilág felé",
        ],
    ),
    (
        "Downloading a model or colibri needs the internet. It stops by itself when done.",
        [
            "Télécharger un modèle ou colibri nécessite Internet. Cela s'arrête tout seul à la fin.",
            "Scaricare un modello o colibri richiede internet. Si ferma da solo al termine.",
            "Descargar un modelo o colibri necesita internet. Se detiene solo al terminar.",
            "Descarregar um modelo ou o colibri precisa de internet. Para sozinho quando termina.",
            "Das Herunterladen eines Modells oder von colibri braucht das Internet. Es endet von selbst.",
            "Egy modell vagy a colibri letöltéséhez internet kell. Magától leáll, ha kész.",
        ],
    ),
    (
        "A server is also listening for other computers on your network",
        [
            "Un serveur écoute aussi les autres ordinateurs de votre réseau",
            "Un server è in ascolto anche per gli altri computer della tua rete",
            "Un servidor también escucha a otros ordenadores de tu red",
            "Um servidor também escuta outros computadores da tua rede",
            "Ein Server lauscht auch auf andere Computer in deinem Netzwerk",
            "Egy szerver a hálózatod más számítógépeire is figyel",
        ],
    ),
    (
        "Since it started, the app opened {} connections, all to this computer.",
        [
            "Depuis son démarrage, l'application a ouvert {} connexions, toutes vers cet ordinateur.",
            "Dall'avvio l'app ha aperto {} connessioni, tutte verso questo computer.",
            "Desde que arrancó, la aplicación abrió {} conexiones, todas hacia este ordenador.",
            "Desde que arrancou, a aplicação abriu {} ligações, todas para este computador.",
            "Seit dem Start hat die App {} Verbindungen geöffnet, alle zu diesem Computer.",
            "Az indulás óta az alkalmazás {} kapcsolatot nyitott, mind ehhez a számítógéphez.",
        ],
    ),
    (
        "Since it started, the app opened {} connections; {} to other addresses:",
        [
            "Depuis son démarrage, l'application a ouvert {} connexions ; {} vers d'autres adresses :",
            "Dall'avvio l'app ha aperto {} connessioni; {} verso altri indirizzi:",
            "Desde que arrancó, la aplicación abrió {} conexiones; {} hacia otras direcciones:",
            "Desde que arrancou, a aplicação abriu {} ligações; {} para outros endereços:",
            "Seit dem Start hat die App {} Verbindungen geöffnet; {} zu anderen Adressen:",
            "Az indulás óta az alkalmazás {} kapcsolatot nyitott; ebből {} más címekre:",
        ],
    ),
    (
        "Show the evidence (every connection)",
        [
            "Afficher la preuve (toutes les connexions)",
            "Mostra la prova (tutte le connessioni)",
            "Mostrar la prueba (todas las conexiones)",
            "Mostrar a prova (todas as ligações)",
            "Beweis anzeigen (alle Verbindungen)",
            "A bizonyíték mutatása (minden kapcsolat)",
        ],
    ),
    (
        "These processes have no open sockets.",
        [
            "Ces processus n'ont aucun socket ouvert.",
            "Questi processi non hanno socket aperti.",
            "Estos procesos no tienen sockets abiertos.",
            "Estes processos não têm sockets abertos.",
            "Diese Prozesse haben keine offenen Sockets.",
            "Ezeknek a folyamatoknak nincs nyitott socketje.",
        ],
    ),
    (
        "Process",
        [
            "Processus",
            "Processo",
            "Proceso",
            "Processo",
            "Prozess",
            "Folyamat",
        ],
    ),
    ("Kind", ["Type", "Tipo", "Tipo", "Tipo", "Art", "Típus"]),
    (
        "Local",
        ["Local", "Locale", "Local", "Local", "Lokal", "Helyi"],
    ),
    ("To", ["Vers", "Verso", "Hacia", "Para", "Nach", "Cél"]),
    (
        "listening",
        [
            "en écoute",
            "in ascolto",
            "escuchando",
            "à escuta",
            "wartet",
            "figyel",
        ],
    ),
    (
        "external",
        [
            "externe", "esterna", "externa", "externa", "extern", "külső",
        ],
    ),
    (
        "local",
        ["locale", "locale", "local", "local", "lokal", "helyi"],
    ),
    (
        "You can also check for yourself with your system's own tools, without relying on this app:",
        [
            "Vous pouvez aussi vérifier vous-même avec les outils de votre système, sans vous fier à cette application :",
            "Puoi anche verificare da solo con gli strumenti del tuo sistema, senza fidarti di questa app:",
            "También puedes comprobarlo tú mismo con las herramientas de tu sistema, sin fiarte de esta aplicación:",
            "Também podes verificar tu próprio com as ferramentas do teu sistema, sem confiares nesta aplicação:",
            "Du kannst es auch selbst mit den Werkzeugen deines Systems prüfen, ohne dich auf diese App zu verlassen:",
            "Magad is ellenőrizheted a rendszered saját eszközeivel, anélkül, hogy erre az alkalmazásra hagyatkoznál:",
        ],
    ),
    (
        "This is a snapshot of right now. The only times the internet is used are the ones you ask for: downloading colibri, downloading a model, or checking for updates.",
        [
            "Ceci est un instantané du moment présent. Internet n'est utilisé que lorsque vous le demandez : télécharger colibri, télécharger un modèle ou chercher des mises à jour.",
            "Questa è un'istantanea del momento. Internet viene usato solo quando lo chiedi tu: scaricare colibri, scaricare un modello o cercare aggiornamenti.",
            "Esto es una instantánea del momento actual. Internet solo se usa cuando tú lo pides: descargar colibri, descargar un modelo o buscar actualizaciones.",
            "Isto é um retrato do momento atual. A internet só é usada quando tu pedes: descarregar o colibri, descarregar um modelo ou procurar atualizações.",
            "Das ist eine Momentaufnahme. Das Internet wird nur genutzt, wenn du es verlangst: colibri herunterladen, ein Modell herunterladen oder nach Updates suchen.",
            "Ez a pillanatnyi állapot. Az internetet csak akkor használja, amikor te kéred: a colibri letöltése, egy modell letöltése vagy a frissítések keresése.",
        ],
    ),
    (
        "colibri has a new version ({}). Updating brings speed-ups and new models.",
        [
            "colibri a une nouvelle version ({}). La mise à jour apporte de la vitesse et de nouveaux modèles.",
            "colibri ha una nuova versione ({}). L'aggiornamento porta più velocità e nuovi modelli.",
            "colibri tiene una versión nueva ({}). Actualizar trae más velocidad y modelos nuevos.",
            "O colibri tem uma versão nova ({}). Atualizar traz mais velocidade e modelos novos.",
            "colibri hat eine neue Version ({}). Das Update bringt mehr Tempo und neue Modelle.",
            "A colibrinek új verziója van ({}). A frissítés gyorsabbá tesz, és új modelleket hoz.",
        ],
    ),
    (
        "Stop the model and update",
        [
            "Arrêter le modèle et mettre à jour",
            "Ferma il modello e aggiorna",
            "Detener el modelo y actualizar",
            "Parar o modelo e atualizar",
            "Modell stoppen und aktualisieren",
            "Modell leállítása és frissítés",
        ],
    ),
    (
        "Update colibri",
        [
            "Mettre à jour colibri",
            "Aggiorna colibri",
            "Actualizar colibri",
            "Atualizar o colibri",
            "colibri aktualisieren",
            "A colibri frissítése",
        ],
    ),
    (
        "Colibri Companion {} is available.",
        [
            "Colibri Companion {} est disponible.",
            "Colibri Companion {} è disponibile.",
            "Colibri Companion {} está disponible.",
            "O Colibri Companion {} está disponível.",
            "Colibri Companion {} ist verfügbar.",
            "A Colibri Companion {} elérhető.",
        ],
    ),
    (
        "See the download",
        [
            "Voir le téléchargement",
            "Vedi il download",
            "Ver la descarga",
            "Ver a transferência",
            "Download ansehen",
            "Letöltés megtekintése",
        ],
    ),
    (
        "Later",
        [
            "Plus tard",
            "Più tardi",
            "Más tarde",
            "Mais tarde",
            "Später",
            "Később",
        ],
    ),
    (
        "At most 3 pictures at a time",
        [
            "3 images au maximum à la fois",
            "Al massimo 3 immagini alla volta",
            "Máximo 3 imágenes a la vez",
            "No máximo 3 imagens de cada vez",
            "Höchstens 3 Bilder auf einmal",
            "Egyszerre legfeljebb 3 kép",
        ],
    ),
    (
        "Picture «{}» goes with your message. It only works with models that can see pictures.",
        [
            "L'image «{}» part avec votre message. Cela ne marche qu'avec les modèles qui voient les images.",
            "L'immagine «{}» parte con il messaggio. Funziona solo con i modelli che vedono le immagini.",
            "La imagen «{}» se envía con tu mensaje. Solo funciona con modelos que ven imágenes.",
            "A imagem «{}» vai com a sua mensagem. Só funciona com modelos que veem imagens.",
            "Das Bild «{}» geht mit Ihrer Nachricht mit. Das klappt nur mit Modellen, die Bilder sehen können.",
            "A(z) «{}» kép az üzenettel együtt megy. Csak olyan modellekkel működik, amelyek látnak képeket.",
        ],
    ),
    (
        "Reading «{}»…",
        [
            "Lecture de «{}»…",
            "Leggo «{}»…",
            "Leyendo «{}»…",
            "A ler «{}»…",
            "«{}» wird gelesen…",
            "«{}» olvasása…",
        ],
    ),
    (
        "I can't read the file",
        [
            "je ne peux pas lire le fichier",
            "non riesco a leggere il file",
            "no puedo leer el archivo",
            "não consigo ler o ficheiro",
            "ich kann die Datei nicht lesen",
            "nem tudom olvasni a fájlt",
        ],
    ),
    (
        "it is too large",
        [
            "il est trop volumineux",
            "è troppo grande",
            "es demasiado grande",
            "é demasiado grande",
            "sie ist zu groß",
            "túl nagy",
        ],
    ),
    (
        "it is not text or a picture I can read",
        [
            "ce n'est ni du texte ni une image que je sais lire",
            "non è testo né un'immagine che so leggere",
            "no es texto ni una imagen que sepa leer",
            "não é texto nem uma imagem que eu saiba ler",
            "es ist weder Text noch ein Bild, das ich lesen kann",
            "nem szöveg, sem olyan kép, amit el tudok olvasni",
        ],
    ),
    (
        "I need {} to read it",
        [
            "il me faut {} pour le lire",
            "mi serve {} per leggerlo",
            "necesito {} para leerlo",
            "preciso de {} para o ler",
            "ich brauche {}, um sie zu lesen",
            "{} kell az olvasásához",
        ],
    ),
    (
        "it failed",
        [
            "échec",
            "non è riuscito",
            "no ha funcionado",
            "falhou",
            "es hat nicht geklappt",
            "nem sikerült",
        ],
    ),
    (
        "Added «{}» (the long text was cut)",
        [
            "«{}» ajouté (le long texte a été coupé)",
            "Aggiunto «{}» (il testo lungo è stato tagliato)",
            "Añadido «{}» (el texto largo se ha recortado)",
            "«{}» adicionado (o texto longo foi cortado)",
            "«{}» hinzugefügt (der lange Text wurde gekürzt)",
            "«{}» hozzáadva (a hosszú szöveg le lett vágva)",
        ],
    ),
    (
        "Could not open the preview",
        [
            "Impossible d'ouvrir l'aperçu",
            "Impossibile aprire l'anteprima",
            "No se pudo abrir la vista previa",
            "Não foi possível abrir a pré-visualização",
            "Die Vorschau konnte nicht geöffnet werden",
            "Nem sikerült megnyitni az előnézetet",
        ],
    ),
    (
        "Code",
        ["Code", "Codice", "Código", "Código", "Code", "Kód"],
    ),
    (
        "Preview",
        [
            "Aperçu",
            "Anteprima",
            "Vista previa",
            "Pré-visualização",
            "Vorschau",
            "Előnézet",
        ],
    ),
    (
        "The page opens in your browser and may run scripts the model wrote. Open it only if you trust what you see above.",
        [
            "La page s'ouvre dans votre navigateur et peut exécuter des scripts écrits par le modèle. N'ouvrez que si vous faites confiance à ce que vous voyez ci-dessus.",
            "La pagina si apre nel browser e può eseguire script scritti dal modello. Aprila solo se ti fidi di ciò che vedi sopra.",
            "La página se abre en tu navegador y puede ejecutar scripts escritos por el modelo. Ábrela solo si confías en lo que ves arriba.",
            "A página abre no navegador e pode executar scripts escritos pelo modelo. Abra apenas se confiar no que vê acima.",
            "Die Seite öffnet sich im Browser und kann vom Modell geschriebene Skripte ausführen. Öffnen Sie sie nur, wenn Sie dem oben Gezeigten vertrauen.",
            "Az oldal a böngészőben nyílik meg, és futtathat a modell írta szkripteket. Csak akkor nyisd meg, ha megbízol a fentiekben.",
        ],
    ),
    (
        "Open anyway",
        [
            "Ouvrir quand même",
            "Apri comunque",
            "Abrir de todos modos",
            "Abrir mesmo assim",
            "Trotzdem öffnen",
            "Megnyitás mégis",
        ],
    ),
    (
        "I couldn't find colibri on this computer, so I'm downloading it now.",
        [
            "Je n'ai pas trouvé colibri sur cet ordinateur, je le télécharge donc maintenant.",
            "Non ho trovato colibri su questo computer, quindi lo sto scaricando ora.",
            "No encontré colibri en este ordenador, así que lo estoy descargando ahora.",
            "Não encontrei o colibri neste computador, por isso estou a descarregá-lo agora.",
            "Ich habe colibri auf diesem Computer nicht gefunden und lade es jetzt herunter.",
            "Nem találtam colibrit ezen a gépen, ezért most letöltöm.",
        ],
    ),
    (
        "Apple Silicon Mac: I use the Mac's graphics chip through Metal (unified memory).",
        [
            "Mac Apple Silicon : j'utilise la puce graphique du Mac via Metal (mémoire unifiée).",
            "Mac Apple Silicon: uso il chip grafico del Mac tramite Metal (memoria unificata).",
            "Mac con Apple Silicon: uso el chip gráfico del Mac mediante Metal (memoria unificada).",
            "Mac com Apple Silicon: uso o chip gráfico do Mac através do Metal (memória unificada).",
            "Apple-Silicon-Mac: Ich nutze den Grafikchip des Mac über Metal (einheitlicher Speicher).",
            "Apple Silicon Mac: a Mac grafikus chipjét használom Metalon keresztül (egységes memória).",
        ],
    ),
    (
        "Use the Mac's graphics chip (Metal, experimental in colibri)",
        [
            "Utiliser la puce graphique du Mac (Metal, expérimental dans colibri)",
            "Usa il chip grafico del Mac (Metal, sperimentale in colibri)",
            "Usar el chip gráfico del Mac (Metal, experimental en colibri)",
            "Usar o chip gráfico do Mac (Metal, experimental no colibri)",
            "Grafikchip des Mac nutzen (Metal, in colibri experimentell)",
            "A Mac grafikus chipjének használata (Metal, a colibriben kísérleti)",
        ],
    ),
    (
        "AI chip found:",
        [
            "Puce d'IA trouvée :",
            "Chip di IA trovato:",
            "Chip de IA encontrado:",
            "Chip de IA encontrado:",
            "KI-Chip gefunden:",
            "MI-chip találva:",
        ],
    ),
    (
        "Use the AI chip (NPU), experimental: reduced precision, answers may differ",
        [
            "Utiliser la puce d'IA (NPU), expérimental : précision réduite, les réponses peuvent différer",
            "Usa il chip di IA (NPU), sperimentale: precisione ridotta, le risposte possono differire",
            "Usar el chip de IA (NPU), experimental: precisión reducida, las respuestas pueden variar",
            "Usar o chip de IA (NPU), experimental: precisão reduzida, as respostas podem diferir",
            "KI-Chip (NPU) nutzen, experimentell: geringere Genauigkeit, Antworten können abweichen",
            "Az MI-chip (NPU) használata, kísérleti: csökkentett pontosság, a válaszok eltérhetnek",
        ],
    ),
    (
        "It only works with GLM models and needs colibri's XDNA package; if it isn't usable, colibri carries on the normal way.",
        [
            "Cela ne marche qu'avec les modèles GLM et demande le paquet XDNA de colibri ; s'il n'est pas utilisable, colibri continue normalement.",
            "Funziona solo con i modelli GLM e richiede il pacchetto XDNA di colibri; se non è utilizzabile, colibri prosegue normalmente.",
            "Solo funciona con modelos GLM y necesita el paquete XDNA de colibri; si no se puede usar, colibri sigue de la forma habitual.",
            "Só funciona com modelos GLM e precisa do pacote XDNA do colibri; se não for utilizável, o colibri segue pelo caminho normal.",
            "Es funktioniert nur mit GLM-Modellen und braucht das XDNA-Paket von colibri; ist es nicht nutzbar, läuft colibri normal weiter.",
            "Csak GLM modellekkel működik, és a colibri XDNA csomagját igényli; ha nem használható, a colibri a megszokott módon megy tovább.",
        ],
    ),
    (
        "colibri can't use this chip yet; the model runs on the processor or the graphics card.",
        [
            "colibri ne peut pas encore utiliser cette puce ; le modèle tourne sur le processeur ou la carte graphique.",
            "colibri non può ancora usare questo chip; il modello gira sul processore o sulla scheda video.",
            "colibri aún no puede usar este chip; el modelo se ejecuta en el procesador o la tarjeta gráfica.",
            "O colibri ainda não pode usar este chip; o modelo corre no processador ou na placa gráfica.",
            "colibri kann diesen Chip noch nicht nutzen; das Modell läuft auf dem Prozessor oder der Grafikkarte.",
            "A colibri még nem tudja használni ezt a chipet; a modell a processzoron vagy a videokártyán fut.",
        ],
    ),
    (
        "No dedicated AI chip (NPU) found. If there is one, colibri only supports it experimentally, and this app doesn't switch it on behind your back.",
        [
            "Aucune puce d'IA dédiée (NPU) trouvée. S'il y en a une, colibri ne la gère qu'à titre expérimental, et cette application ne l'active pas en cachette.",
            "Nessun chip di IA dedicato (NPU) trovato. Se c'è, colibri lo supporta solo in modo sperimentale e questa app non lo attiva di nascosto.",
            "No se encontró ningún chip de IA dedicado (NPU). Si lo hay, colibri solo lo admite de forma experimental y esta aplicación no lo activa a escondidas.",
            "Não foi encontrado nenhum chip de IA dedicado (NPU). Se existir, o colibri só o suporta de forma experimental e esta aplicação não o liga às escondidas.",
            "Kein dedizierter KI-Chip (NPU) gefunden. Falls es einen gibt, unterstützt colibri ihn nur experimentell, und diese App schaltet ihn nicht heimlich ein.",
            "Nem találtam dedikált MI-chipet (NPU). Ha van, a colibri csak kísérleti jelleggel támogatja, és ez az alkalmazás nem kapcsolja be a háttérben.",
        ],
    ),
    (
        "Checked just now: {} processes (the app and the colibri engine), {} open sockets.",
        [
            "Vérifié à l'instant : {} processus (l'application et le moteur colibri), {} connexions ouvertes.",
            "Controllato ora: {} processi (l'app e il motore colibri), {} socket aperti.",
            "Comprobado ahora: {} procesos (la aplicación y el motor colibri), {} sockets abiertos.",
            "Verificado agora: {} processos (a aplicação e o motor colibri), {} sockets abertos.",
            "Gerade geprüft: {} Prozesse (die App und die colibri-Engine), {} offene Sockets.",
            "Az imént ellenőrizve: {} folyamat (az alkalmazás és a colibri motor), {} nyitott socket.",
        ],
    ),
    (
        "Legal",
        [
            "Mentions légales",
            "Note legali",
            "Legal",
            "Legal",
            "Rechtliches",
            "Jogi tudnivalók",
        ],
    ),
    (
        "Tips",
        [
            "Conseils", "Consigli", "Consejos", "Dicas", "Tipps", "Tippek",
        ],
    ),
    (
        "Explore",
        [
            "Explorer",
            "Esplora",
            "Explorar",
            "Explorar",
            "Entdecken",
            "Felfedezés",
        ],
    ),
    (
        "Bold",
        [
            "Gras",
            "Grassetto",
            "Negrita",
            "Negrito",
            "Fett",
            "Félkövér",
        ],
    ),
    (
        "Italic",
        [
            "Italique", "Corsivo", "Cursiva", "Itálico", "Kursiv", "Dőlt",
        ],
    ),
    (
        "Superscript",
        [
            "Exposant",
            "Apice",
            "Superíndice",
            "Expoente",
            "Hochgestellt",
            "Felső index",
        ],
    ),
    (
        "Subscript",
        [
            "Indice",
            "Pedice",
            "Subíndice",
            "Índice inferior",
            "Tiefgestellt",
            "Alsó index",
        ],
    ),
    (
        "Inline code",
        [
            "Code en ligne",
            "Codice in linea",
            "Código en línea",
            "Código em linha",
            "Inline-Code",
            "Soron belüli kód",
        ],
    ),
    (
        "Code block",
        [
            "Bloc de code",
            "Blocco di codice",
            "Bloque de código",
            "Bloco de código",
            "Codeblock",
            "Kódblokk",
        ],
    ),
    (
        "Heading 1",
        [
            "Titre 1",
            "Titolo 1",
            "Título 1",
            "Título 1",
            "Überschrift 1",
            "1. címsor",
        ],
    ),
    (
        "Heading 2",
        [
            "Titre 2",
            "Titolo 2",
            "Título 2",
            "Título 2",
            "Überschrift 2",
            "2. címsor",
        ],
    ),
    (
        "Heading 3",
        [
            "Titre 3",
            "Titolo 3",
            "Título 3",
            "Título 3",
            "Überschrift 3",
            "3. címsor",
        ],
    ),
    (
        "Bulleted list",
        [
            "Liste à puces",
            "Elenco puntato",
            "Lista con viñetas",
            "Lista com marcadores",
            "Aufzählung",
            "Felsorolás",
        ],
    ),
    (
        "Numbered list",
        [
            "Liste numérotée",
            "Elenco numerato",
            "Lista numerada",
            "Lista numerada",
            "Nummerierte Liste",
            "Számozott lista",
        ],
    ),
    (
        "Checkbox list",
        [
            "Liste à cocher",
            "Elenco con caselle",
            "Lista de casillas",
            "Lista de verificação",
            "Checkliste",
            "Jelölőnégyzetes lista",
        ],
    ),
    (
        "Quote",
        [
            "Citation",
            "Citazione",
            "Cita",
            "Citação",
            "Zitat",
            "Idézet",
        ],
    ),
    (
        "Dash line",
        [
            "Ligne de séparation",
            "Linea di separazione",
            "Línea separadora",
            "Linha separadora",
            "Trennlinie",
            "Elválasztó vonal",
        ],
    ),
    (
        "Link",
        [
            "Lien",
            "Collegamento",
            "Enlace",
            "Ligação",
            "Link",
            "Hivatkozás",
        ],
    ),
    (
        "Table",
        [
            "Tableau",
            "Tabella",
            "Tabla",
            "Tabela",
            "Tabelle",
            "Táblázat",
        ],
    ),
    (
        "Math",
        [
            "Formule mathématique",
            "Formula matematica",
            "Fórmula matemática",
            "Fórmula matemática",
            "Mathe-Formel",
            "Matematikai képlet",
        ],
    ),
    (
        "Emoji",
        ["Émoji", "Emoji", "Emoji", "Emoji", "Emoji", "Emoji"],
    ),
    (
        "File",
        ["Fichier", "File", "Archivo", "Ficheiro", "Datei", "Fájl"],
    ),
    (
        "Table size:",
        [
            "Taille du tableau :",
            "Dimensione tabella:",
            "Tamaño de la tabla:",
            "Tamanho da tabela:",
            "Tabellengröße:",
            "Táblázat mérete:",
        ],
    ),
    (
        "Drop a file on the window, or type its path:",
        [
            "Déposez un fichier sur la fenêtre ou saisissez son chemin :",
            "Trascina un file sulla finestra o scrivi il suo percorso:",
            "Suelta un archivo en la ventana o escribe su ruta:",
            "Largue um ficheiro na janela ou escreva o seu caminho:",
            "Ziehen Sie eine Datei auf das Fenster oder geben Sie ihren Pfad ein:",
            "Húzz egy fájlt az ablakra, vagy írd be az útvonalát:",
        ],
    ),
    (
        "Add",
        [
            "Ajouter",
            "Aggiungi",
            "Añadir",
            "Adicionar",
            "Hinzufügen",
            "Hozzáadás",
        ],
    ),
    (
        "AI models can make mistakes. For medical, legal or financial matters, ask a specialist.",
        [
            "Les modèles d'IA peuvent se tromper. Pour les questions médicales, juridiques ou financières, consultez un spécialiste.",
            "I modelli di IA possono sbagliare. Per questioni mediche, legali o finanziarie, rivolgiti a uno specialista.",
            "Los modelos de IA pueden equivocarse. Para temas médicos, legales o financieros, consulta a un especialista.",
            "Os modelos de IA podem errar. Em assuntos médicos, jurídicos ou financeiros, consulte um especialista.",
            "KI-Modelle können Fehler machen. Bei medizinischen, rechtlichen oder finanziellen Fragen fragen Sie eine Fachperson.",
            "Az MI-modellek tévedhetnek. Orvosi, jogi vagy pénzügyi kérdésekben fordulj szakemberhez.",
        ],
    ),
    (
        "Other free programs that run AI on your own computer, like colibri. They open in your browser only when you click. Check each site yourself; this is not a complete catalogue.",
        [
            "D'autres programmes gratuits qui font tourner l'IA sur votre ordinateur, comme colibri. Ils s'ouvrent dans votre navigateur seulement si vous cliquez. Vérifiez chaque site vous-même ; ce n'est pas un catalogue complet.",
            "Altri programmi gratuiti che eseguono l'IA sul tuo computer, come colibri. Si aprono nel browser solo quando fai clic. Controlla tu stesso ogni sito; non è un catalogo completo.",
            "Otros programas gratuitos que ejecutan IA en tu propio ordenador, como colibri. Se abren en el navegador solo cuando haces clic. Revisa cada sitio tú mismo; no es un catálogo completo.",
            "Outros programas gratuitos que executam IA no seu próprio computador, como o colibri. Só abrem no navegador quando clica. Verifique cada site por si; não é um catálogo completo.",
            "Weitere kostenlose Programme, die KI auf Ihrem eigenen Computer ausführen, wie colibri. Sie öffnen sich im Browser nur auf Klick. Prüfen Sie jede Seite selbst; dies ist kein vollständiger Katalog.",
            "További ingyenes programok, amelyek a saját gépeden futtatnak MI-t, mint a colibri. Csak kattintásra nyílnak meg a böngészőben. Ellenőrizd magad az oldalakat; ez nem teljes katalógus.",
        ],
    ),
    (
        "Shortcuts",
        [
            "Raccourcis",
            "Scorciatoie",
            "Atajos",
            "Atalhos",
            "Tastenkürzel",
            "Gyorsbillentyűk",
        ],
    ),
    (
        "Privacy and terms",
        [
            "Confidentialité et conditions",
            "Privacy e condizioni",
            "Privacidad y condiciones",
            "Privacidade e condições",
            "Datenschutz und Bedingungen",
            "Adatvédelem és feltételek",
        ],
    ),
    (
        "Be specific: say what you want, for whom, and in what form.",
        [
            "Soyez précis : dites ce que vous voulez, pour qui et sous quelle forme.",
            "Sii preciso: di' cosa vuoi, per chi e in quale forma.",
            "Sé específico: di qué quieres, para quién y en qué forma.",
            "Seja específico: diga o que quer, para quem e em que formato.",
            "Seien Sie konkret: sagen Sie, was Sie wollen, für wen und in welcher Form.",
            "Légy pontos: mondd meg, mit szeretnél, kinek és milyen formában.",
        ],
    ),
    (
        "Ask for rewrites: “shorter”, “simpler”, “warmer”.",
        [
            "Demandez des reformulations : « plus court », « plus simple », « plus chaleureux ».",
            "Chiedi riscritture: «più breve», «più semplice», «più caldo».",
            "Pide reescrituras: «más corto», «más simple», «más cálido».",
            "Peça reescritas: «mais curto», «mais simples», «mais caloroso».",
            "Bitten Sie um Umformulierungen: „kürzer“, „einfacher“, „wärmer“.",
            "Kérj átírást: „rövidebben”, „egyszerűbben”, „melegebben”.",
        ],
    ),
    (
        "Give it a role: “you are a history teacher explaining to a student”.",
        [
            "Donnez-lui un rôle : « tu es un professeur d'histoire qui explique à un élève ».",
            "Dagli un ruolo: «sei un insegnante di storia che spiega a uno studente».",
            "Dale un papel: «eres un profesor de historia que explica a un alumno».",
            "Dê-lhe um papel: «és um professor de história a explicar a um aluno».",
            "Geben Sie ihm eine Rolle: „Du bist ein Geschichtslehrer, der einem Schüler etwas erklärt“.",
            "Adj neki szerepet: „történelemtanár vagy, aki egy diáknak magyaráz”.",
        ],
    ),
    (
        "Check important facts: a local model can be wrong.",
        [
            "Vérifiez les informations importantes : un modèle local peut se tromper.",
            "Verifica le informazioni importanti: un modello locale può sbagliare.",
            "Comprueba los datos importantes: un modelo local puede equivocarse.",
            "Verifique as informações importantes: um modelo local pode errar.",
            "Prüfen Sie wichtige Angaben: Auch ein lokales Modell kann sich irren.",
            "Ellenőrizd a fontos információkat: egy helyi modell is tévedhet.",
        ],
    ),
    (
        "Compare models: install two from the Models window and try the same question on both.",
        [
            "Comparez les modèles : installez-en deux depuis la fenêtre Modèles et posez-leur la même question.",
            "Confronta i modelli: installane due dalla finestra Modelli e prova la stessa domanda su entrambi.",
            "Compara modelos: instala dos desde la ventana Modelos y prueba la misma pregunta en ambos.",
            "Compare modelos: instale dois na janela Modelos e experimente a mesma pergunta nos dois.",
            "Vergleichen Sie Modelle: Installieren Sie zwei im Modelle-Fenster und stellen Sie beiden dieselbe Frage.",
            "Hasonlíts össze modelleket: telepíts kettőt a Modellek ablakból, és tedd fel mindkettőnek ugyanazt a kérdést.",
        ],
    ),
    (
        "send the message",
        [
            "envoyer le message",
            "invia il messaggio",
            "enviar el mensaje",
            "enviar a mensagem",
            "Nachricht senden",
            "üzenet elküldése",
        ],
    ),
    (
        "new line",
        [
            "nouvelle ligne",
            "nuova riga",
            "nueva línea",
            "nova linha",
            "neue Zeile",
            "új sor",
        ],
    ),
    (
        "new conversation",
        [
            "nouvelle conversation",
            "nuova conversazione",
            "nueva conversación",
            "nova conversa",
            "neue Unterhaltung",
            "új beszélgetés",
        ],
    ),
    (
        "settings",
        [
            "réglages",
            "impostazioni",
            "ajustes",
            "definições",
            "Einstellungen",
            "beállítások",
        ],
    ),
    (
        "Privacy / GDPR",
        [
            "Confidentialité / RGPD",
            "Privacy / GDPR",
            "Privacidad / RGPD",
            "Privacidade / RGPD",
            "Datenschutz / DSGVO",
            "Adatvédelem / GDPR",
        ],
    ),
    (
        "Cookies",
        [
            "Cookies", "Cookie", "Cookies", "Cookies", "Cookies", "Sütik",
        ],
    ),
    (
        "Terms",
        [
            "Conditions",
            "Condizioni",
            "Condiciones",
            "Condições",
            "Bedingungen",
            "Feltételek",
        ],
    ),
    (
        "The app has no accounts, sends no telemetry and calls no outside AI servers. Conversations stay in the app's data folder on your computer. The only optional internet connection is the update check on GitHub, which you can switch off in Settings; the Resources window shows the evidence of connections.",
        [
            "L'application n'a pas de comptes, n'envoie aucune télémétrie et n'appelle aucun serveur d'IA extérieur. Les conversations restent dans le dossier de données de l'application, sur votre ordinateur. La seule connexion internet facultative est la vérification des mises à jour sur GitHub, que vous pouvez désactiver dans les Réglages ; la fenêtre Ressources montre la preuve des connexions.",
            "L'app non ha account, non invia telemetria e non chiama server di IA esterni. Le conversazioni restano nella cartella dati dell'app, sul tuo computer. L'unica connessione a internet facoltativa è il controllo degli aggiornamenti su GitHub, che puoi disattivare nelle Impostazioni; la finestra Risorse mostra la prova delle connessioni.",
            "La aplicación no tiene cuentas, no envía telemetría y no llama a servidores de IA externos. Las conversaciones se quedan en la carpeta de datos de la aplicación, en tu ordenador. La única conexión a internet opcional es la comprobación de actualizaciones en GitHub, que puedes desactivar en Ajustes; la ventana Recursos muestra la prueba de las conexiones.",
            "A aplicação não tem contas, não envia telemetria e não chama servidores de IA externos. As conversas ficam na pasta de dados da aplicação, no seu computador. A única ligação opcional à internet é a verificação de atualizações no GitHub, que pode desativar nas Definições; a janela Recursos mostra a prova das ligações.",
            "Die App hat keine Konten, sendet keine Telemetrie und ruft keine externen KI-Server auf. Unterhaltungen bleiben im Datenordner der App auf Ihrem Computer. Die einzige optionale Internetverbindung ist die Update-Prüfung auf GitHub, die Sie in den Einstellungen abschalten können; das Fenster Ressourcen zeigt den Nachweis der Verbindungen.",
            "Az alkalmazásnak nincsenek fiókjai, nem küld telemetriát, és nem hív külső MI-szervereket. A beszélgetések az alkalmazás adatmappájában maradnak, a te gépeden. Az egyetlen választható internetkapcsolat a GitHubon végzett frissítésellenőrzés, amelyet a Beállításokban kikapcsolhatsz; az Erőforrások ablak mutatja a kapcsolatok bizonyítékát.",
        ],
    ),
    (
        "The desktop app uses no cookies.",
        [
            "L'application de bureau n'utilise aucun cookie.",
            "L'app desktop non usa cookie.",
            "La aplicación de escritorio no usa cookies.",
            "A aplicação de ambiente de trabalho não usa cookies.",
            "Die Desktop-App verwendet keine Cookies.",
            "Az asztali alkalmazás nem használ sütiket.",
        ],
    ),
    (
        "The program is provided as is, without warranty, under the Apache-2.0 licence. Model answers can be wrong; you are responsible for how you use them.",
        [
            "Le programme est fourni tel quel, sans garantie, sous licence Apache-2.0. Les réponses des modèles peuvent être fausses ; vous êtes responsable de leur usage.",
            "Il programma è fornito così com'è, senza garanzie, con licenza Apache-2.0. Le risposte dei modelli possono essere sbagliate; sei responsabile di come le usi.",
            "El programa se ofrece tal cual, sin garantías, bajo la licencia Apache-2.0. Las respuestas de los modelos pueden ser erróneas; eres responsable de cómo las uses.",
            "O programa é fornecido tal como está, sem garantias, sob a licença Apache-2.0. As respostas dos modelos podem estar erradas; é responsável pela forma como as usa.",
            "Das Programm wird wie besehen, ohne Gewähr, unter der Apache-2.0-Lizenz bereitgestellt. Antworten der Modelle können falsch sein; Sie sind für ihre Verwendung verantwortlich.",
            "A programot jelenlegi állapotában, garancia nélkül, az Apache-2.0 licenc alatt kapod. A modellek válaszai tévedhetnek; te felelsz a felhasználásukért.",
        ],
    ),
    (
        "Where models are saved",
        [
            "Où les modèles sont enregistrés",
            "Dove vengono salvati i modelli",
            "Dónde se guardan los modelos",
            "Onde os modelos são guardados",
            "Wo die Modelle gespeichert werden",
            "Hová menti a program a modelleket",
        ],
    ),
    (
        "Next to this app",
        [
            "À côté de cette application",
            "Accanto a questa app",
            "Junto a esta aplicación",
            "Junto a esta aplicação",
            "Neben dieser App",
            "Az alkalmazás mellett",
        ],
    ),
    (
        "In my home folder",
        [
            "Dans mon dossier personnel",
            "Nella mia cartella personale",
            "En mi carpeta personal",
            "Na minha pasta pessoal",
            "In meinem Benutzerordner",
            "A saját mappámban",
        ],
    ),
    (
        "On another disk:",
        [
            "Sur un autre disque :",
            "Su un altro disco:",
            "En otro disco:",
            "Noutro disco:",
            "Auf einem anderen Datenträger:",
            "Másik lemezen:",
        ],
    ),
    (
        "or any folder:",
        [
            "ou n'importe quel dossier :",
            "oppure una cartella qualsiasi:",
            "o cualquier carpeta:",
            "ou qualquer pasta:",
            "oder ein beliebiger Ordner:",
            "vagy bármelyik mappa:",
        ],
    ),
    (
        "I can't write to «{}». Pick another folder or disk.",
        [
            "Je ne peux pas écrire dans «{}». Choisissez un autre dossier ou disque.",
            "Non riesco a scrivere in «{}». Scegli un'altra cartella o un altro disco.",
            "No puedo escribir en «{}». Elige otra carpeta u otro disco.",
            "Não consigo escrever em «{}». Escolha outra pasta ou outro disco.",
            "Ich kann nicht in «{}» schreiben. Wählen Sie einen anderen Ordner oder Datenträger.",
            "Nem tudok írni ide: «{}». Válassz másik mappát vagy lemezt.",
        ],
    ),
    (
        "Models you already downloaded stay where they are and still show in the list. If you pick “Next to this app”, a model of tens of GB uses the disk the app is installed on.",
        [
            "Les modèles déjà téléchargés restent où ils sont et apparaissent toujours dans la liste. Si vous choisissez « À côté de cette application », un modèle de dizaines de Go occupe le disque où l'application est installée.",
            "I modelli già scaricati restano dove sono e compaiono ancora nell'elenco. Se scegli «Accanto a questa app», un modello da decine di GB occupa il disco su cui è installata l'app.",
            "Los modelos ya descargados se quedan donde están y siguen apareciendo en la lista. Si eliges «Junto a esta aplicación», un modelo de decenas de GB ocupa el disco donde está instalada la aplicación.",
            "Os modelos já descarregados ficam onde estão e continuam na lista. Se escolher «Junto a esta aplicação», um modelo de dezenas de GB ocupa o disco onde a aplicação está instalada.",
            "Bereits heruntergeladene Modelle bleiben, wo sie sind, und erscheinen weiter in der Liste. Wählen Sie «Neben dieser App», belegt ein Modell mit Dutzenden GB den Datenträger, auf dem die App installiert ist.",
            "A már letöltött modellek ott maradnak, ahol vannak, és továbbra is látszanak a listában. Ha az «Az alkalmazás mellett» lehetőséget választod, a több tíz GB-os modell azt a lemezt foglalja, ahová az alkalmazás telepítve van.",
        ],
    ),
    (
        "Pick the first file (the one with “-00001-of-…” in its name) or a single .gguf file.",
        [
            "Choisissez le premier fichier (celui avec « -00001-of-… » dans son nom) ou un fichier .gguf unique.",
            "Scegli il primo file (quello con «-00001-of-…» nel nome) oppure un singolo file .gguf.",
            "Elige el primer archivo (el que lleva «-00001-of-…» en el nombre) o un único archivo .gguf.",
            "Escolha o primeiro ficheiro (o que tem «-00001-of-…» no nome) ou um único ficheiro .gguf.",
            "Wählen Sie die erste Datei (die mit «-00001-of-…» im Namen) oder eine einzelne .gguf-Datei.",
            "Válaszd az első fájlt (amelynek a nevében «-00001-of-…» szerepel) vagy egyetlen .gguf fájlt.",
        ],
    ),
    (
        "Missing from the same folder: {}",
        [
            "Absents du même dossier : {}",
            "Mancanti nella stessa cartella: {}",
            "Faltan en la misma carpeta: {}",
            "Em falta na mesma pasta: {}",
            "Fehlen im selben Ordner: {}",
            "Hiányzik ugyanabból a mappából: {}",
        ],
    ),
    (
        "That folder doesn't look like a colibri model (config.json is missing).",
        [
            "Ce dossier ne ressemble pas à un modèle colibri (config.json est introuvable).",
            "Questa cartella non sembra un modello colibri (manca config.json).",
            "Esa carpeta no parece un modelo colibri (falta config.json).",
            "Esta pasta não parece um modelo colibri (falta o config.json).",
            "Dieser Ordner sieht nicht wie ein colibri-Modell aus (config.json fehlt).",
            "Ez a mappa nem tűnik colibri modellnek (hiányzik a config.json).",
        ],
    ),
    (
        "Model file found. Press “Use this model” in the Models window.",
        [
            "Fichier de modèle trouvé. Appuyez sur « Utiliser ce modèle » dans la fenêtre Modèles.",
            "File del modello trovato. Premi «Usa questo modello» nella finestra Modelli.",
            "Archivo de modelo encontrado. Pulsa «Usar este modelo» en la ventana Modelos.",
            "Ficheiro de modelo encontrado. Prima «Usar este modelo» na janela Modelos.",
            "Modelldatei gefunden. Drücken Sie im Fenster Modelle auf «Dieses Modell verwenden».",
            "Modellfájl megtalálva. Nyomd meg a «Modell használata» gombot a Modellek ablakban.",
        ],
    ),
    (
        "I already have a model downloaded",
        [
            "J'ai déjà un modèle téléchargé",
            "Ho già un modello scaricato",
            "Ya tengo un modelo descargado",
            "Já tenho um modelo descarregado",
            "Ich habe bereits ein Modell heruntergeladen",
            "Már van letöltött modellem",
        ],
    ),
    (
        "If you have .gguf files (for example “…-00001-of-00008.gguf”), pick the first file; the other parts must be in the same folder. colibri converts them into a model folder beside them (this can take a while and needs extra space), then I start it. You can also pick a ready-made colibri model folder.",
        [
            "Si vous avez des fichiers .gguf (par exemple « …-00001-of-00008.gguf »), choisissez le premier ; les autres parties doivent être dans le même dossier. colibri les convertit en un dossier de modèle à côté (cela peut prendre du temps et demande de la place en plus), puis je le démarre. Vous pouvez aussi choisir un dossier de modèle colibri déjà prêt.",
            "Se hai file .gguf (ad esempio «…-00001-of-00008.gguf»), scegli il primo; le altre parti devono stare nella stessa cartella. colibri li converte in una cartella di modello accanto (può richiedere tempo e spazio extra), poi lo avvio. Puoi anche scegliere una cartella di modello colibri già pronta.",
            "Si tienes archivos .gguf (por ejemplo «…-00001-of-00008.gguf»), elige el primero; las demás partes deben estar en la misma carpeta. colibri los convierte en una carpeta de modelo a su lado (puede tardar y necesita espacio extra) y luego lo arranco. También puedes elegir una carpeta de modelo colibri ya preparada.",
            "Se tem ficheiros .gguf (por exemplo «…-00001-of-00008.gguf»), escolha o primeiro; as outras partes têm de estar na mesma pasta. O colibri converte-os numa pasta de modelo ao lado (pode demorar e precisa de espaço extra) e depois arranco-o. Também pode escolher uma pasta de modelo colibri já pronta.",
            "Wenn Sie .gguf-Dateien haben (zum Beispiel «…-00001-of-00008.gguf»), wählen Sie die erste; die übrigen Teile müssen im selben Ordner liegen. colibri wandelt sie in einen Modellordner daneben um (das kann dauern und braucht zusätzlichen Platz), danach starte ich es. Sie können auch einen fertigen colibri-Modellordner wählen.",
            "Ha vannak .gguf fájljaid (például «…-00001-of-00008.gguf»), válaszd ki az elsőt; a többi résznek ugyanabban a mappában kell lennie. A colibri mellé egy modellmappává alakítja őket (ez sokáig tarthat és plusz helyet igényel), aztán elindítom. Választhatsz kész colibri modellmappát is.",
        ],
    ),
    (
        "Path of the first .gguf file or of the folder",
        [
            "Chemin du premier fichier .gguf ou du dossier",
            "Percorso del primo file .gguf o della cartella",
            "Ruta del primer archivo .gguf o de la carpeta",
            "Caminho do primeiro ficheiro .gguf ou da pasta",
            "Pfad der ersten .gguf-Datei oder des Ordners",
            "Az első .gguf fájl vagy a mappa útvonala",
        ],
    ),
    (
        "You can also drag the file onto the app window.",
        [
            "Vous pouvez aussi glisser le fichier sur la fenêtre de l'application.",
            "Puoi anche trascinare il file sulla finestra dell'app.",
            "También puedes arrastrar el archivo a la ventana de la aplicación.",
            "Também pode arrastar o ficheiro para a janela da aplicação.",
            "Sie können die Datei auch auf das App-Fenster ziehen.",
            "A fájlt az alkalmazás ablakára is ráhúzhatod.",
        ],
    ),
    (
        "Local programs",
        [
            "Programmes locaux",
            "Programmi locali",
            "Programas locales",
            "Programas locais",
            "Lokale Programme",
            "Helyi programok",
        ],
    ),
    (
        "Compare",
        [
            "Comparer",
            "Confronta",
            "Comparar",
            "Comparar",
            "Vergleichen",
            "Összehasonlítás",
        ],
    ),
    (
        "Chat",
        [
            "Conversation",
            "Conversazione",
            "Conversación",
            "Conversa",
            "Unterhaltung",
            "Beszélgetés",
        ],
    ),
    (
        "Step-by-step thinking",
        [
            "Réflexion pas à pas",
            "Ragionamento passo passo",
            "Razonamiento paso a paso",
            "Raciocínio passo a passo",
            "Schrittweises Denken",
            "Lépésenkénti gondolkodás",
        ],
    ),
    (
        "Uses tools",
        [
            "Utilise des outils",
            "Usa strumenti",
            "Usa herramientas",
            "Usa ferramentas",
            "Nutzt Werkzeuge",
            "Eszközöket használ",
        ],
    ),
    (
        "Sees pictures",
        [
            "Voit les images",
            "Vede le immagini",
            "Ve imágenes",
            "Vê imagens",
            "Sieht Bilder",
            "Lát képeket",
        ],
    ),
    (
        "Specialised in code",
        [
            "Spécialisé en code",
            "Specializzato nel codice",
            "Especializado en código",
            "Especializado em código",
            "Auf Code spezialisiert",
            "Kódra specializált",
        ],
    ),
    (
        "Makes pictures",
        [
            "Crée des images",
            "Crea immagini",
            "Crea imágenes",
            "Cria imagens",
            "Erzeugt Bilder",
            "Képeket készít",
        ],
    ),
    (
        "Decisions (labels, probabilities)",
        [
            "Décisions (étiquettes, probabilités)",
            "Decisioni (etichette, probabilità)",
            "Decisiones (etiquetas, probabilidades)",
            "Decisões (etiquetas, probabilidades)",
            "Entscheidungen (Labels, Wahrscheinlichkeiten)",
            "Döntések (címkék, valószínűségek)",
        ],
    ),
    (
        "Detailed sheet",
        [
            "Fiche détaillée",
            "Scheda dettagliata",
            "Ficha detallada",
            "Ficha detalhada",
            "Detailblatt",
            "Részletes adatlap",
        ],
    ),
    (
        "What it can do",
        [
            "Ce qu'il sait faire",
            "Cosa sa fare",
            "Qué sabe hacer",
            "O que sabe fazer",
            "Was es kann",
            "Mire képes",
        ],
    ),
    (
        "(colibri doesn't say)",
        [
            "(colibri ne le dit pas)",
            "(colibri non lo dice)",
            "(colibri no lo dice)",
            "(o colibri não diz)",
            "(colibri sagt es nicht)",
            "(a colibri nem mondja)",
        ],
    ),
    (
        "Context (tokens):",
        [
            "Contexte (jetons) :",
            "Contesto (token):",
            "Contexto (tokens):",
            "Contexto (tokens):",
            "Kontext (Tokens):",
            "Kontextus (tokenek):",
        ],
    ),
    (
        "Files it can take in this app",
        [
            "Fichiers qu'il peut recevoir dans cette application",
            "File che può ricevere in questa app",
            "Archivos que puede recibir en esta aplicación",
            "Ficheiros que pode receber nesta aplicação",
            "Dateien, die es in dieser App annehmen kann",
            "Fájlok, amelyeket ebben az alkalmazásban fogadhat",
        ],
    ),
    (
        "Text, code, CSV, JSON, HTML, PDF, Word, Excel, PowerPoint and OpenDocument: I read them and put the text in your message.",
        [
            "Texte, code, CSV, JSON, HTML, PDF, Word, Excel, PowerPoint et OpenDocument : je les lis et place le texte dans votre message.",
            "Testo, codice, CSV, JSON, HTML, PDF, Word, Excel, PowerPoint e OpenDocument: li leggo e metto il testo nel messaggio.",
            "Texto, código, CSV, JSON, HTML, PDF, Word, Excel, PowerPoint y OpenDocument: los leo y pongo el texto en tu mensaje.",
            "Texto, código, CSV, JSON, HTML, PDF, Word, Excel, PowerPoint e OpenDocument: eu leio-os e coloco o texto na sua mensagem.",
            "Text, Code, CSV, JSON, HTML, PDF, Word, Excel, PowerPoint und OpenDocument: Ich lese sie und setze den Text in Ihre Nachricht.",
            "Szöveg, kód, CSV, JSON, HTML, PDF, Word, Excel, PowerPoint és OpenDocument: én elolvasom, és a szöveget az üzenetedbe teszem.",
        ],
    ),
    (
        "Pictures (PNG, JPG, GIF, WebP, BMP): the model can look at them.",
        [
            "Images (PNG, JPG, GIF, WebP, BMP) : le modèle peut les regarder.",
            "Immagini (PNG, JPG, GIF, WebP, BMP): il modello può guardarle.",
            "Imágenes (PNG, JPG, GIF, WebP, BMP): el modelo puede verlas.",
            "Imagens (PNG, JPG, GIF, WebP, BMP): o modelo pode vê-las.",
            "Bilder (PNG, JPG, GIF, WebP, BMP): das Modell kann sie ansehen.",
            "Képek (PNG, JPG, GIF, WebP, BMP): a modell meg tudja nézni őket.",
        ],
    ),
    (
        "Pictures: no, or colibri doesn't say it can look at them.",
        [
            "Images : non, ou colibri ne dit pas qu'il peut les regarder.",
            "Immagini: no, oppure colibri non dice che può guardarle.",
            "Imágenes: no, o colibri no dice que pueda verlas.",
            "Imagens: não, ou o colibri não diz que as pode ver.",
            "Bilder: nein, oder colibri sagt nicht, dass es sie ansehen kann.",
            "Képek: nem, vagy a colibri nem mondja, hogy meg tudja nézni őket.",
        ],
    ),
    (
        "What it can produce",
        [
            "Ce qu'il peut produire",
            "Cosa può produrre",
            "Qué puede producir",
            "O que pode produzir",
            "Was es erzeugen kann",
            "Mit tud létrehozni",
        ],
    ),
    (
        "Pictures from text (through colibri, not in the chat here).",
        [
            "Images à partir de texte (via colibri, pas dans la conversation ici).",
            "Immagini da testo (tramite colibri, non nella chat qui).",
            "Imágenes a partir de texto (mediante colibri, no en el chat de aquí).",
            "Imagens a partir de texto (através do colibri, não na conversa aqui).",
            "Bilder aus Text (über colibri, nicht im Chat hier).",
            "Képek szövegből (a colibrin keresztül, nem az itteni csevegésben).",
        ],
    ),
    (
        "Text. From an answer you can save any code or table block as a file: .py, .rs, .html, .csv, .json, .md, .svg and more.",
        [
            "Du texte. D'une réponse, vous pouvez enregistrer tout bloc de code ou de tableau comme fichier : .py, .rs, .html, .csv, .json, .md, .svg et plus.",
            "Testo. Da una risposta puoi salvare come file qualsiasi blocco di codice o tabella: .py, .rs, .html, .csv, .json, .md, .svg e altri.",
            "Texto. De una respuesta puedes guardar como archivo cualquier bloque de código o tabla: .py, .rs, .html, .csv, .json, .md, .svg y más.",
            "Texto. De uma resposta pode guardar como ficheiro qualquer bloco de código ou tabela: .py, .rs, .html, .csv, .json, .md, .svg e mais.",
            "Text. Aus einer Antwort können Sie jeden Code- oder Tabellenblock als Datei speichern: .py, .rs, .html, .csv, .json, .md, .svg und mehr.",
            "Szöveg. A válaszból bármely kód- vagy táblázatblokkot fájlként mentheted: .py, .rs, .html, .csv, .json, .md, .svg és más.",
        ],
    ),
    (
        "Graphics / processor:",
        [
            "Carte graphique / processeur :",
            "Scheda video / processore:",
            "Tarjeta gráfica / procesador:",
            "Placa gráfica / processador:",
            "Grafikkarte / Prozessor:",
            "Videokártya / processzor:",
        ],
    ),
    (
        "Compare with online AIs",
        [
            "Comparer avec les IA en ligne",
            "Confronta con le IA online",
            "Comparar con las IA en línea",
            "Comparar com as IA online",
            "Mit Online-KIs vergleichen",
            "Összehasonlítás az online MI-kkel",
        ],
    ),
    (
        "Comparison: local models and online AIs",
        [
            "Comparaison : modèles locaux et IA en ligne",
            "Confronto: modelli locali e IA online",
            "Comparación: modelos locales e IA en línea",
            "Comparação: modelos locais e IA online",
            "Vergleich: lokale Modelle und Online-KIs",
            "Összehasonlítás: helyi modellek és online MI-k",
        ],
    ),
    (
        "Local models: the data comes from the colibri page; “?” means the page doesn't say. Online AIs: only features their own sites advertise, no scores or tests; check the site, they change often. The sure difference: local models run on your computer, with no account and without sending your conversation to anyone else.",
        [
            "Modèles locaux : les données viennent de la page de colibri ; « ? » signifie que la page ne le dit pas. IA en ligne : seulement les fonctions que leurs sites annoncent, sans scores ni tests ; vérifiez sur le site, elles changent souvent. La différence sûre : les modèles locaux tournent sur votre ordinateur, sans compte et sans envoyer votre conversation à quelqu'un d'autre.",
            "Modelli locali: i dati vengono dalla pagina di colibri; «?» significa che la pagina non lo dice. IA online: solo funzioni che i loro siti pubblicizzano, senza punteggi né test; controlla il sito, cambiano spesso. La differenza certa: i modelli locali girano sul tuo computer, senza account e senza inviare la conversazione ad altri.",
            "Modelos locales: los datos vienen de la página de colibri; «?» significa que la página no lo dice. IA en línea: solo funciones que anuncian sus propios sitios, sin puntuaciones ni pruebas; revisa el sitio, cambian a menudo. La diferencia segura: los modelos locales se ejecutan en tu ordenador, sin cuenta y sin enviar tu conversación a nadie más.",
            "Modelos locais: os dados vêm da página do colibri; «?» significa que a página não diz. IA online: apenas funções que os próprios sites anunciam, sem pontuações nem testes; verifique o site, mudam muitas vezes. A diferença segura: os modelos locais correm no seu computador, sem conta e sem enviar a sua conversa a mais ninguém.",
            "Lokale Modelle: Die Daten stammen von der colibri-Seite; „?“ heißt, die Seite sagt es nicht. Online-KIs: nur Funktionen, die ihre eigenen Seiten bewerben, ohne Punkte oder Tests; prüfen Sie die Seite, sie ändern sich oft. Der sichere Unterschied: Lokale Modelle laufen auf Ihrem Computer, ohne Konto und ohne Ihre Unterhaltung an andere zu senden.",
            "Helyi modellek: az adatok a colibri oldaláról származnak; a „?” azt jelenti, hogy az oldal nem mondja. Online MI-k: csak olyan funkciók, amelyeket saját oldaluk hirdet, pontszámok és tesztek nélkül; ellenőrizd az oldalon, gyakran változnak. A biztos különbség: a helyi modellek a te gépeden futnak, fiók nélkül, és a beszélgetésedet nem küldik el senkinek.",
        ],
    ),
    (
        "Local models (colibri)",
        [
            "Modèles locaux (colibri)",
            "Modelli locali (colibri)",
            "Modelos locales (colibri)",
            "Modelos locais (colibri)",
            "Lokale Modelle (colibri)",
            "Helyi modellek (colibri)",
        ],
    ),
    (
        "Download",
        [
            "Téléchargement",
            "Download",
            "Descarga",
            "Transferência",
            "Download",
            "Letöltés",
        ],
    ),
    (
        "Online AIs",
        [
            "IA en ligne",
            "IA online",
            "IA en línea",
            "IA online",
            "Online-KIs",
            "Online MI-k",
        ],
    ),
    (
        "Files in",
        [
            "Fichiers reçus",
            "File in ingresso",
            "Archivos admitidos",
            "Ficheiros aceites",
            "Dateien rein",
            "Fájlok be",
        ],
    ),
    (
        "Pictures in",
        [
            "Images reçues",
            "Immagini in ingresso",
            "Imágenes admitidas",
            "Imagens aceites",
            "Bilder rein",
            "Képek be",
        ],
    ),
    (
        "Web search",
        [
            "Recherche web",
            "Ricerca web",
            "Búsqueda web",
            "Pesquisa na web",
            "Websuche",
            "Webes keresés",
        ],
    ),
    (
        "Runs on your PC",
        [
            "Tourne chez vous",
            "Gira sul tuo PC",
            "Se ejecuta en tu PC",
            "Corre no seu PC",
            "Läuft auf Ihrem PC",
            "A te gépeden fut",
        ],
    ),
    (
        "For colibri: files and pictures depend on the model you pick (see its sheet); pictures are made only with Qwen-Image-2.1; this app does not search the web.",
        [
            "Pour colibri : les fichiers et images dépendent du modèle choisi (voir sa fiche) ; les images ne sont créées qu'avec Qwen-Image-2.1 ; cette application ne cherche pas sur le web.",
            "Per colibri: file e immagini dipendono dal modello scelto (vedi la sua scheda); le immagini si creano solo con Qwen-Image-2.1; questa app non cerca sul web.",
            "Para colibri: los archivos y las imágenes dependen del modelo elegido (mira su ficha); las imágenes solo se crean con Qwen-Image-2.1; esta aplicación no busca en la web.",
            "Para o colibri: os ficheiros e as imagens dependem do modelo escolhido (veja a sua ficha); as imagens só se criam com o Qwen-Image-2.1; esta aplicação não pesquisa na web.",
            "Für colibri: Dateien und Bilder hängen vom gewählten Modell ab (siehe sein Detailblatt); Bilder entstehen nur mit Qwen-Image-2.1; diese App sucht nicht im Web.",
            "A colibrinél: a fájlok és képek a választott modelltől függenek (lásd az adatlapját); képet csak a Qwen-Image-2.1 készít; ez az alkalmazás nem keres a weben.",
        ],
    ),
    (
        "Any program on this computer that speaks “OpenAI” or “Anthropic” can use colibri through the same local address. Nothing leaves the computer. Copy the settings below into that program.",
        [
            "Tout programme de cet ordinateur qui parle « OpenAI » ou « Anthropic » peut utiliser colibri via la même adresse locale. Rien ne quitte l'ordinateur. Copiez les réglages ci-dessous dans ce programme.",
            "Qualsiasi programma di questo computer che parla «OpenAI» o «Anthropic» può usare colibri tramite lo stesso indirizzo locale. Nulla esce dal computer. Copia le impostazioni qui sotto in quel programma.",
            "Cualquier programa de este ordenador que hable «OpenAI» o «Anthropic» puede usar colibri a través de la misma dirección local. Nada sale del ordenador. Copia los ajustes de abajo en ese programa.",
            "Qualquer programa deste computador que fale «OpenAI» ou «Anthropic» pode usar o colibri através do mesmo endereço local. Nada sai do computador. Copie as definições abaixo para esse programa.",
            "Jedes Programm auf diesem Computer, das «OpenAI» oder «Anthropic» spricht, kann colibri über dieselbe lokale Adresse nutzen. Nichts verlässt den Computer. Kopieren Sie die Einstellungen unten in dieses Programm.",
            "A gépen lévő bármely program, amely «OpenAI» vagy «Anthropic» protokollt beszél, használhatja a colibrit ugyanazon a helyi címen. Semmi nem hagyja el a számítógépet. Másold a lenti beállításokat abba a programba.",
        ],
    ),
    (
        "Programs with an “OpenAI provider” setting (code editors, agents, interfaces)",
        [
            "Programmes avec un réglage « fournisseur OpenAI » (éditeurs de code, agents, interfaces)",
            "Programmi con un'impostazione «provider OpenAI» (editor di codice, agenti, interfacce)",
            "Programas con un ajuste de «proveedor OpenAI» (editores de código, agentes, interfaces)",
            "Programas com uma definição de «fornecedor OpenAI» (editores de código, agentes, interfaces)",
            "Programme mit einer Einstellung «OpenAI-Anbieter» (Code-Editoren, Agenten, Oberflächen)",
            "Programok «OpenAI-szolgáltató» beállítással (kódszerkesztők, ügynökök, felületek)",
        ],
    ),
    (
        "Enter the base URL, the model name and any non-empty key (it isn't checked).",
        [
            "Saisissez l'adresse de base, le nom du modèle et une clé quelconque non vide (elle n'est pas vérifiée).",
            "Inserisci l'indirizzo base, il nome del modello e una chiave qualsiasi non vuota (non viene controllata).",
            "Introduce la dirección base, el nombre del modelo y cualquier clave no vacía (no se comprueba).",
            "Introduza o endereço base, o nome do modelo e uma chave qualquer não vazia (não é verificada).",
            "Geben Sie die Basis-URL, den Modellnamen und einen beliebigen nicht leeren Schlüssel ein (er wird nicht geprüft).",
            "Add meg az alapcímet, a modell nevét és egy tetszőleges nem üres kulcsot (nem ellenőrzi).",
        ],
    ),
    (
        "colibri also answers on the Anthropic-compatible address (/v1/messages).",
        [
            "colibri répond aussi à l'adresse compatible Anthropic (/v1/messages).",
            "colibri risponde anche all'indirizzo compatibile Anthropic (/v1/messages).",
            "colibri también responde en la dirección compatible con Anthropic (/v1/messages).",
            "O colibri também responde no endereço compatível com a Anthropic (/v1/messages).",
            "colibri antwortet auch unter der Anthropic-kompatiblen Adresse (/v1/messages).",
            "A colibri az Anthropic-kompatibilis címen (/v1/messages) is válaszol.",
        ],
    ),
    (
        "Programs that use MCP (for example Claude Desktop or Claude Code) can ask colibri to recommend, install, start or stop models: the command is “coli mcp”.",
        [
            "Les programmes qui utilisent MCP (par exemple Claude Desktop ou Claude Code) peuvent demander à colibri de recommander, installer, démarrer ou arrêter des modèles : la commande est « coli mcp ».",
            "I programmi che usano MCP (ad esempio Claude Desktop o Claude Code) possono chiedere a colibri di consigliare, installare, avviare o fermare modelli: il comando è «coli mcp».",
            "Los programas que usan MCP (por ejemplo Claude Desktop o Claude Code) pueden pedir a colibri que recomiende, instale, inicie o detenga modelos: el comando es «coli mcp».",
            "Os programas que usam MCP (por exemplo Claude Desktop ou Claude Code) podem pedir ao colibri que recomende, instale, inicie ou pare modelos: o comando é «coli mcp».",
            "Programme, die MCP nutzen (zum Beispiel Claude Desktop oder Claude Code), können colibri bitten, Modelle zu empfehlen, zu installieren, zu starten oder zu stoppen: Der Befehl lautet «coli mcp».",
            "Az MCP-t használó programok (például a Claude Desktop vagy a Claude Code) megkérhetik a colibrit, hogy ajánljon, telepítsen, indítson vagy állítson le modelleket: a parancs «coli mcp».",
        ],
    ),
    (
        "Open the colibri dashboard in the browser",
        [
            "Ouvrir le tableau de bord colibri dans le navigateur",
            "Apri la dashboard di colibri nel browser",
            "Abrir el panel de colibri en el navegador",
            "Abrir o painel do colibri no navegador",
            "colibri-Dashboard im Browser öffnen",
            "A colibri irányítópultjának megnyitása a böngészőben",
        ],
    ),
    (
        "Open the Downloads folder",
        [
            "Ouvrir le dossier Téléchargements",
            "Apri la cartella Download",
            "Abrir la carpeta Descargas",
            "Abrir a pasta Transferências",
            "Ordner Downloads öffnen",
            "A Letöltések mappa megnyitása",
        ],
    ),
    (
        "Programs work only while a model is running here. This app can't promise every third-party program works: colibri officially tested the OpenAI and Anthropic SDKs; the others use the same protocol.",
        [
            "Les programmes ne fonctionnent que tant qu'un modèle tourne ici. Cette application ne peut pas promettre que chaque programme tiers fonctionne : colibri a testé officiellement les SDK OpenAI et Anthropic ; les autres utilisent le même protocole.",
            "I programmi funzionano solo finché qui gira un modello. Questa app non può promettere che ogni programma di terzi funzioni: colibri ha testato ufficialmente gli SDK OpenAI e Anthropic; gli altri usano lo stesso protocollo.",
            "Los programas solo funcionan mientras un modelo se ejecuta aquí. Esta aplicación no puede prometer que cada programa de terceros funcione: colibri probó oficialmente los SDK de OpenAI y Anthropic; los demás usan el mismo protocolo.",
            "Os programas só funcionam enquanto um modelo corre aqui. Esta aplicação não pode prometer que cada programa de terceiros funcione: o colibri testou oficialmente os SDK da OpenAI e da Anthropic; os outros usam o mesmo protocolo.",
            "Programme funktionieren nur, solange hier ein Modell läuft. Diese App kann nicht versprechen, dass jedes Fremdprogramm funktioniert: colibri hat die OpenAI- und Anthropic-SDKs offiziell getestet; die anderen nutzen dasselbe Protokoll.",
            "A programok csak addig működnek, amíg itt fut egy modell. Az alkalmazás nem ígérheti, hogy minden külső program működik: a colibri hivatalosan az OpenAI és az Anthropic SDK-kat tesztelte; a többi ugyanazt a protokollt használja.",
        ],
    ),
    (
        "More ▾",
        [
            "Plus ▾",
            "Altro ▾",
            "Más ▾",
            "Mais ▾",
            "Mehr ▾",
            "Továbbiak ▾",
        ],
    ),
    (
        "Compare models",
        [
            "Comparer les modèles",
            "Confronta i modelli",
            "Comparar modelos",
            "Comparar modelos",
            "Modelle vergleichen",
            "Modellek összehasonlítása",
        ],
    ),
    (
        "What the model's maker says",
        [
            "Ce que dit le créateur du modèle",
            "Cosa dice il creatore del modello",
            "Lo que dice el creador del modelo",
            "O que diz o criador do modelo",
            "Was der Hersteller des Modells sagt",
            "Amit a modell készítője állít",
        ],
    ),
    (
        "text",
        ["texte", "testo", "texto", "texto", "Text", "szöveg"],
    ),
    (
        "pictures",
        [
            "images",
            "immagini",
            "imágenes",
            "imagens",
            "Bilder",
            "képek",
        ],
    ),
    (
        "video",
        ["vidéo", "video", "vídeo", "vídeo", "Video", "videó"],
    ),
    (
        "audio",
        ["audio", "audio", "audio", "áudio", "Audio", "hang"],
    ),
    (
        "Takes in:",
        [
            "Accepte :",
            "Accetta:",
            "Acepta:",
            "Aceita:",
            "Nimmt auf:",
            "Fogad:",
        ],
    ),
    (
        "Maximum context (tokens):",
        [
            "Contexte maximal (jetons) :",
            "Contesto massimo (token):",
            "Contexto máximo (tokens):",
            "Contexto máximo (tokens):",
            "Maximaler Kontext (Token):",
            "Maximális kontextus (token):",
        ],
    ),
    (
        "Parameters active per token:",
        [
            "Paramètres actifs par jeton :",
            "Parametri attivi per token:",
            "Parámetros activos por token:",
            "Parâmetros ativos por token:",
            "Aktive Parameter pro Token:",
            "Aktív paraméterek tokenenként:",
        ],
    ),
    (
        "Licence (maker):",
        [
            "Licence (créateur) :",
            "Licenza (creatore):",
            "Licencia (creador):",
            "Licença (criador):",
            "Lizenz (Hersteller):",
            "Licenc (készítő):",
        ],
    ),
    (
        "Note: the model may be able to do more than colibri uses today. What counts here is the ticks above (from the colibri page); the maker's claims show only the potential. Data looked up online in November 2026; check the model's own page.",
        [
            "Remarque : le modèle peut faire plus que ce que colibri utilise aujourd'hui. Ce qui compte ici, ce sont les coches ci-dessus (issues de la page colibri) ; les affirmations du créateur n'indiquent que le potentiel. Données cherchées en ligne en novembre 2026 ; vérifiez la page du modèle.",
            "Nota: il modello potrebbe fare più di quanto colibri usa oggi. Qui contano le spunte sopra (dalla pagina di colibri); le affermazioni del creatore mostrano solo il potenziale. Dati cercati online a novembre 2026; controlla la pagina del modello.",
            "Nota: el modelo puede hacer más de lo que colibri usa hoy. Aquí cuentan las marcas de arriba (de la página de colibri); lo que dice el creador solo muestra el potencial. Datos buscados en línea en noviembre de 2026; revisa la página del modelo.",
            "Nota: o modelo pode fazer mais do que o colibri usa hoje. Aqui contam as marcas acima (da página do colibri); o que diz o criador mostra apenas o potencial. Dados pesquisados online em novembro de 2026; confira a página do modelo.",
            "Hinweis: Das Modell kann womöglich mehr, als colibri heute nutzt. Hier zählen die Häkchen oben (von der colibri-Seite); die Angaben des Herstellers zeigen nur das Potenzial. Daten im November 2026 online recherchiert; prüfe die Seite des Modells.",
            "Megjegyzés: a modell többre lehet képes, mint amit a colibri ma használ. Itt a fenti pipák számítanak (a colibri oldaláról); a készítő állításai csak a lehetőséget mutatják. Az adatok 2026 novemberében online keresve; ellenőrizd a modell oldalát.",
        ],
    ),
    (
        "The files are big: sending them in {} parts, one after another.",
        [
            "Les fichiers sont gros : je les envoie en {} parties, l'une après l'autre.",
            "I file sono grandi: li invio in {} parti, una dopo l'altra.",
            "Los archivos son grandes: los envío en {} partes, una tras otra.",
            "Os arquivos são grandes: envio em {} partes, uma após a outra.",
            "Die Dateien sind groß: Ich sende sie in {} Teilen nacheinander.",
            "A fájlok nagyok: {} részletben küldöm el őket, egymás után.",
        ],
    ),
    (
        "No files to read were found",
        [
            "Aucun fichier à lire trouvé",
            "Nessun file da leggere trovato",
            "No encontré archivos que leer",
            "Não encontrei arquivos para ler",
            "Keine lesbaren Dateien gefunden",
            "Nem találtam olvasható fájlt",
        ],
    ),
    (
        "Reading {} files",
        [
            "Lecture de {} fichiers",
            "Leggo {} file",
            "Leyendo {} archivos",
            "Lendo {} arquivos",
            "Lese {} Dateien",
            "{} fájl olvasása",
        ],
    ),
    (
        " (only the first ones; there are more)",
        [
            " (seulement les premiers ; il y en a plus)",
            " (solo i primi; ce ne sono altri)",
            " (solo los primeros; hay más)",
            " (apenas os primeiros; há mais)",
            " (nur die ersten; es gibt mehr)",
            " (csak az első néhány; több van)",
        ],
    ),
    (
        "Remove all",
        [
            "Tout retirer",
            "Rimuovi tutto",
            "Quitar todo",
            "Remover tudo",
            "Alle entfernen",
            "Összes eltávolítása",
        ],
    ),
    (
        "files,",
        [
            "fichiers,",
            "file,",
            "archivos,",
            "arquivos,",
            "Dateien,",
            "fájl,",
        ],
    ),
    (
        "thousand characters",
        [
            "mille caractères",
            "mila caratteri",
            "mil caracteres",
            "mil caracteres",
            "tausend Zeichen",
            "ezer karakter",
        ],
    ),
    (
        "Prompts",
        [
            "Prompts", "Prompt", "Prompts", "Prompts", "Prompts", "Promptok",
        ],
    ),
    (
        "Normal",
        ["Normal", "Normale", "Normal", "Normal", "Normal", "Normál"],
    ),
    (
        "Exact",
        ["Exact", "Esatto", "Exacto", "Exato", "Exakt", "Pontos"],
    ),
    (
        "Didactic",
        [
            "Didactique",
            "Didattico",
            "Didáctico",
            "Didático",
            "Lehrreich",
            "Oktató",
        ],
    ),
    (
        "Socratic",
        [
            "Socratique",
            "Socratico",
            "Socrático",
            "Socrático",
            "Sokratisch",
            "Szókratészi",
        ],
    ),
    (
        "Contrarian",
        [
            "Contradicteur",
            "Bastian contrario",
            "Contrario",
            "Contrário",
            "Widerspruch",
            "Ellenvélemény",
        ],
    ),
    (
        "Brainstorm",
        [
            "Remue-méninges",
            "Brainstorming",
            "Lluvia de ideas",
            "Tempestade de ideias",
            "Ideensturm",
            "Ötletroham",
        ],
    ),
    (
        "Telegraphic",
        [
            "Télégraphique",
            "Telegrafico",
            "Telegráfico",
            "Telegráfico",
            "Telegrammstil",
            "Táviratstílus",
        ],
    ),
    (
        "Coach",
        ["Coach", "Coach", "Coach", "Coach", "Coach", "Coach"],
    ),
    (
        "Could not save the prompts",
        [
            "Impossible d'enregistrer les prompts",
            "Impossibile salvare i prompt",
            "No pude guardar los prompts",
            "Não consegui salvar os prompts",
            "Prompts konnten nicht gespeichert werden",
            "Nem sikerült menteni a promptokat",
        ],
    ),
    (
        "Answer style:",
        [
            "Style de réponse :",
            "Stile di risposta:",
            "Estilo de respuesta:",
            "Estilo de resposta:",
            "Antwortstil:",
            "Válasz stílusa:",
        ],
    ),
    (
        "Prompts (Ctrl+P)",
        [
            "Prompts (Ctrl+P)",
            "Prompt (Ctrl+P)",
            "Prompts (Ctrl+P)",
            "Prompts (Ctrl+P)",
            "Prompts (Strg+P)",
            "Promptok (Ctrl+P)",
        ],
    ),
    (
        "Search…",
        [
            "Rechercher…",
            "Cerca…",
            "Buscar…",
            "Pesquisar…",
            "Suchen…",
            "Keresés…",
        ],
    ),
    (
        "★ Favourites",
        [
            "★ Favoris",
            "★ Preferiti",
            "★ Favoritos",
            "★ Favoritos",
            "★ Favoriten",
            "★ Kedvencek",
        ],
    ),
    (
        "Mine",
        ["Les miens", "Miei", "Míos", "Meus", "Meine", "Sajátjaim"],
    ),
    (
        "Path of a JSON file to import",
        [
            "Chemin d'un fichier JSON à importer",
            "Percorso di un file JSON da importare",
            "Ruta de un archivo JSON para importar",
            "Caminho de um arquivo JSON para importar",
            "Pfad einer zu importierenden JSON-Datei",
            "Importálandó JSON-fájl útvonala",
        ],
    ),
    (
        "Prompt library",
        [
            "Bibliothèque de prompts",
            "Libreria di prompt",
            "Biblioteca de prompts",
            "Biblioteca de prompts",
            "Prompt-Bibliothek",
            "Prompt-könyvtár",
        ],
    ),
    (
        "+ New prompt",
        [
            "+ Nouveau prompt",
            "+ Nuovo prompt",
            "+ Nuevo prompt",
            "+ Novo prompt",
            "+ Neuer Prompt",
            "+ Új prompt",
        ],
    ),
    ("All", ["Tous", "Tutti", "Todos", "Todos", "Alle", "Mind"]),
    (
        "No prompt matches. Try another word or category.",
        [
            "Aucun prompt ne correspond. Essayez un autre mot ou une autre catégorie.",
            "Nessun prompt corrisponde. Prova un'altra parola o categoria.",
            "Ningún prompt coincide. Prueba otra palabra o categoría.",
            "Nenhum prompt corresponde. Tente outra palavra ou categoria.",
            "Kein Prompt passt. Versuche ein anderes Wort oder eine andere Kategorie.",
            "Egy prompt sem egyezik. Próbálj másik szót vagy kategóriát.",
        ],
    ),
    (
        "Pick a prompt from the list. Fill in the blanks, then put it in your message.",
        [
            "Choisissez un prompt dans la liste. Remplissez les blancs, puis mettez-le dans votre message.",
            "Scegli un prompt dalla lista. Compila gli spazi vuoti, poi mettilo nel messaggio.",
            "Elige un prompt de la lista. Rellena los huecos y luego ponlo en tu mensaje.",
            "Escolha um prompt da lista. Preencha os espaços e depois coloque-o na mensagem.",
            "Wähle einen Prompt aus der Liste. Fülle die Lücken aus und setze ihn dann in deine Nachricht.",
            "Válassz egy promptot a listából. Töltsd ki a hézagokat, majd tedd az üzenetbe.",
        ],
    ),
    (
        "Put in message",
        [
            "Mettre dans le message",
            "Metti nel messaggio",
            "Poner en el mensaje",
            "Pôr na mensagem",
            "In Nachricht einfügen",
            "Üzenetbe teszem",
        ],
    ),
    (
        "Export my prompts (JSON)",
        [
            "Exporter mes prompts (JSON)",
            "Esporta i miei prompt (JSON)",
            "Exportar mis prompts (JSON)",
            "Exportar meus prompts (JSON)",
            "Meine Prompts exportieren (JSON)",
            "Promptjaim exportálása (JSON)",
        ],
    ),
    (
        "Saved to",
        [
            "Enregistré dans",
            "Salvato in",
            "Guardado en",
            "Salvo em",
            "Gespeichert in",
            "Elmentve ide:",
        ],
    ),
    (
        "Import",
        [
            "Importer",
            "Importa",
            "Importar",
            "Importar",
            "Importieren",
            "Importálás",
        ],
    ),
    (
        "New prompts:",
        [
            "Nouveaux prompts :",
            "Nuovi prompt:",
            "Prompts nuevos:",
            "Novos prompts:",
            "Neue Prompts:",
            "Új promptok:",
        ],
    ),
    (
        "Not a prompt file",
        [
            "Ce n'est pas un fichier de prompts",
            "Non è un file di prompt",
            "No es un archivo de prompts",
            "Não é um arquivo de prompts",
            "Keine Prompt-Datei",
            "Nem prompt-fájl",
        ],
    ),
    (
        "My own prompt",
        [
            "Mon propre prompt",
            "Il mio prompt",
            "Mi propio prompt",
            "Meu próprio prompt",
            "Mein eigener Prompt",
            "Saját promptom",
        ],
    ),
    (
        "Title",
        ["Titre", "Titolo", "Título", "Título", "Titel", "Cím"],
    ),
    (
        "Category",
        [
            "Catégorie",
            "Categoria",
            "Categoría",
            "Categoria",
            "Kategorie",
            "Kategória",
        ],
    ),
    (
        "Text (use [WORDS] for the blanks)",
        [
            "Texte (mettez [MOTS] pour les blancs)",
            "Testo (usa [PAROLE] per gli spazi vuoti)",
            "Texto (usa [PALABRAS] para los huecos)",
            "Texto (use [PALAVRAS] para os espaços)",
            "Text (nutze [WÖRTER] für die Lücken)",
            "Szöveg ([SZAVAK] jelöli a kitöltendő helyeket)",
        ],
    ),
    (
        "It needs a title and a text.",
        [
            "Il faut un titre et un texte.",
            "Servono un titolo e un testo.",
            "Hace falta un título y un texto.",
            "É preciso um título e um texto.",
            "Es braucht einen Titel und einen Text.",
            "Cím és szöveg is kell.",
        ],
    ),
    (
        "Save",
        [
            "Enregistrer",
            "Salva",
            "Guardar",
            "Salvar",
            "Speichern",
            "Mentés",
        ],
    ),
    (
        "Online connections",
        [
            "Connexions en ligne",
            "Connessioni online",
            "Conexiones en línea",
            "Conexões online",
            "Online-Verbindungen",
            "Online kapcsolatok",
        ],
    ),
    (
        "Platform catalogue",
        [
            "Catalogue de plateformes",
            "Catalogo di piattaforme",
            "Catálogo de plataformas",
            "Catálogo de plataformas",
            "Plattformkatalog",
            "Platformkatalógus",
        ],
    ),
    (
        "Documents and notes",
        [
            "Documents et notes",
            "Documenti e note",
            "Documentos y notas",
            "Documentos e notas",
            "Dokumente und Notizen",
            "Dokumentumok és jegyzetek",
        ],
    ),
    (
        "Projects and tasks",
        [
            "Projets et tâches",
            "Progetti e attività",
            "Proyectos y tareas",
            "Projetos e tarefas",
            "Projekte und Aufgaben",
            "Projektek és feladatok",
        ],
    ),
    (
        "Messages and alerts",
        [
            "Messages et alertes",
            "Messaggi e avvisi",
            "Mensajes y alertas",
            "Mensagens e alertas",
            "Nachrichten und Benachrichtigungen",
            "Üzenetek és értesítések",
        ],
    ),
    (
        "Mail and calendar",
        [
            "Courrier et agenda",
            "Posta e calendario",
            "Correo y calendario",
            "Correio e calendário",
            "E-Mail und Kalender",
            "Levelezés és naptár",
        ],
    ),
    (
        "Web and search",
        [
            "Web et recherche",
            "Web e ricerca",
            "Web y búsqueda",
            "Web e pesquisa",
            "Web und Suche",
            "Web és keresés",
        ],
    ),
    (
        "Data and files",
        [
            "Données et fichiers",
            "Dati e file",
            "Datos y archivos",
            "Dados e arquivos",
            "Daten und Dateien",
            "Adatok és fájlok",
        ],
    ),
    (
        "Automation and smart home",
        [
            "Automatisation et maison connectée",
            "Automazione e casa intelligente",
            "Automatización y hogar inteligente",
            "Automação e casa inteligente",
            "Automatisierung und Smart Home",
            "Automatizálás és okosotthon",
        ],
    ),
    (
        "Knowledge and science",
        [
            "Connaissances et science",
            "Conoscenza e scienza",
            "Conocimiento y ciencia",
            "Conhecimento e ciência",
            "Wissen und Wissenschaft",
            "Tudás és tudomány",
        ],
    ),
    (
        "Operations and tools",
        [
            "Opérations et outils",
            "Operazioni e strumenti",
            "Operaciones y herramientas",
            "Operações e ferramentas",
            "Betrieb und Werkzeuge",
            "Üzemeltetés és eszközök",
        ],
    ),
    (
        "Read",
        [
            "Lecture", "Lettura", "Lectura", "Leitura", "Lesen", "Olvasás",
        ],
    ),
    (
        "Search",
        [
            "Recherche",
            "Ricerca",
            "Búsqueda",
            "Pesquisa",
            "Suche",
            "Keresés",
        ],
    ),
    (
        "Write",
        [
            "Écriture",
            "Scrittura",
            "Escritura",
            "Escrita",
            "Schreiben",
            "Írás",
        ],
    ),
    (
        "Notify",
        [
            "Notification",
            "Notifica",
            "Notificación",
            "Notificação",
            "Benachrichtigen",
            "Értesítés",
        ],
    ),
    (
        "Act",
        ["Action", "Azione", "Acción", "Ação", "Handeln", "Cselekvés"],
    ),
    (
        "Could not save",
        [
            "Impossible d'enregistrer",
            "Impossibile salvare",
            "No pude guardar",
            "Não consegui salvar",
            "Konnte nicht speichern",
            "Nem sikerült menteni",
        ],
    ),
    (
        "Online access is off. Turn it on first.",
        [
            "L'accès en ligne est désactivé. Activez-le d'abord.",
            "L'accesso online è disattivato. Attivalo prima.",
            "El acceso en línea está desactivado. Actívalo primero.",
            "O acesso online está desativado. Ative-o primeiro.",
            "Der Online-Zugriff ist aus. Schalte ihn zuerst ein.",
            "Az online hozzáférés ki van kapcsolva. Előbb kapcsold be.",
        ],
    ),
    (
        "Waiting for the answer…",
        [
            "En attente de la réponse…",
            "In attesa della risposta…",
            "Esperando la respuesta…",
            "Aguardando a resposta…",
            "Warte auf die Antwort…",
            "Várom a választ…",
        ],
    ),
    (
        "Connected as",
        [
            "Connecté en tant que",
            "Connesso come",
            "Conectado como",
            "Conectado como",
            "Verbunden als",
            "Bejelentkezve mint",
        ],
    ),
    (
        "repositories",
        [
            "dépôts",
            "repository",
            "repositorios",
            "repositórios",
            "Repositories",
            "tárolók",
        ],
    ),
    (
        "Not a text file.",
        [
            "Ce n'est pas un fichier texte.",
            "Non è un file di testo.",
            "No es un archivo de texto.",
            "Não é um arquivo de texto.",
            "Keine Textdatei.",
            "Nem szövegfájl.",
        ],
    ),
    (
        "Added «{}» from the repository",
        [
            "«{}» ajouté depuis le dépôt",
            "«{}» aggiunto dal repository",
            "«{}» añadido desde el repositorio",
            "«{}» adicionado do repositório",
            "«{}» aus dem Repository hinzugefügt",
            "«{}» hozzáadva a tárolóból",
        ],
    ),
    (
        "Added the open issues",
        [
            "Tickets ouverts ajoutés",
            "Problemi aperti aggiunti",
            "Incidencias abiertas añadidas",
            "Problemas abertos adicionados",
            "Offene Issues hinzugefügt",
            "A nyitott hibajegyek hozzáadva",
        ],
    ),
    (
        "No open issues.",
        [
            "Aucun ticket ouvert.",
            "Nessun problema aperto.",
            "No hay incidencias abiertas.",
            "Nenhum problema aberto.",
            "Keine offenen Issues.",
            "Nincs nyitott hibajegy.",
        ],
    ),
    (
        "Added the web page",
        [
            "Page web ajoutée",
            "Pagina web aggiunta",
            "Página web añadida",
            "Página web adicionada",
            "Webseite hinzugefügt",
            "A weboldal hozzáadva",
        ],
    ),
    (
        "Keep the token on this computer",
        [
            "Garder le jeton sur cet ordinateur",
            "Conserva il token su questo computer",
            "Guardar el token en este equipo",
            "Manter o token neste computador",
            "Token auf diesem Computer behalten",
            "A token megtartása ezen a gépen",
        ],
    ),
    (
        "Allow access to the internet",
        [
            "Autoriser l'accès à Internet",
            "Consenti l'accesso a Internet",
            "Permitir el acceso a Internet",
            "Permitir o acesso à Internet",
            "Internetzugriff erlauben",
            "Internet-hozzáférés engedélyezése",
        ],
    ),
    (
        "Name (e.g. my GitHub)",
        [
            "Nom (par ex. mon GitHub)",
            "Nome (es. il mio GitHub)",
            "Nombre (p. ej. mi GitHub)",
            "Nome (ex.: meu GitHub)",
            "Name (z. B. mein GitHub)",
            "Név (pl. a GitHubom)",
        ],
    ),
    (
        "https://… address of a page",
        [
            "https://… adresse d'une page",
            "https://… indirizzo di una pagina",
            "https://… dirección de una página",
            "https://… endereço de uma página",
            "https://… Adresse einer Seite",
            "https://… egy oldal címe",
        ],
    ),
    (
        "Off by default. While off, the program only talks to the colibri server on this computer. When on, it sends requests only when you press a button, to the address you chose; it never sends your conversations anywhere.",
        [
            "Désactivé par défaut. Tant qu'il est désactivé, le programme ne parle qu'au serveur colibri de cet ordinateur. Activé, il n'envoie des requêtes que lorsque vous appuyez sur un bouton, à l'adresse choisie ; il n'envoie jamais vos conversations ailleurs.",
            "Disattivato per impostazione predefinita. Finché è disattivato, il programma parla solo con il server colibri di questo computer. Se attivo, invia richieste solo quando premi un pulsante, all'indirizzo scelto; non invia mai le tue conversazioni altrove.",
            "Desactivado por defecto. Mientras esté desactivado, el programa solo habla con el servidor colibri de este equipo. Activado, envía solicitudes solo cuando pulsas un botón, a la dirección elegida; nunca envía tus conversaciones a ningún sitio.",
            "Desativado por padrão. Enquanto estiver desativado, o programa só fala com o servidor colibri deste computador. Ativado, envia pedidos apenas quando você aperta um botão, ao endereço escolhido; nunca envia suas conversas a lugar nenhum.",
            "Standardmäßig aus. Solange er aus ist, spricht das Programm nur mit dem colibri-Server auf diesem Computer. Eingeschaltet sendet es Anfragen nur, wenn du eine Schaltfläche drückst, an die gewählte Adresse; deine Unterhaltungen sendet es nie irgendwohin.",
            "Alapértelmezés szerint kikapcsolva. Amíg ki van kapcsolva, a program csak a gépen futó colibri szerverrel beszél. Bekapcsolva csak akkor küld kérést, ha megnyomsz egy gombot, a választott címre; a beszélgetéseidet sehová sem küldi.",
        ],
    ),
    (
        "I need Python for internet access.",
        [
            "J'ai besoin de Python pour accéder à Internet.",
            "Mi serve Python per accedere a Internet.",
            "Necesito Python para acceder a Internet.",
            "Preciso do Python para acessar a Internet.",
            "Ich brauche Python für den Internetzugriff.",
            "Az internet eléréséhez Python kell.",
        ],
    ),
    (
        "Read a web page",
        [
            "Lire une page web",
            "Leggi una pagina web",
            "Leer una página web",
            "Ler uma página web",
            "Webseite lesen",
            "Weboldal olvasása",
        ],
    ),
    (
        "Connected platforms",
        [
            "Plateformes connectées",
            "Piattaforme connesse",
            "Plataformas conectadas",
            "Plataformas conectadas",
            "Verbundene Plattformen",
            "Csatlakoztatott platformok",
        ],
    ),
    (
        "no token",
        [
            "sans jeton",
            "senza token",
            "sin token",
            "sem token",
            "ohne Token",
            "token nélkül",
        ],
    ),
    (
        "token saved",
        [
            "jeton enregistré",
            "token salvato",
            "token guardado",
            "token salvo",
            "Token gespeichert",
            "token mentve",
        ],
    ),
    (
        "token for this session only",
        [
            "jeton pour cette session seulement",
            "token solo per questa sessione",
            "token solo para esta sesión",
            "token só para esta sessão",
            "Token nur für diese Sitzung",
            "token csak erre a munkamenetre",
        ],
    ),
    (
        "No platform yet. Add one below.",
        [
            "Aucune plateforme pour l'instant. Ajoutez-en une ci-dessous.",
            "Nessuna piattaforma per ora. Aggiungine una qui sotto.",
            "Aún no hay plataformas. Añade una abajo.",
            "Ainda não há plataformas. Adicione uma abaixo.",
            "Noch keine Plattform. Füge unten eine hinzu.",
            "Még nincs platform. Adj hozzá egyet lent.",
        ],
    ),
    (
        "Add a platform",
        [
            "Ajouter une plateforme",
            "Aggiungi una piattaforma",
            "Añadir una plataforma",
            "Adicionar uma plataforma",
            "Plattform hinzufügen",
            "Platform hozzáadása",
        ],
    ),
    (
        "API address (change it for your own server)",
        [
            "Adresse de l'API (changez-la pour votre propre serveur)",
            "Indirizzo API (cambialo per il tuo server)",
            "Dirección de la API (cámbiala para tu propio servidor)",
            "Endereço da API (mude para o seu próprio servidor)",
            "API-Adresse (für den eigenen Server ändern)",
            "API-cím (saját szerverhez írd át)",
        ],
    ),
    (
        "Access token (optional for public content)",
        [
            "Jeton d'accès (facultatif pour le contenu public)",
            "Token di accesso (facoltativo per contenuti pubblici)",
            "Token de acceso (opcional para contenido público)",
            "Token de acesso (opcional para conteúdo público)",
            "Zugriffstoken (für öffentliche Inhalte optional)",
            "Hozzáférési token (nyilvános tartalomhoz nem kötelező)",
        ],
    ),
    (
        "Make a read-only token here:",
        [
            "Créez ici un jeton en lecture seule :",
            "Crea qui un token di sola lettura:",
            "Crea aquí un token de solo lectura:",
            "Crie aqui um token somente leitura:",
            "Erstelle hier ein schreibgeschütztes Token:",
            "Itt készíts csak olvasható tokent:",
        ],
    ),
    (
        "If unticked, the token stays in memory until you close the program. If ticked, it is saved unencrypted in a file only your account can read. Use a token with the fewest rights.",
        [
            "Si la case est décochée, le jeton reste en mémoire jusqu'à la fermeture du programme. Si elle est cochée, il est enregistré non chiffré dans un fichier lisible uniquement par votre compte. Utilisez un jeton aux droits minimaux.",
            "Se non è spuntata, il token resta in memoria finché non chiudi il programma. Se è spuntata, viene salvato non cifrato in un file leggibile solo dal tuo account. Usa un token con i minimi diritti.",
            "Si no está marcada, el token queda en memoria hasta que cierres el programa. Si está marcada, se guarda sin cifrar en un archivo que solo tu cuenta puede leer. Usa un token con los mínimos permisos.",
            "Se desmarcada, o token fica na memória até você fechar o programa. Se marcada, é salvo sem criptografia em um arquivo que só a sua conta pode ler. Use um token com o mínimo de permissões.",
            "Ohne Haken bleibt das Token im Speicher, bis du das Programm schließt. Mit Haken wird es unverschlüsselt in einer Datei gespeichert, die nur dein Konto lesen kann. Nutze ein Token mit möglichst wenigen Rechten.",
            "Ha nincs kipipálva, a token a program bezárásáig a memóriában marad. Ha ki van pipálva, titkosítatlanul mentődik egy fájlba, amelyet csak a fiókod olvashat. Minimális jogú tokent használj.",
        ],
    ),
    (
        "Test",
        ["Tester", "Prova", "Probar", "Testar", "Testen", "Teszt"],
    ),
    (
        "List my repositories",
        [
            "Lister mes dépôts",
            "Elenca i miei repository",
            "Listar mis repositorios",
            "Listar meus repositórios",
            "Meine Repositories anzeigen",
            "Tárolóim listázása",
        ],
    ),
    (
        "Repository:",
        [
            "Dépôt :",
            "Repository:",
            "Repositorio:",
            "Repositório:",
            "Repository:",
            "Tároló:",
        ],
    ),
    (
        "Open",
        ["Ouvrir", "Apri", "Abrir", "Abrir", "Öffnen", "Megnyitás"],
    ),
    (
        "Open issues → chat",
        [
            "Tickets ouverts → chat",
            "Problemi aperti → chat",
            "Incidencias abiertas → chat",
            "Problemas abertos → chat",
            "Offene Issues → Chat",
            "Nyitott hibajegyek → chat",
        ],
    ),
    (
        "Add to chat",
        [
            "Ajouter au chat",
            "Aggiungi alla chat",
            "Añadir al chat",
            "Adicionar ao chat",
            "Zum Chat hinzufügen",
            "Hozzáadás a chathez",
        ],
    ),
    (
        "Platform catalogue…",
        [
            "Catalogue de plateformes…",
            "Catalogo di piattaforme…",
            "Catálogo de plataformas…",
            "Catálogo de plataformas…",
            "Plattformkatalog…",
            "Platformkatalógus…",
        ],
    ),
    (
        "Only what works now",
        [
            "Seulement ce qui marche maintenant",
            "Solo ciò che funziona ora",
            "Solo lo que funciona ahora",
            "Apenas o que funciona agora",
            "Nur was jetzt funktioniert",
            "Csak ami most működik",
        ],
    ),
    (
        "Where a local model can connect and how much it can do there, from the most careful (read) to the boldest (act). “Ready” means this program does it today; “via web” works now through an address or a webhook; “planned” does not exist yet. Check each platform's own site.",
        [
            "Où un modèle local peut se connecter et ce qu'il peut y faire, du plus prudent (lire) au plus audacieux (agir). « Prêt » signifie que ce programme le fait aujourd'hui ; « via le web » fonctionne déjà par une adresse ou un webhook ; « prévu » n'existe pas encore. Vérifiez le site de chaque plateforme.",
            "Dove un modello locale può connettersi e quanto può fare lì, dal più prudente (leggere) al più audace (agire). «Pronto» significa che il programma lo fa oggi; «via web» funziona già tramite un indirizzo o un webhook; «previsto» non esiste ancora. Controlla il sito di ogni piattaforma.",
            "Dónde puede conectarse un modelo local y cuánto puede hacer allí, de lo más prudente (leer) a lo más audaz (actuar). «Listo» significa que este programa lo hace hoy; «vía web» ya funciona mediante una dirección o un webhook; «previsto» aún no existe. Revisa el sitio de cada plataforma.",
            "Onde um modelo local pode se conectar e quanto pode fazer lá, do mais prudente (ler) ao mais ousado (agir). «Pronto» significa que este programa já faz isso; «via web» já funciona por um endereço ou webhook; «planejado» ainda não existe. Confira o site de cada plataforma.",
            "Wo sich ein lokales Modell verbinden kann und wie viel es dort darf, vom Vorsichtigsten (lesen) bis zum Mutigsten (handeln). „Bereit“ heißt, das Programm kann es heute; „über Web“ funktioniert schon per Adresse oder Webhook; „geplant“ gibt es noch nicht. Prüfe die Seite jeder Plattform selbst.",
            "Hol csatlakozhat egy helyi modell és mit tehet ott, a legóvatosabbtól (olvasás) a legmerészebbig (cselekvés). A „kész” azt jelenti, hogy a program ma megteszi; a „weben át” már most működik címmel vagy webhookkal; a „tervezett” még nem létezik. Ellenőrizd minden platform saját oldalát.",
        ],
    ),
    (
        "Up to level:",
        [
            "Jusqu'au niveau :",
            "Fino al livello:",
            "Hasta el nivel:",
            "Até o nível:",
            "Bis Stufe:",
            "Legfeljebb eddig a szintig:",
        ],
    ),
    (
        "ready",
        ["prêt", "pronto", "listo", "pronto", "bereit", "kész"],
    ),
    (
        "via web",
        [
            "via le web",
            "via web",
            "vía web",
            "via web",
            "über Web",
            "weben át",
        ],
    ),
    (
        "planned",
        [
            "prévu",
            "previsto",
            "previsto",
            "planejado",
            "geplant",
            "tervezett",
        ],
    ),
    (
        "no account",
        [
            "sans compte",
            "senza account",
            "sin cuenta",
            "sem conta",
            "ohne Konto",
            "fiók nélkül",
        ],
    ),
    (
        "token",
        ["jeton", "token", "token", "token", "Token", "token"],
    ),
    (
        "sign-in",
        [
            "connexion",
            "accesso",
            "inicio de sesión",
            "login",
            "Anmeldung",
            "bejelentkezés",
        ],
    ),
    (
        "webhook address",
        [
            "adresse webhook",
            "indirizzo webhook",
            "dirección webhook",
            "endereço webhook",
            "Webhook-Adresse",
            "webhook-cím",
        ],
    ),
    (
        "The levels",
        [
            "Les niveaux",
            "I livelli",
            "Los niveles",
            "Os níveis",
            "Die Stufen",
            "A szintek",
        ],
    ),
    (
        "Creative models",
        [
            "Modèles créatifs",
            "Modelli creativi",
            "Modelos creativos",
            "Modelos criativos",
            "Kreative Modelle",
            "Kreatív modellek",
        ],
    ),
    (
        "Pictures",
        [
            "Images",
            "Immagini",
            "Imágenes",
            "Imagens",
            "Bilder",
            "Képek",
        ],
    ),
    (
        "Video",
        ["Vidéo", "Video", "Vídeo", "Vídeo", "Video", "Videó"],
    ),
    (
        "Voice (text → sound)",
        [
            "Voix (texte → son)",
            "Voce (testo → suono)",
            "Voz (texto → sonido)",
            "Voz (texto → som)",
            "Stimme (Text → Ton)",
            "Hang (szöveg → hang)",
        ],
    ),
    (
        "Listening (sound → text)",
        [
            "Écoute (son → texte)",
            "Ascolto (suono → testo)",
            "Escucha (sonido → texto)",
            "Escuta (som → texto)",
            "Zuhören (Ton → Text)",
            "Hallgatás (hang → szöveg)",
        ],
    ),
    (
        "Music",
        ["Musique", "Musica", "Música", "Música", "Musik", "Zene"],
    ),
    (
        "The server address is not valid (https://, or http:// on your own network only).",
        [
            "L'adresse du serveur n'est pas valide (https://, ou http:// sur votre réseau uniquement).",
            "L'indirizzo del server non è valido (https://, o http:// solo nella tua rete).",
            "La dirección del servidor no es válida (https://, o http:// solo en tu red).",
            "O endereço do servidor não é válido (https://, ou http:// apenas na sua rede).",
            "Die Serveradresse ist ungültig (https://, oder http:// nur im eigenen Netz).",
            "A szervercím érvénytelen (https://, vagy http:// csak a saját hálózaton).",
        ],
    ),
    (
        "Making it… this can take a few minutes.",
        [
            "Création en cours… cela peut prendre quelques minutes.",
            "Sto creando… può richiedere qualche minuto.",
            "Creándola… puede tardar unos minutos.",
            "Criando… pode levar alguns minutos.",
            "Wird erstellt… das kann einige Minuten dauern.",
            "Készül… ez néhány percig is eltarthat.",
        ],
    ),
    (
        "It failed",
        [
            "Échec",
            "Non è riuscito",
            "Falló",
            "Falhou",
            "Fehlgeschlagen",
            "Nem sikerült",
        ],
    ),
    (
        "Only what can be used commercially",
        [
            "Seulement ce qui est utilisable commercialement",
            "Solo ciò che si può usare commercialmente",
            "Solo lo que se puede usar comercialmente",
            "Apenas o que pode ser usado comercialmente",
            "Nur was kommerziell nutzbar ist",
            "Csak ami kereskedelmileg használható",
        ],
    ),
    (
        "Describe the picture…",
        [
            "Décrivez l'image…",
            "Descrivi l'immagine…",
            "Describe la imagen…",
            "Descreva a imagem…",
            "Beschreibe das Bild…",
            "Írd le a képet…",
        ],
    ),
    (
        "Open models that make pictures, video, voice and music or understand sound, looked up online in November 2026. colibri itself serves only Qwen-Image-2.1; the others run in another program (ComfyUI, LocalAI) that you can point this app at. Licences differ: check each model's page.",
        [
            "Modèles ouverts qui créent images, vidéo, voix et musique ou comprennent le son, recherchés en ligne en novembre 2026. colibri ne sert lui-même que Qwen-Image-2.1 ; les autres tournent dans un autre programme (ComfyUI, LocalAI) vers lequel vous pouvez orienter cette application. Les licences diffèrent : vérifiez la page de chaque modèle.",
            "Modelli aperti che creano immagini, video, voce e musica o capiscono il suono, cercati online a novembre 2026. colibri stesso serve solo Qwen-Image-2.1; gli altri girano in un altro programma (ComfyUI, LocalAI) a cui puoi indirizzare questa app. Le licenze differiscono: controlla la pagina di ogni modello.",
            "Modelos abiertos que crean imágenes, vídeo, voz y música o entienden el sonido, buscados en línea en noviembre de 2026. colibri solo sirve Qwen-Image-2.1; los demás corren en otro programa (ComfyUI, LocalAI) al que puedes apuntar esta aplicación. Las licencias difieren: revisa la página de cada modelo.",
            "Modelos abertos que criam imagens, vídeo, voz e música ou entendem o som, pesquisados online em novembro de 2026. O próprio colibri serve apenas o Qwen-Image-2.1; os outros rodam em outro programa (ComfyUI, LocalAI) para o qual você pode apontar este aplicativo. As licenças diferem: confira a página de cada modelo.",
            "Offene Modelle, die Bilder, Video, Stimme und Musik erzeugen oder Ton verstehen, im November 2026 online recherchiert. colibri selbst bedient nur Qwen-Image-2.1; die anderen laufen in einem anderen Programm (ComfyUI, LocalAI), auf das du diese App richten kannst. Lizenzen unterscheiden sich: prüfe die Seite jedes Modells.",
            "Nyílt modellek, amelyek képet, videót, hangot és zenét készítenek vagy értik a hangot, 2026 novemberében online keresve. A colibri maga csak a Qwen-Image-2.1-et szolgálja ki; a többi másik programban fut (ComfyUI, LocalAI), amelyre rá lehet irányítani ezt az alkalmazást. A licencek eltérnek: nézd meg minden modell oldalát.",
        ],
    ),
    (
        "served by colibri",
        [
            "servi par colibri",
            "servito da colibri",
            "servido por colibri",
            "servido pelo colibri",
            "von colibri bedient",
            "a colibri szolgálja ki",
        ],
    ),
    (
        "commercial: yes",
        [
            "commercial : oui",
            "commerciale: sì",
            "comercial: sí",
            "comercial: sim",
            "kommerziell: ja",
            "kereskedelmi: igen",
        ],
    ),
    (
        "commercial: no",
        [
            "commercial : non",
            "commerciale: no",
            "comercial: no",
            "comercial: não",
            "kommerziell: nein",
            "kereskedelmi: nem",
        ],
    ),
    (
        "commercial: check",
        [
            "commercial : à vérifier",
            "commerciale: da verificare",
            "comercial: verificar",
            "comercial: verificar",
            "kommerziell: prüfen",
            "kereskedelmi: ellenőrizd",
        ],
    ),
    (
        "Programs that run them",
        [
            "Programmes qui les exécutent",
            "Programmi che li eseguono",
            "Programas que los ejecutan",
            "Programas que os executam",
            "Programme, die sie ausführen",
            "Programok, amelyek futtatják őket",
        ],
    ),
    (
        "Make a picture",
        [
            "Créer une image",
            "Crea un'immagine",
            "Crear una imagen",
            "Criar uma imagem",
            "Bild erzeugen",
            "Kép készítése",
        ],
    ),
    (
        "Server:",
        [
            "Serveur :",
            "Server:",
            "Servidor:",
            "Servidor:",
            "Server:",
            "Szerver:",
        ],
    ),
    (
        "Generate",
        [
            "Générer",
            "Genera",
            "Generar",
            "Gerar",
            "Erzeugen",
            "Létrehozás",
        ],
    ),
    (
        "Open the picture",
        [
            "Ouvrir l'image",
            "Apri l'immagine",
            "Abrir la imagen",
            "Abrir a imagem",
            "Bild öffnen",
            "Kép megnyitása",
        ],
    ),
    (
        "Not yet tried against a real colibri server: the request follows the OpenAI convention. With a model other than Qwen-Image-2.1 the server may answer with an error.",
        [
            "Pas encore essayé avec un vrai serveur colibri : la requête suit la convention OpenAI. Avec un modèle autre que Qwen-Image-2.1, le serveur peut répondre par une erreur.",
            "Non ancora provato con un vero server colibri: la richiesta segue la convenzione OpenAI. Con un modello diverso da Qwen-Image-2.1 il server può rispondere con un errore.",
            "Aún no probado con un servidor colibri real: la solicitud sigue la convención de OpenAI. Con un modelo distinto de Qwen-Image-2.1 el servidor puede responder con un error.",
            "Ainda não testado com um servidor colibri real: o pedido segue a convenção da OpenAI. Com um modelo diferente do Qwen-Image-2.1 o servidor pode responder com um erro.",
            "Noch nicht mit einem echten colibri-Server getestet: Die Anfrage folgt der OpenAI-Konvention. Mit einem anderen Modell als Qwen-Image-2.1 kann der Server mit einem Fehler antworten.",
            "Még nem próbáltam valódi colibri szerverrel: a kérés az OpenAI-konvenciót követi. A Qwen-Image-2.1-től eltérő modellnél a szerver hibával válaszolhat.",
        ],
    ),
    (
        "Read aloud",
        [
            "Lire à voix haute",
            "Leggi ad alta voce",
            "Leer en voz alta",
            "Ler em voz alta",
            "Vorlesen",
            "Felolvasás",
        ],
    ),
    (
        "Stop reading",
        [
            "Arrêter la lecture",
            "Interrompi la lettura",
            "Detener la lectura",
            "Parar a leitura",
            "Vorlesen beenden",
            "Felolvasás leállítása",
        ],
    ),
    (
        "Accessibility",
        [
            "Accessibilité",
            "Accessibilità",
            "Accesibilidad",
            "Acessibilidade",
            "Barrierefreiheit",
            "Akadálymentesség",
        ],
    ),
    (
        "Plain language",
        [
            "Langage simple",
            "Linguaggio semplice",
            "Lenguaje sencillo",
            "Linguagem simples",
            "Einfache Sprache",
            "Egyszerű nyelv",
        ],
    ),
    (
        "Low vision",
        [
            "Malvoyance",
            "Ipovisione",
            "Baja visión",
            "Baixa visão",
            "Sehbehinderung",
            "Gyengénlátás",
        ],
    ),
    (
        "Big text, high contrast, a special typeface, big buttons.",
        [
            "Grand texte, fort contraste, une police spéciale, gros boutons.",
            "Testo grande, contrasto alto, un carattere speciale, pulsanti grandi.",
            "Texto grande, alto contraste, una tipografía especial, botones grandes.",
            "Texto grande, alto contraste, uma fonte especial, botões grandes.",
            "Große Schrift, hoher Kontrast, eine spezielle Schriftart, große Schaltflächen.",
            "Nagy szöveg, nagy kontraszt, különleges betűtípus, nagy gombok.",
        ],
    ),
    (
        "Blind (screen reader)",
        [
            "Non-voyant (lecteur d'écran)",
            "Non vedente (lettore di schermo)",
            "Ciego (lector de pantalla)",
            "Cego (leitor de tela)",
            "Blind (Screenreader)",
            "Vak (képernyőolvasó)",
        ],
    ),
    (
        "Answers are read aloud, no movement, few menus.",
        [
            "Les réponses sont lues à voix haute, sans mouvement, peu de menus.",
            "Le risposte vengono lette ad alta voce, senza movimento, pochi menu.",
            "Las respuestas se leen en voz alta, sin movimiento, pocos menús.",
            "As respostas são lidas em voz alta, sem movimento, poucos menus.",
            "Antworten werden vorgelesen, ohne Bewegung, wenige Menüs.",
            "A válaszokat felolvassa, mozgás nélkül, kevés menüvel.",
        ],
    ),
    (
        "Hard to control movements",
        [
            "Mouvements difficiles à contrôler",
            "Movimenti difficili da controllare",
            "Movimientos difíciles de controlar",
            "Movimentos difíceis de controlar",
            "Schwer kontrollierbare Bewegungen",
            "Nehezen irányítható mozgás",
        ],
    ),
    (
        "Very big buttons with space between them.",
        [
            "Très gros boutons avec de l'espace entre eux.",
            "Pulsanti molto grandi con spazio tra loro.",
            "Botones muy grandes con espacio entre ellos.",
            "Botões muito grandes com espaço entre eles.",
            "Sehr große Schaltflächen mit Abstand dazwischen.",
            "Nagyon nagy gombok, köztük hellyel.",
        ],
    ),
    (
        "Dyslexia",
        [
            "Dyslexie",
            "Dislessia",
            "Dislexia",
            "Dislexia",
            "Legasthenie",
            "Diszlexia",
        ],
    ),
    (
        "A readable typeface, bigger text, looser lines, no animation.",
        [
            "Une police lisible, texte agrandi, lignes plus aérées, sans animation.",
            "Un carattere leggibile, testo più grande, righe più distanziate, nessuna animazione.",
            "Una tipografía legible, texto más grande, líneas más espaciadas, sin animación.",
            "Uma fonte legível, texto maior, linhas mais espaçadas, sem animação.",
            "Eine gut lesbare Schrift, größerer Text, mehr Zeilenabstand, keine Animation.",
            "Jól olvasható betűtípus, nagyobb szöveg, ritkább sorok, animáció nélkül.",
        ],
    ),
    (
        "Easier reading / keep it simple",
        [
            "Lecture facilitée / rester simple",
            "Lettura facilitata / tenere tutto semplice",
            "Lectura más fácil / mantenerlo simple",
            "Leitura facilitada / manter simples",
            "Leichter lesen / einfach halten",
            "Könnyebb olvasás / egyszerűen",
        ],
    ),
    (
        "Few menus, lots of space, no animation, and answers in plain language.",
        [
            "Peu de menus, beaucoup d'espace, sans animation, et réponses en langage simple.",
            "Pochi menu, molto spazio, nessuna animazione e risposte in linguaggio semplice.",
            "Pocos menús, mucho espacio, sin animación y respuestas en lenguaje sencillo.",
            "Poucos menus, muito espaço, sem animação e respostas em linguagem simples.",
            "Wenige Menüs, viel Platz, keine Animation und Antworten in einfacher Sprache.",
            "Kevés menü, sok hely, animáció nélkül, és válaszok egyszerű nyelven.",
        ],
    ),
    (
        "Colour blindness",
        [
            "Daltonisme",
            "Daltonismo",
            "Daltonismo",
            "Daltonismo",
            "Farbenblindheit",
            "Színtévesztés",
        ],
    ),
    (
        "Blue and orange instead of green and red.",
        [
            "Bleu et orange au lieu de vert et rouge.",
            "Blu e arancione al posto di verde e rosso.",
            "Azul y naranja en lugar de verde y rojo.",
            "Azul e laranja em vez de verde e vermelho.",
            "Blau und Orange statt Grün und Rot.",
            "Kék és narancs a zöld és piros helyett.",
        ],
    ),
    (
        "Deaf or hard of hearing",
        [
            "Sourd ou malentendant",
            "Sordo o con problemi di udito",
            "Sordo o con dificultades auditivas",
            "Surdo ou com dificuldade de audição",
            "Gehörlos oder schwerhörig",
            "Siket vagy nagyothalló",
        ],
    ),
    (
        "The app uses no sounds; it tells you visually when an answer is ready.",
        [
            "L'application n'utilise aucun son ; elle vous indique visuellement quand une réponse est prête.",
            "L'app non usa suoni; ti avvisa visivamente quando una risposta è pronta.",
            "La aplicación no usa sonidos; te avisa visualmente cuando una respuesta está lista.",
            "O aplicativo não usa sons; avisa visualmente quando uma resposta está pronta.",
            "Die App benutzt keine Töne; sie zeigt visuell an, wenn eine Antwort fertig ist.",
            "Az alkalmazás nem használ hangot; vizuálisan jelzi, ha kész egy válasz.",
        ],
    ),
    (
        "A code block is here",
        [
            "Voici un bloc de code",
            "Qui c'è un blocco di codice",
            "Aquí hay un bloque de código",
            "Aqui há um bloco de código",
            "Hier ist ein Codeblock",
            "Itt egy kódblokk van",
        ],
    ),
    (
        "Cannot read aloud",
        [
            "Lecture à voix haute impossible",
            "Impossibile leggere ad alta voce",
            "No se puede leer en voz alta",
            "Não é possível ler em voz alta",
            "Vorlesen nicht möglich",
            "Nem lehet felolvasni",
        ],
    ),
    (
        "No speech program found. On Linux, install espeak-ng.",
        [
            "Aucun programme de synthèse vocale trouvé. Sous Linux, installez espeak-ng.",
            "Nessun programma di sintesi vocale trovato. Su Linux, installa espeak-ng.",
            "No se encontró ningún programa de voz. En Linux, instala espeak-ng.",
            "Nenhum programa de voz encontrado. No Linux, instale o espeak-ng.",
            "Kein Sprachprogramm gefunden. Unter Linux espeak-ng installieren.",
            "Nem találtam beszédprogramot. Linuxon telepítsd az espeak-ng-t.",
        ],
    ),
    (
        "Pick what suits you. Everything changes at once and can be undone any time. You can combine as many options as you like.",
        [
            "Choisissez ce qui vous convient. Tout change immédiatement et peut être annulé à tout moment. Vous pouvez combiner autant d'options que vous voulez.",
            "Scegli ciò che ti serve. Tutto cambia subito e si può annullare in qualsiasi momento. Puoi combinare quante opzioni vuoi.",
            "Elige lo que te convenga. Todo cambia al instante y se puede deshacer en cualquier momento. Puedes combinar tantas opciones como quieras.",
            "Escolha o que lhe convém. Tudo muda na hora e pode ser desfeito a qualquer momento. Pode combinar quantas opções quiser.",
            "Wähle, was zu dir passt. Alles ändert sich sofort und lässt sich jederzeit rückgängig machen. Du kannst beliebig viele Optionen kombinieren.",
            "Válaszd ki, ami neked megfelel. Minden azonnal változik, és bármikor visszavonható. Annyi lehetőséget kombinálhatsz, amennyit szeretnél.",
        ],
    ),
    (
        "Quick start",
        [
            "Démarrage rapide",
            "Avvio rapido",
            "Inicio rápido",
            "Início rápido",
            "Schnellstart",
            "Gyors indítás",
        ],
    ),
    (
        "Back to the usual settings",
        [
            "Retour aux réglages habituels",
            "Torna alle impostazioni abituali",
            "Volver a los ajustes habituales",
            "Voltar às configurações habituais",
            "Zurück zu den üblichen Einstellungen",
            "Vissza a megszokott beállításokhoz",
        ],
    ),
    (
        "Seeing",
        ["Vue", "Vista", "Vista", "Visão", "Sehen", "Látás"],
    ),
    (
        "Space between lines and buttons",
        [
            "Espace entre les lignes et les boutons",
            "Spazio tra righe e pulsanti",
            "Espacio entre líneas y botones",
            "Espaço entre linhas e botões",
            "Abstand zwischen Zeilen und Schaltflächen",
            "Hely a sorok és a gombok között",
        ],
    ),
    (
        "Typeface made for low vision (Atkinson Hyperlegible)",
        [
            "Police conçue pour la malvoyance (Atkinson Hyperlegible)",
            "Carattere pensato per l'ipovisione (Atkinson Hyperlegible)",
            "Tipografía hecha para baja visión (Atkinson Hyperlegible)",
            "Fonte feita para baixa visão (Atkinson Hyperlegible)",
            "Schriftart für Sehbehinderung (Atkinson Hyperlegible)",
            "Gyengénlátóknak tervezett betűtípus (Atkinson Hyperlegible)",
        ],
    ),
    (
        "Colours for colour blindness (blue and orange instead of green and red)",
        [
            "Couleurs pour le daltonisme (bleu et orange au lieu de vert et rouge)",
            "Colori per il daltonismo (blu e arancione al posto di verde e rosso)",
            "Colores para el daltonismo (azul y naranja en lugar de verde y rojo)",
            "Cores para daltonismo (azul e laranja em vez de verde e vermelho)",
            "Farben für Farbenblindheit (Blau und Orange statt Grün und Rot)",
            "Színek színtévesztéshez (kék és narancs a zöld és piros helyett)",
        ],
    ),
    (
        "A thick outline around the item the keyboard is on",
        [
            "Un contour épais autour de l'élément où se trouve le clavier",
            "Un contorno spesso attorno all'elemento su cui si trova la tastiera",
            "Un contorno grueso alrededor del elemento donde está el teclado",
            "Um contorno grosso ao redor do item em que o teclado está",
            "Ein dicker Rahmen um das Element, auf dem die Tastatur steht",
            "Vastag keret az elem körül, amelyen a billentyűzet áll",
        ],
    ),
    (
        "Reading aloud",
        [
            "Lecture à voix haute",
            "Lettura ad alta voce",
            "Lectura en voz alta",
            "Leitura em voz alta",
            "Vorlesen",
            "Felolvasás",
        ],
    ),
    (
        "Read every answer aloud by itself when it is done",
        [
            "Lire chaque réponse à voix haute toute seule dès qu'elle est terminée",
            "Leggi ogni risposta ad alta voce da sola appena è finita",
            "Leer cada respuesta en voz alta sola cuando termine",
            "Ler cada resposta em voz alta sozinha quando terminar",
            "Jede Antwort automatisch vorlesen, sobald sie fertig ist",
            "Minden válasz automatikus felolvasása, ha kész",
        ],
    ),
    (
        "Speed",
        [
            "Vitesse",
            "Velocità",
            "Velocidad",
            "Velocidade",
            "Geschwindigkeit",
            "Sebesség",
        ],
    ),
    (
        " wpm",
        [
            " mots/min",
            " parole/min",
            " palabras/min",
            " palavras/min",
            " Wörter/Min.",
            " szó/perc",
        ],
    ),
    (
        "Try the voice",
        [
            "Essayer la voix",
            "Prova la voce",
            "Probar la voz",
            "Testar a voz",
            "Stimme testen",
            "Hang kipróbálása",
        ],
    ),
    (
        "Hello! This is how the voice sounds. You can change the speed.",
        [
            "Bonjour ! Voici comment sonne la voix. Vous pouvez changer la vitesse.",
            "Ciao! Ecco come suona la voce. Puoi cambiare la velocità.",
            "¡Hola! Así suena la voz. Puedes cambiar la velocidad.",
            "Olá! É assim que a voz soa. Você pode mudar a velocidade.",
            "Hallo! So klingt die Stimme. Du kannst die Geschwindigkeit ändern.",
            "Szia! Így szól a hang. A sebességet megváltoztathatod.",
        ],
    ),
    (
        "No speech program found. Install espeak-ng (for example “sudo apt install espeak-ng”).",
        [
            "Aucun programme de synthèse vocale trouvé. Installez espeak-ng (par exemple « sudo apt install espeak-ng »).",
            "Nessun programma di sintesi vocale trovato. Installa espeak-ng (per esempio «sudo apt install espeak-ng»).",
            "No se encontró ningún programa de voz. Instala espeak-ng (por ejemplo «sudo apt install espeak-ng»).",
            "Nenhum programa de voz encontrado. Instale o espeak-ng (por exemplo «sudo apt install espeak-ng»).",
            "Kein Sprachprogramm gefunden. Installiere espeak-ng (zum Beispiel „sudo apt install espeak-ng“).",
            "Nem találtam beszédprogramot. Telepítsd az espeak-ng-t (például „sudo apt install espeak-ng”).",
        ],
    ),
    (
        "It uses your system's own voice; on Windows and macOS the voice language is the one set in the system. Every answer also has a “Read aloud” button. Shortcut: Ctrl+Shift+S.",
        [
            "Elle utilise la voix de votre système ; sous Windows et macOS, la langue de la voix est celle réglée dans le système. Chaque réponse a aussi un bouton « Lire à voix haute ». Raccourci : Ctrl+Maj+S.",
            "Usa la voce del tuo sistema; su Windows e macOS la lingua della voce è quella impostata nel sistema. Ogni risposta ha anche un pulsante «Leggi ad alta voce». Scorciatoia: Ctrl+Maiusc+S.",
            "Usa la voz de tu sistema; en Windows y macOS el idioma de la voz es el configurado en el sistema. Cada respuesta tiene también un botón «Leer en voz alta». Atajo: Ctrl+Mayús+S.",
            "Usa a voz do seu sistema; no Windows e no macOS o idioma da voz é o configurado no sistema. Cada resposta também tem um botão «Ler em voz alta». Atalho: Ctrl+Shift+S.",
            "Es wird die Stimme deines Systems verwendet; unter Windows und macOS gilt die im System eingestellte Sprache. Jede Antwort hat außerdem eine Schaltfläche „Vorlesen“. Tastenkürzel: Strg+Umschalt+S.",
            "A rendszer saját hangját használja; Windowson és macOS-en a hang nyelve a rendszerben beállított. Minden válasznál van „Felolvasás” gomb is. Gyorsbillentyű: Ctrl+Shift+S.",
        ],
    ),
    (
        "Movement and hands",
        [
            "Mouvement et mains",
            "Movimento e mani",
            "Movimiento y manos",
            "Movimento e mãos",
            "Bewegung und Hände",
            "Mozgás és kezek",
        ],
    ),
    (
        "Button size",
        [
            "Taille des boutons",
            "Dimensione dei pulsanti",
            "Tamaño de los botones",
            "Tamanho dos botões",
            "Größe der Schaltflächen",
            "Gombok mérete",
        ],
    ),
    (
        "Large",
        ["Grand", "Grande", "Grande", "Grande", "Groß", "Nagy"],
    ),
    (
        "Huge",
        [
            "Très grand",
            "Enorme",
            "Enorme",
            "Enorme",
            "Riesig",
            "Hatalmas",
        ],
    ),
    (
        "The app can be used with the keyboard alone: Tab and Shift+Tab move between items, Space or Enter presses, Esc stops an answer. No mouse, dragging or precise movement is needed. Shortcuts: Ctrl+N new chat, Ctrl+, settings, Ctrl+P prompts, Ctrl+K search, Ctrl+Shift+A this window, Ctrl+Shift+S read the last answer.",
        [
            "L'application s'utilise avec le clavier seul : Tab et Maj+Tab passent d'un élément à l'autre, Espace ou Entrée appuie, Échap arrête une réponse. Ni souris, ni glisser-déposer, ni geste précis nécessaires. Raccourcis : Ctrl+N nouvelle conversation, Ctrl+, réglages, Ctrl+P prompts, Ctrl+K recherche, Ctrl+Maj+A cette fenêtre, Ctrl+Maj+S lire la dernière réponse.",
            "L'app si può usare con la sola tastiera: Tab e Maiusc+Tab passano da un elemento all'altro, Spazio o Invio premono, Esc ferma una risposta. Non servono mouse, trascinamenti o movimenti precisi. Scorciatoie: Ctrl+N nuova chat, Ctrl+, impostazioni, Ctrl+P prompt, Ctrl+K ricerca, Ctrl+Maiusc+A questa finestra, Ctrl+Maiusc+S leggi l'ultima risposta.",
            "La aplicación se puede usar solo con el teclado: Tab y Mayús+Tab pasan de un elemento a otro, Espacio o Intro pulsa, Esc detiene una respuesta. No hace falta ratón, arrastrar ni movimientos precisos. Atajos: Ctrl+N nueva conversación, Ctrl+, ajustes, Ctrl+P prompts, Ctrl+K búsqueda, Ctrl+Mayús+A esta ventana, Ctrl+Mayús+S leer la última respuesta.",
            "O aplicativo pode ser usado só com o teclado: Tab e Shift+Tab passam de um item a outro, Espaço ou Enter pressiona, Esc interrompe uma resposta. Não é preciso mouse, arrastar nem movimentos precisos. Atalhos: Ctrl+N nova conversa, Ctrl+, configurações, Ctrl+P prompts, Ctrl+K busca, Ctrl+Shift+A esta janela, Ctrl+Shift+S ler a última resposta.",
            "Die App lässt sich allein mit der Tastatur bedienen: Tab und Umschalt+Tab wechseln zwischen Elementen, Leertaste oder Enter drückt, Esc stoppt eine Antwort. Maus, Ziehen oder präzise Bewegungen sind nicht nötig. Tastenkürzel: Strg+N neuer Chat, Strg+, Einstellungen, Strg+P Prompts, Strg+K Suche, Strg+Umschalt+A dieses Fenster, Strg+Umschalt+S letzte Antwort vorlesen.",
            "Az alkalmazás egyedül a billentyűzettel is használható: a Tab és a Shift+Tab lép az elemek között, a Szóköz vagy az Enter nyom, az Esc megállít egy választ. Nem kell egér, húzás vagy pontos mozdulat. Gyorsbillentyűk: Ctrl+N új beszélgetés, Ctrl+, beállítások, Ctrl+P promptok, Ctrl+K keresés, Ctrl+Shift+A ez az ablak, Ctrl+Shift+S az utolsó válasz felolvasása.",
        ],
    ),
    (
        "To dictate instead of typing, use your system's dictation: Windows Win+H, macOS press Fn twice, Linux depends on the desktop. It works in any text box here.",
        [
            "Pour dicter au lieu de taper, utilisez la dictée du système : Windows Win+H, macOS appuyez deux fois sur Fn, Linux selon le bureau. Elle fonctionne dans tout champ de texte ici.",
            "Per dettare invece di scrivere usa la dettatura del sistema: Windows Win+H, macOS premi due volte Fn, Linux dipende dal desktop. Funziona in ogni campo di testo qui.",
            "Para dictar en lugar de escribir, usa el dictado del sistema: Windows Win+H, macOS pulsa Fn dos veces, Linux depende del escritorio. Funciona en cualquier cuadro de texto de aquí.",
            "Para ditar em vez de digitar, use o ditado do sistema: Windows Win+H, macOS pressione Fn duas vezes, Linux depende do ambiente. Funciona em qualquer caixa de texto aqui.",
            "Zum Diktieren statt Tippen nutze die Diktierfunktion deines Systems: Windows Win+H, macOS zweimal Fn drücken, Linux je nach Desktop. Sie funktioniert in jedem Textfeld hier.",
            "Gépelés helyett diktálhatsz a rendszer diktálásával: Windows Win+H, macOS kétszer nyomd meg az Fn-t, Linuxon az asztali környezettől függ. Itt minden szövegmezőben működik.",
        ],
    ),
    (
        "Hearing",
        ["Audition", "Udito", "Oído", "Audição", "Hören", "Hallás"],
    ),
    (
        "Flash the taskbar when an answer is ready and the window is in the background",
        [
            "Faire clignoter la barre des tâches quand une réponse est prête et que la fenêtre est en arrière-plan",
            "Fai lampeggiare la barra delle applicazioni quando una risposta è pronta e la finestra è in secondo piano",
            "Parpadear la barra de tareas cuando una respuesta esté lista y la ventana esté en segundo plano",
            "Piscar a barra de tarefas quando uma resposta estiver pronta e a janela estiver em segundo plano",
            "Taskleiste blinken lassen, wenn eine Antwort fertig ist und das Fenster im Hintergrund liegt",
            "A tálca villogtatása, ha kész egy válasz, és az ablak a háttérben van",
        ],
    ),
    (
        "The app makes no sound and never asks you to listen: everything is shown as text.",
        [
            "L'application ne fait aucun bruit et ne demande jamais d'écouter : tout est affiché en texte.",
            "L'app non emette suoni e non chiede mai di ascoltare: tutto è mostrato come testo.",
            "La aplicación no hace ningún sonido y nunca pide escuchar: todo se muestra como texto.",
            "O aplicativo não faz nenhum som e nunca pede para ouvir: tudo é mostrado como texto.",
            "Die App macht keine Geräusche und verlangt nie, etwas zu hören: alles wird als Text gezeigt.",
            "Az alkalmazás nem ad ki hangot, és soha nem kér hallgatást: minden szövegként jelenik meg.",
        ],
    ),
    (
        "Reading and understanding",
        [
            "Lecture et compréhension",
            "Lettura e comprensione",
            "Lectura y comprensión",
            "Leitura e compreensão",
            "Lesen und Verstehen",
            "Olvasás és megértés",
        ],
    ),
    (
        "No animation and no spinning wheels",
        [
            "Sans animation ni roues qui tournent",
            "Nessuna animazione e nessuna rotella che gira",
            "Sin animaciones ni ruedas que giran",
            "Sem animações nem rodinhas girando",
            "Keine Animationen und keine drehenden Räder",
            "Animáció és forgó kerekek nélkül",
        ],
    ),
    (
        "Simple menus: hide the advanced windows",
        [
            "Menus simples : masquer les fenêtres avancées",
            "Menu semplici: nascondi le finestre avanzate",
            "Menús sencillos: ocultar las ventanas avanzadas",
            "Menus simples: ocultar as janelas avançadas",
            "Einfache Menüs: erweiterte Fenster ausblenden",
            "Egyszerű menük: a haladó ablakok elrejtése",
        ],
    ),
    (
        "Answers in plain language",
        [
            "Réponses en langage simple",
            "Risposte in linguaggio semplice",
            "Respuestas en lenguaje sencillo",
            "Respostas em linguagem simples",
            "Antworten in einfacher Sprache",
            "Válaszok egyszerű nyelven",
        ],
    ),
    (
        "Short sentences, everyday words, one idea per sentence. The model follows it as well as it can.",
        [
            "Phrases courtes, mots de tous les jours, une idée par phrase. Le modèle le respecte du mieux qu'il peut.",
            "Frasi brevi, parole di tutti i giorni, un'idea per frase. Il modello la rispetta come può.",
            "Frases cortas, palabras cotidianas, una idea por frase. El modelo lo cumple lo mejor que puede.",
            "Frases curtas, palavras do dia a dia, uma ideia por frase. O modelo segue isso o melhor que pode.",
            "Kurze Sätze, Alltagswörter, ein Gedanke pro Satz. Das Modell hält sich so gut wie möglich daran.",
            "Rövid mondatok, hétköznapi szavak, mondatonként egy gondolat. A modell a lehetőségei szerint betartja.",
        ],
    ),
    (
        "Screen readers",
        [
            "Lecteurs d'écran",
            "Lettori di schermo",
            "Lectores de pantalla",
            "Leitores de tela",
            "Screenreader",
            "Képernyőolvasók",
        ],
    ),
    (
        "The app describes its contents to the system (through AccessKit), so NVDA, JAWS, Narrator, VoiceOver and Orca can read the buttons and texts. I have not yet been able to try the app with a real screen reader: if something is not read well, tell me what and I will fix it. The “Read aloud” button above does not need a screen reader.",
        [
            "L'application décrit son contenu au système (via AccessKit), donc NVDA, JAWS, Narrator, VoiceOver et Orca peuvent lire les boutons et les textes. Je n'ai pas encore pu l'essayer avec un vrai lecteur d'écran : si quelque chose est mal lu, dites-moi quoi et je corrigerai. Le bouton « Lire à voix haute » ci-dessus n'a pas besoin de lecteur d'écran.",
            "L'app descrive il proprio contenuto al sistema (tramite AccessKit), quindi NVDA, JAWS, Narrator, VoiceOver e Orca possono leggere pulsanti e testi. Non ho ancora potuto provarla con un vero lettore di schermo: se qualcosa non viene letto bene, dimmi cosa e correggo. Il pulsante «Leggi ad alta voce» qui sopra non richiede un lettore di schermo.",
            "La aplicación describe su contenido al sistema (mediante AccessKit), así que NVDA, JAWS, Narrator, VoiceOver y Orca pueden leer los botones y textos. Aún no he podido probarla con un lector de pantalla real: si algo no se lee bien, dime qué y lo corrijo. El botón «Leer en voz alta» de arriba no necesita lector de pantalla.",
            "O aplicativo descreve seu conteúdo ao sistema (via AccessKit), então NVDA, JAWS, Narrator, VoiceOver e Orca podem ler os botões e textos. Ainda não pude testá-lo com um leitor de tela real: se algo não for lido bem, diga o que e eu corrijo. O botão «Ler em voz alta» acima não precisa de leitor de tela.",
            "Die App beschreibt ihren Inhalt dem System (über AccessKit), sodass NVDA, JAWS, Narrator, VoiceOver und Orca Schaltflächen und Texte vorlesen können. Ich konnte sie noch nicht mit einem echten Screenreader ausprobieren: Wenn etwas schlecht vorgelesen wird, sag mir was, und ich korrigiere es. Die Schaltfläche „Vorlesen“ oben braucht keinen Screenreader.",
            "Az alkalmazás leírja a tartalmát a rendszernek (az AccessKit révén), így az NVDA, JAWS, Narrator, VoiceOver és Orca fel tudja olvasni a gombokat és szövegeket. Még nem tudtam valódi képernyőolvasóval kipróbálni: ha valami rosszul olvasódik fel, mondd meg, mi, és javítom. A fenti „Felolvasás” gombhoz nem kell képernyőolvasó.",
        ],
    ),
    (
        "System help: Windows Narrator Win+Ctrl+Enter, Magnifier Win+Plus; macOS VoiceOver Cmd+F5, Zoom Option+Cmd+8; Linux Orca Super+Alt+S.",
        [
            "Aide du système : Windows Narrator Win+Ctrl+Entrée, Loupe Win+Plus ; macOS VoiceOver Cmd+F5, Zoom Option+Cmd+8 ; Linux Orca Super+Alt+S.",
            "Aiuti del sistema: Windows Narrator Win+Ctrl+Invio, Lente Win+Più; macOS VoiceOver Cmd+F5, Zoom Opzione+Cmd+8; Linux Orca Super+Alt+S.",
            "Ayuda del sistema: Windows Narrador Win+Ctrl+Intro, Lupa Win+Más; macOS VoiceOver Cmd+F5, Zoom Opción+Cmd+8; Linux Orca Super+Alt+S.",
            "Ajuda do sistema: Windows Narrador Win+Ctrl+Enter, Lupa Win+Mais; macOS VoiceOver Cmd+F5, Zoom Opção+Cmd+8; Linux Orca Super+Alt+S.",
            "Hilfe des Systems: Windows Sprachausgabe Win+Strg+Enter, Lupe Win+Plus; macOS VoiceOver Cmd+F5, Zoom Option+Cmd+8; Linux Orca Super+Alt+S.",
            "Rendszersúgó: Windows Narrátor Win+Ctrl+Enter, Nagyító Win+Plusz; macOS VoiceOver Cmd+F5, Nagyítás Option+Cmd+8; Linux Orca Super+Alt+S.",
        ],
    ),
    (
        "The server address is not valid: only http:// servers on this computer or your own network work here.",
        [
            "L'adresse du serveur n'est pas valide : seuls les serveurs http:// de cet ordinateur ou de votre réseau fonctionnent ici.",
            "L'indirizzo del server non è valido: qui funzionano solo server http:// di questo computer o della tua rete.",
            "La dirección del servidor no es válida: aquí solo funcionan servidores http:// de este ordenador o de tu red.",
            "O endereço do servidor não é válido: aqui só funcionam servidores http:// deste computador ou da sua rede.",
            "Die Serveradresse ist ungültig: Hier funktionieren nur http://-Server auf diesem Computer oder in deinem Netz.",
            "A szervercím érvénytelen: itt csak ezen a számítógépen vagy a saját hálózaton lévő http:// szerverek működnek.",
        ],
    ),
    (
        "Automations",
        [
            "Automatisations",
            "Automazioni",
            "Automatizaciones",
            "Automações",
            "Automatisierungen",
            "Automatizálások",
        ],
    ),
    (
        "Read a folder",
        [
            "Lire un dossier",
            "Leggi una cartella",
            "Leer una carpeta",
            "Ler uma pasta",
            "Ordner lesen",
            "Mappa olvasása",
        ],
    ),
    (
        "Read a file",
        [
            "Lire un fichier",
            "Leggi un file",
            "Leer un archivo",
            "Ler um arquivo",
            "Datei lesen",
            "Fájl olvasása",
        ],
    ),
    (
        "Read from a repository",
        [
            "Lire depuis un dépôt",
            "Leggi da un repository",
            "Leer de un repositorio",
            "Ler de um repositório",
            "Aus einem Repository lesen",
            "Olvasás tárolóból",
        ],
    ),
    (
        "Ask the model",
        [
            "Interroger le modèle",
            "Chiedi al modello",
            "Preguntar al modelo",
            "Perguntar ao modelo",
            "Das Modell fragen",
            "Kérdezd a modellt",
        ],
    ),
    (
        "Save as a file",
        [
            "Enregistrer en fichier",
            "Salva come file",
            "Guardar como archivo",
            "Salvar como arquivo",
            "Als Datei speichern",
            "Mentés fájlként",
        ],
    ),
    (
        "Send to a webhook (with your consent)",
        [
            "Envoyer à un webhook (avec votre accord)",
            "Invia a un webhook (con il tuo consenso)",
            "Enviar a un webhook (con tu consentimiento)",
            "Enviar a um webhook (com o seu consentimento)",
            "An einen Webhook senden (mit deiner Zustimmung)",
            "Küldés webhookra (a te beleegyezéseddel)",
        ],
    ),
    (
        "Show the result",
        [
            "Afficher le résultat",
            "Mostra il risultato",
            "Mostrar el resultado",
            "Mostrar o resultado",
            "Ergebnis anzeigen",
            "Az eredmény megjelenítése",
        ],
    ),
    (
        "The automation finished.",
        [
            "L'automatisation est terminée.",
            "L'automazione è terminata.",
            "La automatización terminó.",
            "A automação terminou.",
            "Die Automatisierung ist fertig.",
            "Az automatizálás befejeződött.",
        ],
    ),
    (
        "The automation stopped; see the journal.",
        [
            "L'automatisation s'est arrêtée ; voir le journal.",
            "L'automazione si è fermata; vedi il registro.",
            "La automatización se detuvo; mira el registro.",
            "A automação parou; veja o registro.",
            "Die Automatisierung wurde gestoppt; siehe Protokoll.",
            "Az automatizálás leállt; nézd meg a naplót.",
        ],
    ),
    ("New", ["Nouveau", "Nuovo", "Nuevo", "Novo", "Neu", "Új"]),
    (
        "Add the starter templates",
        [
            "Ajouter les modèles de départ",
            "Aggiungi i modelli iniziali",
            "Añadir las plantillas iniciales",
            "Adicionar os modelos iniciais",
            "Startvorlagen hinzufügen",
            "Kezdősablonok hozzáadása",
        ],
    ),
    (
        "A short chain of steps that you start, or that runs by itself while the app is open: read a page or a folder, ask the model, save or send. At most 12 steps and 20 model calls; a failed model call is retried 3 times. Text brought in from outside reaches the model only marked as material, not as instructions. Sending always needs your consent and is never done without you. It runs only while the app is open and the model is running.",
        [
            "Une courte chaîne d'étapes que vous lancez, ou qui s'exécute seule tant que l'application est ouverte : lire une page ou un dossier, interroger le modèle, enregistrer ou envoyer. Au plus 12 étapes et 20 appels au modèle ; un appel échoué est retenté 3 fois. Le texte venu de l'extérieur n'atteint le modèle que marqué comme matière, pas comme instructions. L'envoi demande toujours votre accord et ne se fait jamais sans vous. Cela ne fonctionne que si l'application est ouverte et le modèle démarré.",
            "Una breve catena di passi che avvii tu, o che gira da sola finché l'app è aperta: leggere una pagina o una cartella, chiedere al modello, salvare o inviare. Al massimo 12 passi e 20 chiamate al modello; una chiamata fallita viene ritentata 3 volte. Il testo che arriva da fuori raggiunge il modello solo contrassegnato come materiale, non come istruzioni. L'invio chiede sempre il tuo consenso e non avviene mai senza di te. Funziona solo con l'app aperta e il modello avviato.",
            "Una cadena corta de pasos que inicias tú, o que corre sola mientras la aplicación está abierta: leer una página o una carpeta, preguntar al modelo, guardar o enviar. Como máximo 12 pasos y 20 llamadas al modelo; una llamada fallida se reintenta 3 veces. El texto que llega de fuera solo llega al modelo marcado como material, no como instrucciones. Enviar siempre pide tu consentimiento y nunca se hace sin ti. Solo funciona con la aplicación abierta y el modelo en marcha.",
            "Uma cadeia curta de passos que você inicia, ou que roda sozinha enquanto o aplicativo está aberto: ler uma página ou uma pasta, perguntar ao modelo, salvar ou enviar. No máximo 12 passos e 20 chamadas ao modelo; uma chamada que falha é repetida 3 vezes. O texto vindo de fora só chega ao modelo marcado como material, não como instruções. Enviar sempre pede o seu consentimento e nunca é feito sem você. Só funciona com o aplicativo aberto e o modelo em execução.",
            "Eine kurze Kette von Schritten, die du startest oder die von selbst läuft, solange die App offen ist: eine Seite oder einen Ordner lesen, das Modell fragen, speichern oder senden. Höchstens 12 Schritte und 20 Modellaufrufe; ein fehlgeschlagener Aufruf wird 3-mal wiederholt. Text von außen erreicht das Modell nur als Material markiert, nicht als Anweisung. Senden braucht immer deine Zustimmung und geschieht nie ohne dich. Läuft nur, solange die App offen und das Modell gestartet ist.",
            "Lépések rövid láncolata, amelyet te indítasz, vagy amely magától fut, amíg az alkalmazás nyitva van: oldal vagy mappa olvasása, a modell megkérdezése, mentés vagy küldés. Legfeljebb 12 lépés és 20 modellhívás; a sikertelen hívást 3-szor újrapróbálja. A kívülről jött szöveg csak anyagként megjelölve jut el a modellhez, nem utasításként. A küldés mindig a beleegyezésedet kéri, és sosem történik nálad nélkül. Csak akkor fut, ha az alkalmazás nyitva van és a modell fut.",
        ],
    ),
    (
        "New automation",
        [
            "Nouvelle automatisation",
            "Nuova automazione",
            "Nueva automatización",
            "Nova automação",
            "Neue Automatisierung",
            "Új automatizálás",
        ],
    ),
    (
        "Pick an automation on the left, or create a new one.",
        [
            "Choisissez une automatisation à gauche, ou créez-en une nouvelle.",
            "Scegli un'automazione a sinistra, o creane una nuova.",
            "Elige una automatización a la izquierda, o crea una nueva.",
            "Escolha uma automação à esquerda, ou crie uma nova.",
            "Wähle links eine Automatisierung oder erstelle eine neue.",
            "Válassz egy automatizálást balra, vagy hozz létre újat.",
        ],
    ),
    ("Name", ["Nom", "Nome", "Nombre", "Nome", "Name", "Név"]),
    (
        "Start",
        ["Démarrage", "Avvio", "Inicio", "Início", "Start", "Indítás"],
    ),
    (
        "On (runs by itself)",
        [
            "Activée (s'exécute seule)",
            "Attiva (gira da sola)",
            "Activa (corre sola)",
            "Ativa (roda sozinha)",
            "Ein (läuft von selbst)",
            "Bekapcsolva (magától fut)",
        ],
    ),
    (
        "What it gets (address, folder, file or text):",
        [
            "Ce qu'elle reçoit (adresse, dossier, fichier ou texte) :",
            "Cosa riceve (indirizzo, cartella, file o testo):",
            "Lo que recibe (dirección, carpeta, archivo o texto):",
            "O que recebe (endereço, pasta, arquivo ou texto):",
            "Was sie bekommt (Adresse, Ordner, Datei oder Text):",
            "Amit kap (cím, mappa, fájl vagy szöveg):",
        ],
    ),
    (
        "+ Step",
        [
            "+ Étape",
            "+ Passo",
            "+ Paso",
            "+ Passo",
            "+ Schritt",
            "+ Lépés",
        ],
    ),
    (
        "Run now",
        [
            "Lancer maintenant",
            "Esegui ora",
            "Ejecutar ahora",
            "Executar agora",
            "Jetzt ausführen",
            "Futtatás most",
        ],
    ),
    (
        "Delete the automation",
        [
            "Supprimer l'automatisation",
            "Elimina l'automazione",
            "Eliminar la automatización",
            "Excluir a automação",
            "Automatisierung löschen",
            "Az automatizálás törlése",
        ],
    ),
    (
        "Manual only",
        [
            "Manuel seulement",
            "Solo manuale",
            "Solo manual",
            "Somente manual",
            "Nur manuell",
            "Csak kézzel",
        ],
    ),
    (
        "When the app starts",
        [
            "Au démarrage de l'application",
            "All'avvio dell'app",
            "Al iniciar la aplicación",
            "Ao iniciar o aplicativo",
            "Beim Start der App",
            "Az alkalmazás indulásakor",
        ],
    ),
    (
        "Every…",
        [
            "Toutes les…",
            "Ogni…",
            "Cada…",
            "A cada…",
            "Alle…",
            "Minden…",
        ],
    ),
    (
        "The “Send” step never runs without you: a timed run stops at it.",
        [
            "L'étape « Envoyer » ne s'exécute jamais sans vous : une exécution programmée s'arrête là.",
            "Il passo «Invia» non gira mai senza di te: un'esecuzione a tempo si ferma lì.",
            "El paso «Enviar» nunca corre sin ti: una ejecución programada se detiene en él.",
            "O passo «Enviar» nunca roda sem você: uma execução agendada para nele.",
            "Der Schritt „Senden“ läuft nie ohne dich: Ein zeitgesteuerter Lauf stoppt dort.",
            "A „Küldés” lépés sosem fut nálad nélkül: az időzített futás itt megáll.",
        ],
    ),
    (
        "Start the model first.",
        [
            "Démarrez d'abord le modèle.",
            "Avvia prima il modello.",
            "Inicia primero el modelo.",
            "Inicie primeiro o modelo.",
            "Starte zuerst das Modell.",
            "Előbb indítsd el a modellt.",
        ],
    ),
    (
        "Needs your consent",
        [
            "Demande votre accord",
            "Richiede il tuo consenso",
            "Necesita tu consentimiento",
            "Precisa do seu consentimento",
            "Braucht deine Zustimmung",
            "A beleegyezésedre van szükség",
        ],
    ),
    (
        "Yes, send",
        [
            "Oui, envoyer",
            "Sì, invia",
            "Sí, enviar",
            "Sim, enviar",
            "Ja, senden",
            "Igen, küldés",
        ],
    ),
    ("No", ["Non", "No", "No", "Não", "Nein", "Nem"]),
    (
        "Journal",
        [
            "Journal",
            "Registro",
            "Registro",
            "Registro",
            "Protokoll",
            "Napló",
        ],
    ),
    (
        "Result",
        [
            "Résultat",
            "Risultato",
            "Resultado",
            "Resultado",
            "Ergebnis",
            "Eredmény",
        ],
    ),
    (
        "Saved files are in",
        [
            "Les fichiers enregistrés sont dans",
            "I file salvati sono in",
            "Los archivos guardados están en",
            "Os arquivos salvos estão em",
            "Gespeicherte Dateien liegen in",
            "A mentett fájlok itt vannak:",
        ],
    ),
    (
        "Remember",
        [
            "Retenir",
            "Ricorda",
            "Recordar",
            "Lembrar",
            "Merken",
            "Megjegyzés",
        ],
    ),
    (
        "Memory and constellation",
        [
            "Mémoire et constellation",
            "Memoria e costellazione",
            "Memoria y constelación",
            "Memória e constelação",
            "Gedächtnis und Konstellation",
            "Memória és csillagkép",
        ],
    ),
    (
        "Remembered. I will use it in later conversations.",
        [
            "Retenu. Je l'utiliserai dans les conversations suivantes.",
            "Ricordato. Lo userò nelle conversazioni successive.",
            "Recordado. Lo usaré en las conversaciones siguientes.",
            "Lembrado. Vou usá-lo nas próximas conversas.",
            "Gemerkt. Ich nutze es in späteren Unterhaltungen.",
            "Megjegyeztem. A későbbi beszélgetésekben használni fogom.",
        ],
    ),
    (
        "Already remembered.",
        [
            "Déjà retenu.",
            "Già ricordato.",
            "Ya estaba recordado.",
            "Já estava lembrado.",
            "Schon gemerkt.",
            "Már megjegyezve.",
        ],
    ),
    (
        "Everything you have said stays on this computer. With each question I look through all conversations for the closest fragments and give them to the model as background, not as instructions. In very long conversations I keep the recent messages whole and a summary of the old ones. The constellation shows how conversations relate.",
        [
            "Tout ce que vous avez dit reste sur cet ordinateur. À chaque question, je cherche dans toutes les conversations les fragments les plus proches et je les donne au modèle comme contexte, pas comme instructions. Dans les très longues conversations, je garde les messages récents en entier et un résumé des anciens. La constellation montre comment les conversations sont liées.",
            "Tutto ciò che hai detto resta su questo computer. A ogni domanda cerco in tutte le conversazioni i frammenti più vicini e li do al modello come sfondo, non come istruzioni. Nelle conversazioni molto lunghe tengo interi i messaggi recenti e un riassunto di quelli vecchi. La costellazione mostra come si collegano le conversazioni.",
            "Todo lo que has dicho se queda en este ordenador. En cada pregunta busco en todas las conversaciones los fragmentos más cercanos y se los doy al modelo como contexto, no como instrucciones. En conversaciones muy largas mantengo enteros los mensajes recientes y un resumen de los antiguos. La constelación muestra cómo se relacionan las conversaciones.",
            "Tudo o que você disse fica neste computador. A cada pergunta procuro em todas as conversas os trechos mais próximos e os entrego ao modelo como contexto, não como instruções. Em conversas muito longas mantenho inteiras as mensagens recentes e um resumo das antigas. A constelação mostra como as conversas se relacionam.",
            "Alles, was du gesagt hast, bleibt auf diesem Computer. Bei jeder Frage suche ich in allen Unterhaltungen die ähnlichsten Abschnitte und gebe sie dem Modell als Hintergrund, nicht als Anweisung. In sehr langen Unterhaltungen behalte ich die neuen Nachrichten vollständig und eine Zusammenfassung der alten. Die Konstellation zeigt, wie die Unterhaltungen zusammenhängen.",
            "Minden, amit mondtál, ezen a számítógépen marad. Minden kérdésnél átkeresem az összes beszélgetést a legközelebbi részletekért, és háttérként adom a modellnek, nem utasításként. Nagyon hosszú beszélgetésekben a friss üzeneteket egészben tartom, a régiekről pedig összefoglalót. A csillagkép megmutatja, hogyan kapcsolódnak a beszélgetések.",
        ],
    ),
    (
        "Use memory (pinned notes and fragments from other conversations)",
        [
            "Utiliser la mémoire (notes épinglées et fragments d'autres conversations)",
            "Usa la memoria (note fissate e frammenti di altre conversazioni)",
            "Usar la memoria (notas fijadas y fragmentos de otras conversaciones)",
            "Usar a memória (notas fixadas e trechos de outras conversas)",
            "Gedächtnis nutzen (angeheftete Notizen und Abschnitte aus anderen Unterhaltungen)",
            "Memória használata (rögzített jegyzetek és más beszélgetések részletei)",
        ],
    ),
    (
        "Constellation",
        [
            "Constellation",
            "Costellazione",
            "Constelación",
            "Constelação",
            "Konstellation",
            "Csillagkép",
        ],
    ),
    (
        "Pinned notes",
        [
            "Notes épinglées",
            "Note fissate",
            "Notas fijadas",
            "Notas fixadas",
            "Angeheftete Notizen",
            "Rögzített jegyzetek",
        ],
    ),
    (
        "Building the constellation…",
        [
            "Construction de la constellation…",
            "Costruzione della costellazione…",
            "Construyendo la constelación…",
            "Construindo a constelação…",
            "Konstellation wird erstellt…",
            "A csillagkép készül…",
        ],
    ),
    (
        "No conversations yet. After the first one it appears here.",
        [
            "Pas encore de conversations. Après la première, elle apparaît ici.",
            "Ancora nessuna conversazione. Dopo la prima, appare qui.",
            "Aún no hay conversaciones. Tras la primera, aparece aquí.",
            "Ainda não há conversas. Depois da primeira, ela aparece aqui.",
            "Noch keine Unterhaltungen. Nach der ersten erscheint sie hier.",
            "Még nincs beszélgetés. Az első után itt megjelenik.",
        ],
    ),
    (
        "Each star is a conversation, each line a resemblance in content, the colour is the group it belongs to. Press a star, or pick a conversation from the list, to see its details.",
        [
            "Chaque étoile est une conversation, chaque ligne une ressemblance de contenu, la couleur est le groupe auquel elle appartient. Appuyez sur une étoile, ou choisissez une conversation dans la liste, pour voir les détails.",
            "Ogni stella è una conversazione, ogni linea una somiglianza di contenuto, il colore è il gruppo a cui appartiene. Premi una stella, o scegli una conversazione dall'elenco, per vedere i dettagli.",
            "Cada estrella es una conversación, cada línea una semejanza de contenido, el color es el grupo al que pertenece. Pulsa una estrella, o elige una conversación de la lista, para ver los detalles.",
            "Cada estrela é uma conversa, cada linha uma semelhança de conteúdo, a cor é o grupo a que pertence. Toque numa estrela, ou escolha uma conversa da lista, para ver os detalhes.",
            "Jeder Stern ist eine Unterhaltung, jede Linie eine inhaltliche Ähnlichkeit, die Farbe die Gruppe, zu der sie gehört. Tippe auf einen Stern oder wähle eine Unterhaltung aus der Liste, um Details zu sehen.",
            "Minden csillag egy beszélgetés, minden vonal tartalmi hasonlóság, a szín a csoport, amelyhez tartozik. Nyomj egy csillagra, vagy válassz egy beszélgetést a listából a részletekért.",
        ],
    ),
    (
        "— pick —",
        [
            "— choisir —",
            "— scegli —",
            "— elegir —",
            "— escolher —",
            "— wählen —",
            "— válassz —",
        ],
    ),
    (
        "Zoom in",
        [
            "Agrandir",
            "Ingrandisci",
            "Acercar",
            "Aproximar",
            "Vergrößern",
            "Nagyítás",
        ],
    ),
    (
        "Zoom out",
        [
            "Réduire",
            "Riduci",
            "Alejar",
            "Afastar",
            "Verkleinern",
            "Kicsinyítés",
        ],
    ),
    (
        "Reset zoom",
        [
            "Réinitialiser le zoom",
            "Reimposta lo zoom",
            "Restablecer el zoom",
            "Redefinir o zoom",
            "Zoom zurücksetzen",
            "Nagyítás visszaállítása",
        ],
    ),
    (
        "messages",
        [
            "messages",
            "messaggi",
            "mensajes",
            "mensagens",
            "Nachrichten",
            "üzenet",
        ],
    ),
    (
        "Group",
        ["Groupe", "Gruppo", "Grupo", "Grupo", "Gruppe", "Csoport"],
    ),
    (
        "Similar conversations:",
        [
            "Conversations similaires :",
            "Conversazioni simili:",
            "Conversaciones similares:",
            "Conversas semelhantes:",
            "Ähnliche Unterhaltungen:",
            "Hasonló beszélgetések:",
        ],
    ),
    (
        "Open the conversation",
        [
            "Ouvrir la conversation",
            "Apri la conversazione",
            "Abrir la conversación",
            "Abrir a conversa",
            "Unterhaltung öffnen",
            "A beszélgetés megnyitása",
        ],
    ),
    (
        "Pinned notes are given to the model in every conversation. Pin a message with the “Remember” button under it, or write here what you want it to always know (preferences, names, your own rules).",
        [
            "Les notes épinglées sont données au modèle dans chaque conversation. Épinglez un message avec le bouton « Retenir » sous lui, ou écrivez ici ce que vous voulez qu'il sache toujours (préférences, noms, vos propres règles).",
            "Le note fissate vengono date al modello in ogni conversazione. Fissa un messaggio con il pulsante «Ricorda» sotto di esso, o scrivi qui ciò che vuoi che sappia sempre (preferenze, nomi, tue regole).",
            "Las notas fijadas se dan al modelo en cada conversación. Fija un mensaje con el botón «Recordar» que tiene debajo, o escribe aquí lo que quieres que sepa siempre (preferencias, nombres, tus propias reglas).",
            "As notas fixadas são dadas ao modelo em cada conversa. Fixe uma mensagem com o botão «Lembrar» abaixo dela, ou escreva aqui o que quer que ele saiba sempre (preferências, nomes, suas próprias regras).",
            "Angeheftete Notizen bekommt das Modell in jeder Unterhaltung. Hefte eine Nachricht mit der Schaltfläche „Merken“ darunter an, oder schreibe hier, was es immer wissen soll (Vorlieben, Namen, eigene Regeln).",
            "A rögzített jegyzeteket a modell minden beszélgetésben megkapja. Rögzíts egy üzenetet az alatta lévő „Megjegyzés” gombbal, vagy írd ide, amit mindig tudnia kell (preferenciák, nevek, saját szabályaid).",
        ],
    ),
    (
        "A new note…",
        [
            "Une nouvelle note…",
            "Una nuova nota…",
            "Una nota nueva…",
            "Uma nova nota…",
            "Eine neue Notiz…",
            "Új jegyzet…",
        ],
    ),
    (
        "Add the note",
        [
            "Ajouter la note",
            "Aggiungi la nota",
            "Añadir la nota",
            "Adicionar a nota",
            "Notiz hinzufügen",
            "Jegyzet hozzáadása",
        ],
    ),
    (
        "Summary of the current conversation",
        [
            "Résumé de la conversation en cours",
            "Riassunto della conversazione corrente",
            "Resumen de la conversación actual",
            "Resumo da conversa atual",
            "Zusammenfassung der aktuellen Unterhaltung",
            "A jelenlegi beszélgetés összefoglalója",
        ],
    ),
    (
        "Not needed yet: the conversation fits whole in the model's window.",
        [
            "Pas encore nécessaire : la conversation tient entière dans la fenêtre du modèle.",
            "Non ancora necessario: la conversazione sta tutta nella finestra del modello.",
            "Aún no hace falta: la conversación cabe entera en la ventana del modelo.",
            "Ainda não é preciso: a conversa cabe inteira na janela do modelo.",
            "Noch nicht nötig: Die Unterhaltung passt vollständig ins Fenster des Modells.",
            "Még nem szükséges: a beszélgetés egészben elfér a modell ablakában.",
        ],
    ),
    (
        "Covers the first",
        [
            "Couvre les premiers",
            "Copre i primi",
            "Cubre los primeros",
            "Cobre as primeiras",
            "Deckt die ersten",
            "Az első",
        ],
    ),
    (
        "messages.",
        [
            "messages.",
            "messaggi.",
            "mensajes.",
            "mensagens.",
            "Nachrichten ab.",
            "üzenetet fedi le.",
        ],
    ),
    (
        "Delete the summary",
        [
            "Supprimer le résumé",
            "Elimina il riassunto",
            "Eliminar el resumen",
            "Excluir o resumo",
            "Zusammenfassung löschen",
            "Az összefoglaló törlése",
        ],
    ),
    (
        "What the last answer used from memory",
        [
            "Ce que la dernière réponse a pris dans la mémoire",
            "Cosa ha usato dalla memoria l'ultima risposta",
            "Lo que usó la última respuesta de la memoria",
            "O que a última resposta usou da memória",
            "Was die letzte Antwort aus dem Gedächtnis genutzt hat",
            "Mit használt az utolsó válasz a memóriából",
        ],
    ),
    (
        "Nothing yet.",
        [
            "Rien pour l'instant.",
            "Ancora niente.",
            "Nada todavía.",
            "Nada ainda.",
            "Noch nichts.",
            "Még semmi.",
        ],
    ),
    (
        "Search all conversations…",
        [
            "Chercher dans toutes les conversations…",
            "Cerca in tutte le conversazioni…",
            "Buscar en todas las conversaciones…",
            "Pesquisar em todas as conversas…",
            "Alle Unterhaltungen durchsuchen…",
            "Keresés az összes beszélgetésben…",
        ],
    ),
    (
        "Building the index…",
        [
            "Construction de l'index…",
            "Costruzione dell'indice…",
            "Construyendo el índice…",
            "Construindo o índice…",
            "Index wird erstellt…",
            "Az index készül…",
        ],
    ),
    (
        "Indexed:",
        [
            "Indexés :",
            "Indicizzati:",
            "Indexados:",
            "Indexados:",
            "Indiziert:",
            "Indexelve:",
        ],
    ),
    (
        "fragments.",
        [
            "fragments.",
            "frammenti.",
            "fragmentos.",
            "trechos.",
            "Abschnitte.",
            "részlet.",
        ],
    ),
    (
        "Nothing found.",
        [
            "Rien trouvé.",
            "Niente trovato.",
            "No se encontró nada.",
            "Nada encontrado.",
            "Nichts gefunden.",
            "Nincs találat.",
        ],
    ),
    (
        "25 languages; beats Whisper on accuracy at a quarter of the size and far faster.",
        [
            "25 langues ; plus précis que Whisper, pour un quart de la taille et bien plus rapide.",
            "25 lingue; più preciso di Whisper, con un quarto delle dimensioni e molto più veloce.",
            "25 idiomas; más preciso que Whisper, con una cuarta parte del tamaño y mucho más rápido.",
            "25 idiomas; mais preciso que o Whisper, com um quarto do tamanho e muito mais rápido.",
            "25 Sprachen; genauer als Whisper, bei einem Viertel der Größe und deutlich schneller.",
            "25 nyelv; pontosabb a Whispernél, negyed akkora és sokkal gyorsabb.",
        ],
    ),
    (
        "99 languages, punctuation included; the most widely used, now behind newer models on accuracy.",
        [
            "99 langues, ponctuation comprise ; le plus utilisé, désormais dépassé en précision par des modèles plus récents.",
            "99 lingue, punteggiatura inclusa; il più usato, ora superato in precisione da modelli più recenti.",
            "99 idiomas, con puntuación; el más usado, ahora superado en precisión por modelos más nuevos.",
            "99 idiomas, com pontuação; o mais usado, hoje superado em precisão por modelos mais novos.",
            "99 Sprachen, mit Zeichensetzung; das am weitesten verbreitete, bei der Genauigkeit inzwischen hinter neueren Modellen.",
            "99 nyelv, írásjelekkel; a legelterjedtebb, pontosságban már elmarad az újabb modellektől.",
        ],
    ),
    (
        "A bot you create sends you messages and alerts.",
        [
            "Un bot que vous créez vous envoie des messages et des alertes.",
            "Un bot che crei ti invia messaggi e avvisi.",
            "Un bot que creas te envía mensajes y avisos.",
            "Um bot que você cria envia-lhe mensagens e alertas.",
            "Ein Bot, den du erstellst, schickt dir Nachrichten und Warnungen.",
            "Egy általad létrehozott bot üzeneteket és figyelmeztetéseket küld neked.",
        ],
    ),
    (
        "A drop-in OpenAI-style server with text, speech, pictures and embeddings; can run without a graphics card.",
        [
            "Un serveur de type OpenAI prêt à l'emploi pour le texte, la voix, les images et les embeddings ; peut tourner sans carte graphique.",
            "Un server in stile OpenAI pronto all'uso con testo, voce, immagini ed embedding; può funzionare senza scheda grafica.",
            "Un servidor al estilo OpenAI listo para usar, con texto, voz, imágenes y embeddings; puede funcionar sin tarjeta gráfica.",
            "Um servidor ao estilo OpenAI, pronto a usar, com texto, voz, imagens e embeddings; pode funcionar sem placa gráfica.",
            "Ein sofort einsetzbarer Server im OpenAI-Stil für Text, Sprache, Bilder und Embeddings; läuft auch ohne Grafikkarte.",
            "OpenAI-stílusú, azonnal használható szerver szöveghez, beszédhez, képekhez és embeddingekhez; videokártya nélkül is futhat.",
        ],
    ),
    (
        "A private search engine you run yourself; the model searches the web through it.",
        [
            "Un moteur de recherche privé que vous hébergez vous-même ; le modèle cherche sur le web à travers lui.",
            "Un motore di ricerca privato che gestisci tu; il modello cerca sul web attraverso di esso.",
            "Un buscador privado que ejecutas tú mismo; el modelo busca en la web a través de él.",
            "Um motor de busca privado que você mesmo executa; o modelo pesquisa na web através dele.",
            "Eine private Suchmaschine, die du selbst betreibst; das Modell durchsucht darüber das Web.",
            "Saját magad által üzemeltetett privát keresőmotor; a modell ezen keresztül keres a weben.",
        ],
    ),
    (
        "A standard way for programs to offer tools to a model; colibri has its own.",
        [
            "Une manière standard pour des programmes de proposer des outils à un modèle ; colibri a la sienne.",
            "Un modo standard con cui i programmi offrono strumenti a un modello; colibri ha il suo.",
            "Una forma estándar para que los programas ofrezcan herramientas a un modelo; colibri tiene la suya.",
            "Uma forma padrão de os programas oferecerem ferramentas a um modelo; o colibri tem a sua.",
            "Ein Standard, mit dem Programme einem Modell Werkzeuge anbieten; colibri hat seinen eigenen.",
            "Szabványos mód arra, hogy programok eszközöket kínáljanak egy modellnek; a colibrinek saját van.",
        ],
    ),
    (
        "A train leaves at 14:10 and arrives at 17:45. How long is the trip?",
        [
            "Un train part à 14 h 10 et arrive à 17 h 45. Combien dure le trajet ?",
            "Un treno parte alle 14:10 e arriva alle 17:45. Quanto dura il viaggio?",
            "Un tren sale a las 14:10 y llega a las 17:45. ¿Cuánto dura el viaje?",
            "Um comboio parte às 14:10 e chega às 17:45. Quanto dura a viagem?",
            "Ein Zug fährt um 14:10 ab und kommt um 17:45 an. Wie lange dauert die Fahrt?",
            "Egy vonat 14:10-kor indul és 17:45-kor érkezik. Mennyi ideig tart az út?",
        ],
    ),
    (
        "Access refused: the token is missing, expired or not allowed.",
        [
            "Accès refusé : le jeton est absent, expiré ou sans droits.",
            "Accesso rifiutato: il token manca, è scaduto o non ha i permessi.",
            "Acceso denegado: el token falta, ha caducado o no tiene permiso.",
            "Acesso recusado: o token falta, expirou ou não tem permissão.",
            "Zugriff verweigert: Das Token fehlt, ist abgelaufen oder hat keine Berechtigung.",
            "Hozzáférés megtagadva: a token hiányzik, lejárt vagy nincs joga.",
        ],
    ),
    (
        "Act: it runs actions and chains of steps on a schedule",
        [
            "Agir : il exécute des actions et des enchaînements d'étapes selon un horaire",
            "Agire: esegue azioni e catene di passaggi secondo un orario",
            "Actuar: ejecuta acciones y cadenas de pasos según un horario",
            "Agir: executa ações e cadeias de passos num horário",
            "Handeln: führt Aktionen und Schrittfolgen nach Zeitplan aus",
            "Cselekvés: műveleteket és lépéssorozatokat futtat ütemezés szerint",
        ],
    ),
    (
        "Address refused: {}",
        [
            "Adresse refusée : {}",
            "Indirizzo rifiutato: {}",
            "Dirección rechazada: {}",
            "Endereço recusado: {}",
            "Adresse abgelehnt: {}",
            "A cím elutasítva: {}",
        ],
    ),
    (
        "Analysis",
        [
            "Analyse",
            "Analisi",
            "Análisis",
            "Análise",
            "Analyse",
            "Elemzés",
        ],
    ),
    (
        "Ask me up to five questions, one at a time, to help me decide [DECISION]. Then give your honest recommendation and the main reason.",
        [
            "Pose-moi jusqu'à cinq questions, une à la fois, pour m'aider à décider [DÉCISION]. Donne ensuite ta recommandation honnête et la raison principale.",
            "Fammi fino a cinque domande, una alla volta, per aiutarmi a decidere [DECISIONE]. Poi dammi il tuo consiglio sincero e il motivo principale.",
            "Hazme hasta cinco preguntas, una a la vez, para ayudarme a decidir [DECISIÓN]. Luego dame tu recomendación sincera y el motivo principal.",
            "Faz-me até cinco perguntas, uma de cada vez, para me ajudar a decidir [DECISÃO]. Depois dá a tua recomendação sincera e o motivo principal.",
            "Stelle mir bis zu fünf Fragen, eine nach der anderen, damit ich [ENTSCHEIDUNG] treffen kann. Gib dann deine ehrliche Empfehlung und den wichtigsten Grund.",
            "Tegyél fel legfeljebb öt kérdést, egyenként, hogy eldönthessem: [DÖNTÉS]. Utána add meg az őszinte javaslatodat és a fő okot.",
        ],
    ),
    (
        "Ask questions about your tables; the model writes the queries you approve.",
        [
            "Posez des questions sur vos tables ; le modèle écrit les requêtes que vous validez.",
            "Fai domande sulle tue tabelle; il modello scrive le query che approvi.",
            "Haz preguntas sobre tus tablas; el modelo escribe las consultas que tú apruebas.",
            "Faça perguntas sobre as suas tabelas; o modelo escreve as consultas que você aprova.",
            "Stelle Fragen zu deinen Tabellen; das Modell schreibt die Abfragen, die du freigibst.",
            "Kérdezz a táblázataidról; a modell megírja a lekérdezéseket, te jóváhagyod őket.",
        ],
    ),
    (
        "Attach a whole notes folder; the model reads and cites your notes.",
        [
            "Joignez un dossier de notes entier ; le modèle lit et cite vos notes.",
            "Allega un'intera cartella di appunti; il modello legge e cita i tuoi appunti.",
            "Adjunta una carpeta entera de notas; el modelo lee y cita tus notas.",
            "Anexe uma pasta inteira de notas; o modelo lê e cita as suas notas.",
            "Hänge einen ganzen Notizenordner an; das Modell liest und zitiert deine Notizen.",
            "Csatolj egy egész jegyzetmappát; a modell elolvassa és idézi a jegyzeteidet.",
        ],
    ),
    (
        "Best at readable text in pictures, infographics and posters; version 2.0 is 7B and makes 2K images.",
        [
            "Excellent pour le texte lisible dans les images, les infographies et les affiches ; la version 2.0 fait 7B et produit des images 2K.",
            "Il migliore per testo leggibile nelle immagini, infografiche e poster; la versione 2.0 è da 7B e crea immagini 2K.",
            "El mejor para texto legible en imágenes, infografías y carteles; la versión 2.0 es de 7B y crea imágenes 2K.",
            "O melhor para texto legível em imagens, infográficos e cartazes; a versão 2.0 tem 7B e cria imagens 2K.",
            "Am besten für lesbaren Text in Bildern, Infografiken und Plakaten; Version 2.0 hat 7B und erzeugt 2K-Bilder.",
            "A legjobb olvasható szöveghez képeken, infografikákon és plakátokon; a 2.0-s verzió 7B és 2K-s képeket készít.",
        ],
    ),
    (
        "Best open model on the conversational-speech board in October 2026 (Small).",
        [
            "Meilleur modèle ouvert du classement de la parole conversationnelle en octobre 2026 (Small).",
            "Miglior modello aperto nella classifica del parlato conversazionale a ottobre 2026 (Small).",
            "El mejor modelo abierto de la clasificación de habla conversacional en octubre de 2026 (Small).",
            "O melhor modelo aberto do ranking de fala conversacional em outubro de 2026 (Small).",
            "Bestes offenes Modell der Rangliste für Gesprächssprache im Oktober 2026 (Small).",
            "A legjobb nyílt modell a beszélgetési beszéd ranglistáján 2026 októberében (Small).",
        ],
    ),
    (
        "Best-sounding local voice with cloning from a few seconds of audio; adds an inaudible watermark.",
        [
            "La voix locale la plus naturelle, avec clonage à partir de quelques secondes d'audio ; ajoute un filigrane inaudible.",
            "La voce locale dal suono migliore, con clonazione da pochi secondi di audio; aggiunge una filigrana inudibile.",
            "La voz local que mejor suena, con clonación a partir de unos segundos de audio; añade una marca de agua inaudible.",
            "A voz local com melhor som, com clonagem a partir de alguns segundos de áudio; adiciona uma marca de água inaudível.",
            "Die am besten klingende lokale Stimme, mit Klonen aus wenigen Sekunden Audio; fügt ein unhörbares Wasserzeichen hinzu.",
            "A legjobban hangzó helyi hang, néhány másodperces hangból klónozással; hallhatatlan vízjelet ad hozzá.",
        ],
    ),
    (
        "Brainstorm [NUMBER] ideas for [GOAL]. First the obvious ones, then the odd ones, then three that combine two ideas.",
        [
            "Propose [NOMBRE] idées pour [OBJECTIF]. D'abord les évidentes, puis les insolites, puis trois qui combinent deux idées.",
            "Proponi [NUMERO] idee per [OBIETTIVO]. Prima quelle ovvie, poi quelle strane, poi tre che combinano due idee.",
            "Propón [NÚMERO] ideas para [OBJETIVO]. Primero las obvias, luego las raras y después tres que combinen dos ideas.",
            "Propõe [NÚMERO] ideias para [OBJETIVO]. Primeiro as óbvias, depois as estranhas e depois três que combinem duas ideias.",
            "Sammle [ANZAHL] Ideen für [ZIEL]. Zuerst die naheliegenden, dann die ungewöhnlichen, dann drei, die zwei Ideen verbinden.",
            "Találj ki [SZÁM] ötletet ehhez: [CÉL]. Először az egyértelműeket, aztán a szokatlanokat, végül hármat, amely két ötletet ötvöz.",
        ],
    ),
    (
        "Break [PROJECT] into tasks of at most half a day each, in order, with what blocks what, and the first three things I can do today.",
        [
            "Découpe [PROJET] en tâches d'une demi-journée au plus, dans l'ordre, avec ce qui dépend de quoi, et les trois premières choses que je peux faire aujourd'hui.",
            "Suddividi [PROGETTO] in compiti di al massimo mezza giornata ciascuno, in ordine, con le dipendenze, e le prime tre cose che posso fare oggi.",
            "Divide [PROYECTO] en tareas de como máximo media jornada cada una, en orden, con lo que bloquea a qué, y las tres primeras cosas que puedo hacer hoy.",
            "Divide [PROJETO] em tarefas de no máximo meio dia cada, por ordem, com o que bloqueia o quê, e as três primeiras coisas que posso fazer hoje.",
            "Zerlege [PROJEKT] in Aufgaben von höchstens einem halben Tag, in Reihenfolge, mit Abhängigkeiten, und nenne die ersten drei Dinge, die ich heute tun kann.",
            "Bontsd fel a(z) [PROJEKT] feladatait legfeljebb fél napos részekre, sorrendben, a függőségekkel, és add meg az első három dolgot, amit ma megtehetek.",
        ],
    ),
    (
        "Break into tasks",
        [
            "Découper en tâches",
            "Dividere in compiti",
            "Dividir en tareas",
            "Dividir em tarefas",
            "In Aufgaben zerlegen",
            "Feladatokra bontás",
        ],
    ),
    (
        "Browse your repositories, read files and open issues into the chat.",
        [
            "Parcourez vos dépôts, lisez des fichiers et amenez des tickets (issues) dans le chat.",
            "Sfoglia i tuoi repository, leggi file e porta le issue nella chat.",
            "Explora tus repositorios, lee archivos y trae incidencias (issues) al chat.",
            "Navegue pelos seus repositórios, leia ficheiros e traga issues para o chat.",
            "Durchstöbere deine Repositories, lies Dateien und hole Issues in den Chat.",
            "Böngészd a tárolóidat, olvass fájlokat, és hozd be a hibajegyeket (issue-kat) a csevegésbe.",
        ],
    ),
    (
        "Check my Romanian",
        [
            "Vérifier mon roumain",
            "Controlla il mio rumeno",
            "Revisar mi rumano",
            "Verificar o meu romeno",
            "Mein Rumänisch prüfen",
            "Ellenőrizd a románomat",
        ],
    ),
    (
        "Check this Romanian text for diacritics, agreement, punctuation and the usual traps (s-au/sau, i-a/ia, într-o/Intr-o). List each correction with the rule.

[TEXT]",
        [
            "Vérifie ce texte roumain : diacritiques, accords, ponctuation et pièges habituels (s-au/sau, i-a/ia, într-o/Intr-o). Liste chaque correction avec la règle.

[TEXTE]",
            "Controlla questo testo rumeno: segni diacritici, concordanze, punteggiatura e le solite trappole (s-au/sau, i-a/ia, într-o/Intr-o). Elenca ogni correzione con la regola.

[TESTO]",
            "Revisa este texto en rumano: diacríticos, concordancia, puntuación y las trampas habituales (s-au/sau, i-a/ia, într-o/Intr-o). Enumera cada corrección con su regla.

[TEXTO]",
            "Revê este texto em romeno: diacríticos, concordância, pontuação e as armadilhas habituais (s-au/sau, i-a/ia, într-o/Intr-o). Lista cada correção com a regra.

[TEXTO]",
            "Prüfe diesen rumänischen Text auf Diakritika, Kongruenz, Zeichensetzung und die üblichen Fallen (s-au/sau, i-a/ia, într-o/Intr-o). Liste jede Korrektur mit der Regel auf.

[TEXT]",
            "Ellenőrizd ezt a román szöveget: diakritikus jelek, egyeztetés, központozás és a szokásos buktatók (s-au/sau, i-a/ia, într-o/Intr-o). Sorold fel az egyes javításokat a szabállyal.

[SZÖVEG]",
        ],
    ),
    (
        "Cinematic motion; one render of about 75 seconds on a single RTX 4090 reported.",
        [
            "Mouvement cinématographique ; un rendu d'environ 75 secondes signalé sur une seule RTX 4090.",
            "Movimento cinematografico; è segnalato un rendering di circa 75 secondi su una sola RTX 4090.",
            "Movimiento cinematográfico; se informa de un render de unos 75 segundos en una sola RTX 4090.",
            "Movimento cinematográfico; foi reportado um render de cerca de 75 segundos numa única RTX 4090.",
            "Kinoreife Bewegung; ein Render von etwa 75 Sekunden auf einer einzigen RTX 4090 wird berichtet.",
            "Filmszerű mozgás; egyetlen RTX 4090-en körülbelül 75 másodperces renderelésről számolnak be.",
        ],
    ),
    (
        "Compare [OPTION A] and [OPTION B] for [GOAL]. Use a table with the criteria that matter, then give a recommendation and say what would change it.",
        [
            "Compare [OPTION A] et [OPTION B] pour [OBJECTIF]. Utilise un tableau avec les critères qui comptent, puis donne une recommandation et dis ce qui la changerait.",
            "Confronta [OPZIONE A] e [OPZIONE B] per [OBIETTIVO]. Usa una tabella con i criteri che contano, poi dai una raccomandazione e dì cosa la cambierebbe.",
            "Compara [OPCIÓN A] y [OPCIÓN B] para [OBJETIVO]. Usa una tabla con los criterios que importan, luego da una recomendación y di qué la cambiaría.",
            "Compara [OPÇÃO A] e [OPÇÃO B] para [OBJETIVO]. Usa uma tabela com os critérios que importam, depois dá uma recomendação e diz o que a mudaria.",
            "Vergleiche [OPTION A] und [OPTION B] für [ZIEL]. Nutze eine Tabelle mit den wichtigen Kriterien, gib dann eine Empfehlung und sage, was sie ändern würde.",
            "Hasonlítsd össze a(z) [A OPCIÓ] és a(z) [B OPCIÓ] lehetőséget ehhez: [CÉL]. Használj táblázatot a fontos szempontokkal, majd adj javaslatot, és mondd meg, mi változtatná meg.",
        ],
    ),
    (
        "Compare options",
        [
            "Comparer des options",
            "Confrontare opzioni",
            "Comparar opciones",
            "Comparar opções",
            "Optionen vergleichen",
            "Lehetőségek összehasonlítása",
        ],
    ),
    (
        "Cook with what I have",
        [
            "Cuisiner avec ce que j'ai",
            "Cucinare con quello che ho",
            "Cocinar con lo que tengo",
            "Cozinhar com o que tenho",
            "Kochen mit dem, was ich habe",
            "Főzés abból, ami van",
        ],
    ),
    (
        "Could not reach the server: {}",
        [
            "Impossible d'atteindre le serveur : {}",
            "Impossibile raggiungere il server: {}",
            "No se pudo llegar al servidor: {}",
            "Não foi possível chegar ao servidor: {}",
            "Der Server ist nicht erreichbar: {}",
            "Nem sikerült elérni a szervert: {}",
        ],
    ),
    (
        "Critique my idea",
        [
            "Critiquer mon idée",
            "Criticare la mia idea",
            "Criticar mi idea",
            "Criticar a minha ideia",
            "Meine Idee kritisieren",
            "Bíráld az ötletemet",
        ],
    ),
    (
        "Daily brief (web page)",
        [
            "Résumé quotidien (page web)",
            "Riepilogo giornaliero (pagina web)",
            "Resumen diario (página web)",
            "Resumo diário (página web)",
            "Tagesüberblick (Webseite)",
            "Napi összefoglaló (weboldal)",
        ],
    ),
    (
        "Dialogue between several speakers, with laughs and pauses.",
        [
            "Dialogue entre plusieurs voix, avec rires et pauses.",
            "Dialogo tra più voci, con risate e pause.",
            "Diálogo entre varias voces, con risas y pausas.",
            "Diálogo entre várias vozes, com risos e pausas.",
            "Dialog zwischen mehreren Sprechern, mit Lachen und Pausen.",
            "Párbeszéd több beszélő között, nevetéssel és szünetekkel.",
        ],
    ),
    (
        "Digest of a folder",
        [
            "Résumé d'un dossier",
            "Riassunto di una cartella",
            "Resumen de una carpeta",
            "Resumo de uma pasta",
            "Zusammenfassung eines Ordners",
            "Egy mappa kivonata",
        ],
    ),
    (
        "Drop any folder: text, PDF, Office, HTML, images.",
        [
            "Déposez n'importe quel dossier : texte, PDF, Office, HTML, images.",
            "Trascina qualsiasi cartella: testo, PDF, Office, HTML, immagini.",
            "Suelta cualquier carpeta: texto, PDF, Office, HTML, imágenes.",
            "Largue qualquer pasta: texto, PDF, Office, HTML, imagens.",
            "Ziehe einen beliebigen Ordner hinein: Text, PDF, Office, HTML, Bilder.",
            "Húzz be bármilyen mappát: szöveg, PDF, Office, HTML, képek.",
        ],
    ),
    (
        "Everyday",
        [
            "Quotidien",
            "Quotidiano",
            "Cotidiano",
            "Dia a dia",
            "Alltag",
            "Mindennapok",
        ],
    ),
    (
        "Explain [TOPIC] as if I were [AGE] years old, with one everyday example and one question to check I understood.",
        [
            "Explique [SUJET] comme si j'avais [ÂGE] ans, avec un exemple du quotidien et une question pour vérifier que j'ai compris.",
            "Spiega [ARGOMENTO] come se avessi [ETÀ] anni, con un esempio quotidiano e una domanda per verificare che abbia capito.",
            "Explica [TEMA] como si tuviera [EDAD] años, con un ejemplo cotidiano y una pregunta para comprobar que lo entendí.",
            "Explica [TEMA] como se eu tivesse [IDADE] anos, com um exemplo do dia a dia e uma pergunta para verificar se percebi.",
            "Erkläre [THEMA], als wäre ich [ALTER] Jahre alt, mit einem Alltagsbeispiel und einer Frage, um zu prüfen, ob ich es verstanden habe.",
            "Magyarázd el a(z) [TÉMA] témát úgy, mintha [KOR] éves lennék, egy hétköznapi példával és egy kérdéssel, amellyel ellenőrizheted, hogy megértettem.",
        ],
    ),
    (
        "Explain briefly why the sky is blue.",
        [
            "Explique brièvement pourquoi le ciel est bleu.",
            "Spiega brevemente perché il cielo è blu.",
            "Explica brevemente por qué el cielo es azul.",
            "Explica brevemente por que o céu é azul.",
            "Erkläre kurz, warum der Himmel blau ist.",
            "Magyarázd el röviden, miért kék az ég.",
        ],
    ),
    (
        "Explain simply",
        [
            "Expliquer simplement",
            "Spiegare in modo semplice",
            "Explicar de forma sencilla",
            "Explicar de forma simples",
            "Einfach erklären",
            "Egyszerű magyarázat",
        ],
    ),
    (
        "Explain this code",
        [
            "Expliquer ce code",
            "Spiegare questo codice",
            "Explicar este código",
            "Explicar este código",
            "Diesen Code erklären",
            "Magyarázd el ezt a kódot",
        ],
    ),
    (
        "Explain what this [LANGUAGE] code does, step by step, for someone new to it. Point out bugs and risky parts.

[CODE]",
        [
            "Explique ce que fait ce code [LANGAGE], pas à pas, pour un débutant. Signale les bogues et les parties risquées.

[CODE]",
            "Spiega cosa fa questo codice [LINGUAGGIO], passo per passo, a chi è alle prime armi. Segnala bug e parti rischiose.

[CODICE]",
            "Explica qué hace este código en [LENGUAJE], paso a paso, para alguien que empieza. Señala errores y partes arriesgadas.

[CÓDIGO]",
            "Explica o que faz este código em [LINGUAGEM], passo a passo, para quem está a começar. Aponta erros e partes arriscadas.

[CÓDIGO]",
            "Erkläre Schritt für Schritt, was dieser [SPRACHE]-Code macht, für Einsteiger. Weise auf Fehler und riskante Stellen hin.

[CODE]",
            "Magyarázd el lépésről lépésre, mit csinál ez a [NYELV] kód, kezdőknek. Mutass rá a hibákra és a kockázatos részekre.

[KÓD]",
        ],
    ),
    (
        "Expressive speech in about nine languages (released March 2026).",
        [
            "Voix expressive dans une neuvaine de langues (sortie en mars 2026).",
            "Voce espressiva in circa nove lingue (rilasciato a marzo 2026).",
            "Voz expresiva en unos nueve idiomas (publicado en marzo de 2026).",
            "Voz expressiva em cerca de nove idiomas (lançado em março de 2026).",
            "Ausdrucksstarke Sprache in etwa neun Sprachen (veröffentlicht im März 2026).",
            "Kifejező beszéd mintegy kilenc nyelven (2026 márciusában jelent meg).",
        ],
    ),
    (
        "Fast, permissively licensed.",
        [
            "Rapide, sous licence permissive.",
            "Veloce, con licenza permissiva.",
            "Rápido, con licencia permisiva.",
            "Rápido, com licença permissiva.",
            "Schnell, mit freizügiger Lizenz.",
            "Gyors, megengedő licenccel.",
        ],
    ),
    (
        "Find the bug",
        [
            "Trouver le bogue",
            "Trovare il bug",
            "Encontrar el error",
            "Encontrar o erro",
            "Den Fehler finden",
            "A hiba megkeresése",
        ],
    ),
    (
        "Flashcards",
        [
            "Cartes mémoire",
            "Flashcard",
            "Tarjetas de estudio",
            "Cartões de estudo",
            "Lernkarten",
            "Tanulókártyák",
        ],
    ),
    (
        "Formal letter",
        [
            "Lettre formelle",
            "Lettera formale",
            "Carta formal",
            "Carta formal",
            "Formeller Brief",
            "Hivatalos levél",
        ],
    ),
    (
        "Free encyclopedia in every language; read articles by address.",
        [
            "Encyclopédie libre dans toutes les langues ; lisez des articles par adresse.",
            "Enciclopedia libera in ogni lingua; leggi gli articoli tramite indirizzo.",
            "Enciclopedia libre en todos los idiomas; lee artículos por dirección.",
            "Enciclopédia livre em todas as línguas; leia artigos pelo endereço.",
            "Freie Enzyklopädie in jeder Sprache; Artikel über die Adresse lesen.",
            "Szabad enciklopédia minden nyelven; cikkek olvasása cím alapján.",
        ],
    ),
    (
        "Free push notifications to your phone from any script or model.",
        [
            "Notifications push gratuites vers votre téléphone depuis n'importe quel script ou modèle.",
            "Notifiche push gratuite sul telefono da qualsiasi script o modello.",
            "Notificaciones push gratuitas a tu teléfono desde cualquier script o modelo.",
            "Notificações push gratuitas para o telemóvel a partir de qualquer script ou modelo.",
            "Kostenlose Push-Benachrichtigungen aufs Handy, von jedem Skript oder Modell.",
            "Ingyenes push-értesítések a telefonodra bármely szkriptből vagy modellből.",
        ],
    ),
    (
        "Free, non-profit hosting and self-hosted forges; read repositories and issues.",
        [
            "Hébergement gratuit à but non lucratif et forges auto-hébergées ; lisez dépôts et tickets.",
            "Hosting gratuito senza scopo di lucro e forge self-hosted; leggi repository e issue.",
            "Alojamiento gratuito sin ánimo de lucro y forjas autoalojadas; lee repositorios e incidencias.",
            "Alojamento gratuito sem fins lucrativos e forjas auto-hospedadas; leia repositórios e issues.",
            "Kostenloses, gemeinnütziges Hosting und selbst betriebene Forges; Repositories und Issues lesen.",
            "Ingyenes, nonprofit tárhely és saját üzemeltetésű kódforgók; tárolók és hibajegyek olvasása.",
        ],
    ),
    (
        "From the README, propose 5 concrete tasks in order of importance.",
        [
            "À partir du README, propose 5 tâches concrètes par ordre d'importance.",
            "Dal README, proponi 5 compiti concreti in ordine di importanza.",
            "A partir del README, propón 5 tareas concretas por orden de importancia.",
            "A partir do README, propõe 5 tarefas concretas por ordem de importância.",
            "Schlage anhand der README 5 konkrete Aufgaben nach Wichtigkeit vor.",
            "A README alapján javasolj 5 konkrét feladatot fontossági sorrendben.",
        ],
    ),
    (
        "Gift ideas",
        [
            "Idées de cadeaux",
            "Idee regalo",
            "Ideas de regalo",
            "Ideias de presentes",
            "Geschenkideen",
            "Ajándékötletek",
        ],
    ),
    (
        "Give me [NUMBER] idioms about [THEME] in [LANGUAGE], with a literal translation, the real meaning and an example sentence.",
        [
            "Donne-moi [NOMBRE] expressions idiomatiques sur [THÈME] en [LANGUE], avec une traduction littérale, le vrai sens et une phrase d'exemple.",
            "Dammi [NUMERO] modi di dire su [TEMA] in [LINGUA], con traduzione letterale, significato reale e una frase d'esempio.",
            "Dame [NÚMERO] modismos sobre [TEMA] en [IDIOMA], con traducción literal, significado real y una frase de ejemplo.",
            "Dá-me [NÚMERO] expressões idiomáticas sobre [TEMA] em [LÍNGUA], com tradução literal, o significado real e uma frase de exemplo.",
            "Gib mir [ANZAHL] Redewendungen zu [THEMA] auf [SPRACHE], mit wörtlicher Übersetzung, der eigentlichen Bedeutung und einem Beispielsatz.",
            "Adj [SZÁM] szólást a(z) [TÉMA] témáról [NYELV] nyelven, szó szerinti fordítással, valódi jelentéssel és egy példamondattal.",
        ],
    ),
    (
        "Give me ten different titles for [WORK]. Mix serious, playful and mysterious ones, and mark your favourite.",
        [
            "Donne-moi dix titres différents pour [ŒUVRE]. Mélange sérieux, ludiques et mystérieux, et marque ton préféré.",
            "Dammi dieci titoli diversi per [OPERA]. Mescola seri, giocosi e misteriosi, e segna il tuo preferito.",
            "Dame diez títulos distintos para [OBRA]. Mezcla serios, juguetones y misteriosos, y marca tu favorito.",
            "Dá-me dez títulos diferentes para [OBRA]. Mistura sérios, divertidos e misteriosos, e marca o teu preferido.",
            "Gib mir zehn verschiedene Titel für [WERK]. Mische ernste, spielerische und geheimnisvolle und markiere deinen Favoriten.",
            "Adj tíz különböző címet ehhez: [MŰ]. Keverd a komolyakat, a játékosakat és a titokzatosakat, és jelöld meg a kedvencedet.",
        ],
    ),
    (
        "Give me ten gift ideas for [PERSON] who likes [INTERESTS], budget [BUDGET]. Group them: practical, sentimental, surprising.",
        [
            "Donne-moi dix idées de cadeaux pour [PERSONNE] qui aime [CENTRES D'INTÉRÊT], budget [BUDGET]. Groupe-les : pratiques, sentimentales, surprenantes.",
            "Dammi dieci idee regalo per [PERSONA] a cui piace [INTERESSI], budget [BUDGET]. Raggruppale: pratiche, sentimentali, sorprendenti.",
            "Dame diez ideas de regalo para [PERSONA] a quien le gusta [INTERESES], presupuesto [PRESUPUESTO]. Agrúpalas: prácticas, sentimentales, sorprendentes.",
            "Dá-me dez ideias de presente para [PESSOA] que gosta de [INTERESSES], orçamento [ORÇAMENTO]. Agrupa-as: práticas, sentimentais, surpreendentes.",
            "Gib mir zehn Geschenkideen für [PERSON], die [INTERESSEN] mag, Budget [BUDGET]. Gruppiere sie: praktisch, gefühlvoll, überraschend.",
            "Adj tíz ajándékötletet [SZEMÉLY] számára, aki szereti ezt: [ÉRDEKLŐDÉS], költségkeret: [KERET]. Csoportosítsd: praktikus, érzelmes, meglepő.",
        ],
    ),
    (
        "Help me decide",
        [
            "Aide-moi à décider",
            "Aiutami a decidere",
            "Ayúdame a decidir",
            "Ajuda-me a decidir",
            "Hilf mir bei der Entscheidung",
            "Segíts dönteni",
        ],
    ),
    (
        "Here is data: [DATA]. Describe what stands out, what looks wrong, and which three questions it can and cannot answer.",
        [
            "Voici des données : [DONNÉES]. Décris ce qui ressort, ce qui semble erroné, et à quelles trois questions elles peuvent et ne peuvent pas répondre.",
            "Ecco i dati: [DATI]. Descrivi cosa spicca, cosa sembra sbagliato e a quali tre domande possono e non possono rispondere.",
            "Estos son los datos: [DATOS]. Describe qué destaca, qué parece erróneo y a qué tres preguntas pueden y no pueden responder.",
            "Aqui estão os dados: [DADOS]. Descreve o que se destaca, o que parece errado e a que três perguntas podem e não podem responder.",
            "Hier sind Daten: [DATEN]. Beschreibe, was auffällt, was falsch aussieht und welche drei Fragen sie beantworten können und welche nicht.",
            "Itt vannak az adatok: [ADATOK]. Írd le, mi tűnik ki, mi látszik hibásnak, és melyik három kérdésre adhatnak, illetve nem adhatnak választ.",
        ],
    ),
    (
        "Here is my CV: [CV]. And the job: [JOB]. Suggest what to emphasise, what to cut and three honest sentences for the cover letter. Do not invent anything.",
        [
            "Voici mon CV : [CV]. Et le poste : [POSTE]. Suggère ce qu'il faut mettre en avant, ce qu'il faut couper et trois phrases honnêtes pour la lettre de motivation. N'invente rien.",
            "Ecco il mio CV: [CV]. E il lavoro: [LAVORO]. Suggerisci cosa mettere in evidenza, cosa tagliare e tre frasi oneste per la lettera di presentazione. Non inventare nulla.",
            "Este es mi CV: [CV]. Y el puesto: [PUESTO]. Sugiere qué destacar, qué quitar y tres frases honestas para la carta de presentación. No inventes nada.",
            "Aqui está o meu CV: [CV]. E a vaga: [VAGA]. Sugere o que destacar, o que cortar e três frases honestas para a carta de apresentação. Não inventes nada.",
            "Hier ist mein Lebenslauf: [LEBENSLAUF]. Und die Stelle: [STELLE]. Schlage vor, was ich hervorheben und streichen soll, und drei ehrliche Sätze für das Anschreiben. Erfinde nichts.",
            "Itt az önéletrajzom: [ÖNÉLETRAJZ]. És az állás: [ÁLLÁS]. Javasold, mit emeljek ki, mit hagyjak ki, és adj három őszinte mondatot a motivációs levélhez. Ne találj ki semmit.",
        ],
    ),
    (
        "High accuracy, with a mode that can also answer questions about the audio.",
        [
            "Grande précision, avec un mode qui peut aussi répondre à des questions sur l'audio.",
            "Alta precisione, con una modalità che può anche rispondere a domande sull'audio.",
            "Alta precisión, con un modo que también puede responder preguntas sobre el audio.",
            "Alta precisão, com um modo que também responde a perguntas sobre o áudio.",
            "Hohe Genauigkeit, mit einem Modus, der auch Fragen zum Audio beantworten kann.",
            "Nagy pontosság, olyan móddal, amely a hanganyaggal kapcsolatos kérdésekre is válaszol.",
        ],
    ),
    (
        "I am thinking about [DECISION]. List the pros, the cons and the risks, ask me the three questions that matter most, and wait for my answers.",
        [
            "Je réfléchis à [DÉCISION]. Liste les avantages, les inconvénients et les risques, pose-moi les trois questions les plus importantes et attends mes réponses.",
            "Sto pensando a [DECISIONE]. Elenca pro, contro e rischi, fammi le tre domande più importanti e aspetta le mie risposte.",
            "Estoy pensando en [DECISIÓN]. Enumera pros, contras y riesgos, hazme las tres preguntas más importantes y espera mis respuestas.",
            "Estou a pensar em [DECISÃO]. Lista os prós, os contras e os riscos, faz-me as três perguntas mais importantes e espera pelas minhas respostas.",
            "Ich überlege [ENTSCHEIDUNG]. Liste Vor- und Nachteile und Risiken auf, stelle mir die drei wichtigsten Fragen und warte auf meine Antworten.",
            "Azon gondolkodom: [DÖNTÉS]. Sorold fel az előnyöket, a hátrányokat és a kockázatokat, tedd fel a három legfontosabb kérdést, és várd meg a válaszaimat.",
        ],
    ),
    (
        "I have 3 apples, eat one, then buy twice as many as I have left. How many apples do I have?",
        [
            "J'ai 3 pommes, j'en mange une, puis j'en achète deux fois plus que ce qui me reste. Combien ai-je de pommes ?",
            "Ho 3 mele, ne mangio una, poi ne compro il doppio di quelle che mi restano. Quante mele ho?",
            "Tengo 3 manzanas, me como una y luego compro el doble de las que me quedan. ¿Cuántas manzanas tengo?",
            "Tenho 3 maçãs, como uma e depois compro o dobro das que me sobram. Quantas maçãs tenho?",
            "Ich habe 3 Äpfel, esse einen und kaufe dann doppelt so viele, wie mir bleiben. Wie viele Äpfel habe ich?",
            "Van 3 almám, megeszem egyet, majd vásárolok kétszer annyit, mint amennyi maradt. Hány almám van?",
        ],
    ),
    (
        "I have [INGREDIENTS], [TIME] minutes and [EQUIPMENT]. Suggest three meals, then give the full recipe of the one I pick.",
        [
            "J'ai [INGRÉDIENTS], [TEMPS] minutes et [ÉQUIPEMENT]. Propose trois repas, puis donne la recette complète de celui que je choisis.",
            "Ho [INGREDIENTI], [TEMPO] minuti e [ATTREZZATURA]. Suggerisci tre pasti, poi dai la ricetta completa di quello che scelgo.",
            "Tengo [INGREDIENTES], [TIEMPO] minutos y [EQUIPO]. Sugiere tres comidas y luego da la receta completa de la que elija.",
            "Tenho [INGREDIENTES], [TEMPO] minutos e [EQUIPAMENTO]. Sugere três refeições e depois dá a receita completa da que eu escolher.",
            "Ich habe [ZUTATEN], [ZEIT] Minuten und [AUSSTATTUNG]. Schlage drei Gerichte vor und gib dann das vollständige Rezept für das, das ich wähle.",
            "Van [HOZZÁVALÓK], [IDŐ] percem és [FELSZERELÉS]. Javasolj három ételt, majd add meg annak a teljes receptjét, amelyiket kiválasztom.",
        ],
    ),
    (
        "I need Python to reach the internet.",
        [
            "J'ai besoin de Python pour accéder à Internet.",
            "Mi serve Python per raggiungere Internet.",
            "Necesito Python para llegar a internet.",
            "Preciso do Python para chegar à internet.",
            "Ich brauche Python, um ins Internet zu kommen.",
            "Szükségem van a Pythonra, hogy elérjem az internetet.",
        ],
    ),
    (
        "Idioms",
        [
            "Expressions idiomatiques",
            "Modi di dire",
            "Modismos",
            "Expressões idiomáticas",
            "Redewendungen",
            "Szólások",
        ],
    ),
    (
        "If all roses are flowers and some flowers fade quickly, does it follow that some roses fade quickly? Explain.",
        [
            "Si toutes les roses sont des fleurs et que certaines fleurs fanent vite, s'ensuit-il que certaines roses fanent vite ? Explique.",
            "Se tutte le rose sono fiori e alcuni fiori appassiscono in fretta, ne segue che alcune rose appassiscono in fretta? Spiega.",
            "Si todas las rosas son flores y algunas flores se marchitan rápido, ¿se deduce que algunas rosas se marchitan rápido? Explica.",
            "Se todas as rosas são flores e algumas flores murcham depressa, conclui-se que algumas rosas murcham depressa? Explica.",
            "Wenn alle Rosen Blumen sind und manche Blumen schnell verwelken, folgt dann, dass manche Rosen schnell verwelken? Erkläre.",
            "Ha minden rózsa virág, és néhány virág gyorsan elhervad, következik-e ebből, hogy néhány rózsa gyorsan elhervad? Magyarázd el.",
        ],
    ),
    (
        "Improve my text",
        [
            "Améliorer mon texte",
            "Migliorare il mio testo",
            "Mejorar mi texto",
            "Melhorar o meu texto",
            "Meinen Text verbessern",
            "Javítsd a szövegemet",
        ],
    ),
    (
        "Improve the following text: fix mistakes, make it clearer and shorter, keep my meaning and voice. Show the result, then list what you changed.

[TEXT]",
        [
            "Améliore le texte suivant : corrige les erreurs, rends-le plus clair et plus court, garde mon sens et ma voix. Montre le résultat, puis liste ce que tu as changé.

[TEXTE]",
            "Migliora il testo seguente: correggi gli errori, rendilo più chiaro e breve, mantieni il mio senso e la mia voce. Mostra il risultato, poi elenca cosa hai cambiato.

[TESTO]",
            "Mejora el siguiente texto: corrige los errores, hazlo más claro y corto, mantén mi sentido y mi voz. Muestra el resultado y luego enumera lo que cambiaste.

[TEXTO]",
            "Melhora o texto seguinte: corrige os erros, torna-o mais claro e curto, mantém o meu sentido e a minha voz. Mostra o resultado e depois lista o que mudaste.

[TEXTO]",
            "Verbessere den folgenden Text: Korrigiere Fehler, mache ihn klarer und kürzer, behalte meinen Sinn und meine Stimme. Zeige das Ergebnis und liste dann auf, was du geändert hast.

[TEXT]",
            "Javítsd a következő szöveget: javítsd a hibákat, tedd világosabbá és rövidebbé, őrizd meg a jelentését és a hangomat. Mutasd meg az eredményt, majd sorold fel, mit változtattál.

[SZÖVEG]",
        ],
    ),
    (
        "Knowledge",
        [
            "Connaissances",
            "Conoscenze",
            "Conocimientos",
            "Conhecimentos",
            "Wissen",
            "Ismeretek",
        ],
    ),
    (
        "Lets an AI assistant drive ComfyUI through the Model Context Protocol (planned here).",
        [
            "Permet à un assistant IA de piloter ComfyUI via le Model Context Protocol (prévu ici).",
            "Permette a un assistente IA di controllare ComfyUI tramite il Model Context Protocol (previsto qui).",
            "Permite que un asistente de IA maneje ComfyUI mediante el Model Context Protocol (previsto aquí).",
            "Permite que um assistente de IA controle o ComfyUI através do Model Context Protocol (previsto aqui).",
            "Lässt einen KI-Assistenten ComfyUI über das Model Context Protocol steuern (hier geplant).",
            "Lehetővé teszi, hogy egy MI-asszisztens a Model Context Protocolon át vezérelje a ComfyUI-t (itt tervezett).",
        ],
    ),
    (
        "Light voices for small devices and home announcements; the original MIT repository was archived in October 2025.",
        [
            "Voix légères pour petits appareils et annonces domestiques ; le dépôt MIT d'origine a été archivé en octobre 2025.",
            "Voci leggere per piccoli dispositivi e annunci domestici; il repository MIT originale è stato archiviato a ottobre 2025.",
            "Voces ligeras para dispositivos pequeños y avisos domésticos; el repositorio MIT original se archivó en octubre de 2025.",
            "Vozes leves para dispositivos pequenos e avisos domésticos; o repositório MIT original foi arquivado em outubro de 2025.",
            "Leichte Stimmen für kleine Geräte und Haus-Ansagen; das ursprüngliche MIT-Repository wurde im Oktober 2025 archiviert.",
            "Könnyű hangok kis eszközökhöz és otthoni bemondásokhoz; az eredeti MIT-tárolót 2025 októberében archiválták.",
        ],
    ),
    (
        "Lowest average error among the open models listed in October 2026; very strong on Chinese.",
        [
            "Erreur moyenne la plus faible parmi les modèles ouverts recensés en octobre 2026 ; très fort en chinois.",
            "Errore medio più basso tra i modelli aperti elencati a ottobre 2026; molto forte sul cinese.",
            "Menor error medio entre los modelos abiertos listados en octubre de 2026; muy fuerte en chino.",
            "Menor erro médio entre os modelos abertos listados em outubro de 2026; muito forte em chinês.",
            "Geringster mittlerer Fehler unter den im Oktober 2026 gelisteten offenen Modellen; sehr stark bei Chinesisch.",
            "A 2026 októberében listázott nyílt modellek közül a legalacsonyabb átlagos hiba; kínaiban nagyon erős.",
        ],
    ),
    (
        "Lyrics to full songs with vocals; heavy to run.",
        [
            "Des paroles aux chansons complètes avec voix ; lourd à exécuter.",
            "Dai testi a canzoni complete con voce; pesante da eseguire.",
            "De la letra a canciones completas con voz; pesado de ejecutar.",
            "Da letra a canções completas com voz; pesado de executar.",
            "Vom Liedtext zum vollständigen Song mit Gesang; aufwendig auszuführen.",
            "A dalszövegtől a teljes, énekes dalig; nehéz futtatni.",
        ],
    ),
    (
        "Make [NUMBER] flashcards from the text below as a two-column table: question | answer.

[TEXT]",
        [
            "Crée [NOMBRE] cartes mémoire à partir du texte ci-dessous, sous forme de tableau à deux colonnes : question | réponse.

[TEXTE]",
            "Crea [NUMERO] flashcard dal testo qui sotto, come tabella a due colonne: domanda | risposta.

[TESTO]",
            "Crea [NÚMERO] tarjetas de estudio con el texto de abajo, como tabla de dos columnas: pregunta | respuesta.

[TEXTO]",
            "Cria [NÚMERO] cartões de estudo a partir do texto abaixo, como tabela de duas colunas: pergunta | resposta.

[TEXTO]",
            "Erstelle [ANZAHL] Lernkarten aus dem folgenden Text als zweispaltige Tabelle: Frage | Antwort.

[TEXT]",
            "Készíts [SZÁM] tanulókártyát az alábbi szövegből kétoszlopos táblázatként: kérdés | válasz.

[SZÖVEG]",
        ],
    ),
    (
        "Make a study plan to learn [SUBJECT] in [TIME] with [HOURS] hours a week. Give weekly goals and a way to check progress.",
        [
            "Fais un plan d'étude pour apprendre [MATIÈRE] en [DURÉE] avec [HEURES] heures par semaine. Donne des objectifs hebdomadaires et un moyen de vérifier les progrès.",
            "Crea un piano di studio per imparare [MATERIA] in [TEMPO] con [ORE] ore a settimana. Dai obiettivi settimanali e un modo per verificare i progressi.",
            "Haz un plan de estudio para aprender [MATERIA] en [TIEMPO] con [HORAS] horas a la semana. Da objetivos semanales y una forma de comprobar el avance.",
            "Faz um plano de estudo para aprender [MATÉRIA] em [TEMPO] com [HORAS] horas por semana. Dá metas semanais e uma forma de verificar o progresso.",
            "Erstelle einen Lernplan, um [FACH] in [ZEIT] zu lernen, mit [STUNDEN] Stunden pro Woche. Gib Wochenziele und eine Möglichkeit, den Fortschritt zu prüfen.",
            "Készíts tanulási tervet a(z) [TÁRGY] megtanulásához [IDŐ] alatt, heti [ÓRA] órával. Adj heti célokat és módot a haladás ellenőrzésére.",
        ],
    ),
    (
        "Makes video and its sound together in one pass, up to 4K; fast and flexible to adapt.",
        [
            "Produit la vidéo et son son en une seule passe, jusqu'en 4K ; rapide et facile à adapter.",
            "Crea video e audio insieme in un solo passaggio, fino a 4K; veloce e flessibile da adattare.",
            "Crea el vídeo y su sonido a la vez, en una sola pasada, hasta 4K; rápido y flexible de adaptar.",
            "Cria o vídeo e o seu som de uma só vez, até 4K; rápido e flexível de adaptar.",
            "Erzeugt Video und Ton zusammen in einem Durchgang, bis 4K; schnell und flexibel anpassbar.",
            "A videót és a hangját egy menetben készíti, 4K-ig; gyors és rugalmasan igazítható.",
        ],
    ),
    (
        "Meeting notes",
        [
            "Notes de réunion",
            "Appunti di riunione",
            "Notas de reunión",
            "Notas de reunião",
            "Besprechungsnotizen",
            "Megbeszélés jegyzetei",
        ],
    ),
    (
        "Model cards and datasets; read pages by address.",
        [
            "Fiches de modèles et jeux de données ; lisez les pages par adresse.",
            "Schede di modelli e dataset; leggi le pagine tramite indirizzo.",
            "Fichas de modelos y conjuntos de datos; lee páginas por dirección.",
            "Fichas de modelos e conjuntos de dados; leia páginas pelo endereço.",
            "Modellkarten und Datensätze; Seiten über die Adresse lesen.",
            "Modellkártyák és adathalmazok; oldalak olvasása cím alapján.",
        ],
    ),
    (
        "Multilingual local music model; promising but still rough for finished songs.",
        [
            "Modèle musical local multilingue ; prometteur mais encore brut pour des chansons finies.",
            "Modello musicale locale multilingue; promettente ma ancora grezzo per canzoni finite.",
            "Modelo musical local multilingüe; prometedor pero aún tosco para canciones terminadas.",
            "Modelo musical local multilingue; promissor, mas ainda tosco para canções acabadas.",
            "Mehrsprachiges lokales Musikmodell; vielversprechend, aber für fertige Songs noch grob.",
            "Többnyelvű helyi zenei modell; ígéretes, de kész dalokhoz még nyers.",
        ],
    ),
    (
        "Multilingual, streaming, with voice cloning.",
        [
            "Multilingue, en continu, avec clonage de voix.",
            "Multilingue, in streaming, con clonazione vocale.",
            "Multilingüe, en streaming, con clonación de voz.",
            "Multilingue, em streaming, com clonagem de voz.",
            "Mehrsprachig, Streaming, mit Stimmklonen.",
            "Többnyelvű, folyamatos, hangklónozással.",
        ],
    ),
    (
        "Needs a Google sign-in.",
        [
            "Nécessite une connexion Google.",
            "Richiede l'accesso con Google.",
            "Requiere iniciar sesión con Google.",
            "Requer início de sessão Google.",
            "Erfordert eine Google-Anmeldung.",
            "Google-bejelentkezést igényel.",
        ],
    ),
    (
        "Needs a Microsoft sign-in.",
        [
            "Nécessite une connexion Microsoft.",
            "Richiede l'accesso con Microsoft.",
            "Requiere iniciar sesión con Microsoft.",
            "Requer início de sessão Microsoft.",
            "Erfordert eine Microsoft-Anmeldung.",
            "Microsoft-bejelentkezést igényel.",
        ],
    ),
    (
        "Node-based workflows for pictures, video and audio, with a local REST API on port 8188; wrappers add an OpenAI-style images endpoint.",
        [
            "Flux de travail à nœuds pour images, vidéo et audio, avec une API REST locale sur le port 8188 ; des adaptateurs ajoutent un point d'accès d'images de style OpenAI.",
            "Flussi di lavoro a nodi per immagini, video e audio, con API REST locale sulla porta 8188; alcuni adattatori aggiungono un endpoint immagini in stile OpenAI.",
            "Flujos de trabajo con nodos para imágenes, vídeo y audio, con una API REST local en el puerto 8188; algunos adaptadores añaden un punto de imágenes al estilo OpenAI.",
            "Fluxos de trabalho com nós para imagens, vídeo e áudio, com API REST local na porta 8188; alguns adaptadores acrescentam um ponto de imagens ao estilo OpenAI.",
            "Knotenbasierte Abläufe für Bilder, Video und Audio, mit lokaler REST-API auf Port 8188; Adapter ergänzen einen Bilder-Endpunkt im OpenAI-Stil.",
            "Csomópontalapú munkafolyamatok képekhez, videóhoz és hanghoz, helyi REST API-val a 8188-as porton; kiegészítők OpenAI-stílusú képvégpontot adnak hozzá.",
        ],
    ),
    (
        "Not found (wrong address or name, or a private repository).",
        [
            "Introuvable (adresse ou nom erroné, ou dépôt privé).",
            "Non trovato (indirizzo o nome sbagliato, oppure repository privato).",
            "No encontrado (dirección o nombre erróneo, o repositorio privado).",
            "Não encontrado (endereço ou nome errado, ou repositório privado).",
            "Nicht gefunden (falsche Adresse oder falscher Name oder privates Repository).",
            "Nem található (hibás cím vagy név, vagy privát tároló).",
        ],
    ),
    (
        "Notify: it sends messages and alerts",
        [
            "Notifier : il envoie des messages et des alertes",
            "Notificare: invia messaggi e avvisi",
            "Notificar: envía mensajes y avisos",
            "Notificar: envia mensagens e alertas",
            "Benachrichtigen: sendet Nachrichten und Warnungen",
            "Értesítés: üzeneteket és figyelmeztetéseket küld",
        ],
    ),
    (
        "Old copies of pages, books and recordings.",
        [
            "Anciennes copies de pages, de livres et d'enregistrements.",
            "Vecchie copie di pagine, libri e registrazioni.",
            "Copias antiguas de páginas, libros y grabaciones.",
            "Cópias antigas de páginas, livros e gravações.",
            "Alte Kopien von Seiten, Büchern und Aufnahmen.",
            "Oldalak, könyvek és felvételek régi másolatai.",
        ],
    ),
    (
        "Older but still capable open video models.",
        [
            "Modèles vidéo ouverts plus anciens mais encore performants.",
            "Modelli video aperti più vecchi ma ancora capaci.",
            "Modelos de vídeo abiertos más antiguos pero aún capaces.",
            "Modelos de vídeo abertos mais antigos mas ainda capazes.",
            "Ältere, aber noch leistungsfähige offene Videomodelle.",
            "Régebbi, de még mindig képes nyílt videómodellek.",
        ],
    ),
    (
        "Open catalogues of scholarly works.",
        [
            "Catalogues ouverts d'œuvres savantes.",
            "Cataloghi aperti di opere accademiche.",
            "Catálogos abiertos de obras académicas.",
            "Catálogos abertos de obras académicas.",
            "Offene Kataloge wissenschaftlicher Werke.",
            "Tudományos művek nyílt katalógusai.",
        ],
    ),
    (
        "Open public data.",
        [
            "Données publiques ouvertes.",
            "Dati pubblici aperti.",
            "Datos públicos abiertos.",
            "Dados públicos abertos.",
            "Offene öffentliche Daten.",
            "Nyílt közérdekű adatok.",
        ],
    ),
    (
        "Open research papers; read abstracts and PDFs by address.",
        [
            "Articles de recherche ouverts ; lisez résumés et PDF par adresse.",
            "Articoli scientifici aperti; leggi abstract e PDF tramite indirizzo.",
            "Artículos científicos abiertos; lee resúmenes y PDF por dirección.",
            "Artigos científicos abertos; leia resumos e PDF pelo endereço.",
            "Offene Forschungsarbeiten; Zusammenfassungen und PDFs über die Adresse lesen.",
            "Nyílt kutatási cikkek; kivonatok és PDF-ek olvasása cím alapján.",
        ],
    ),
    (
        "Open, decentralised chat; read and send in rooms.",
        [
            "Messagerie ouverte et décentralisée ; lisez et envoyez dans des salons.",
            "Chat aperta e decentralizzata; leggi e invia nelle stanze.",
            "Chat abierto y descentralizado; lee y envía en salas.",
            "Chat aberto e descentralizado; leia e envie em salas.",
            "Offener, dezentraler Chat; in Räumen lesen und senden.",
            "Nyílt, decentralizált csevegés; olvasás és küldés szobákban.",
        ],
    ),
    (
        "Open-source automation: the model calls a workflow with a webhook, the workflow does the rest.",
        [
            "Automatisation open source : le modèle appelle un flux de travail par webhook, le flux fait le reste.",
            "Automazione open source: il modello chiama un flusso con un webhook, il flusso fa il resto.",
            "Automatización de código abierto: el modelo llama a un flujo con un webhook y el flujo hace el resto.",
            "Automação de código aberto: o modelo chama um fluxo com um webhook e o fluxo faz o resto.",
            "Open-Source-Automatisierung: Das Modell ruft einen Ablauf per Webhook auf, den Rest erledigt der Ablauf.",
            "Nyílt forráskódú automatizálás: a modell webhookkal hív meg egy munkafolyamatot, a többit az elvégzi.",
        ],
    ),
    (
        "Open-source project tools; read work packages.",
        [
            "Outils de projet open source ; lisez les lots de travaux.",
            "Strumenti di progetto open source; leggi i pacchetti di lavoro.",
            "Herramientas de proyecto de código abierto; lee paquetes de trabajo.",
            "Ferramentas de projeto de código aberto; leia pacotes de trabalho.",
            "Open-Source-Projektwerkzeuge; Arbeitspakete lesen.",
            "Nyílt forráskódú projekteszközök; munkacsomagok olvasása.",
        ],
    ),
    (
        "Permissive licence with generation, editing and personalisation in one model.",
        [
            "Licence permissive, avec génération, édition et personnalisation dans un seul modèle.",
            "Licenza permissiva con generazione, modifica e personalizzazione in un solo modello.",
            "Licencia permisiva con generación, edición y personalización en un solo modelo.",
            "Licença permissiva com geração, edição e personalização num só modelo.",
            "Freizügige Lizenz, mit Erzeugung, Bearbeitung und Anpassung in einem Modell.",
            "Megengedő licenc; generálás, szerkesztés és személyre szabás egyetlen modellben.",
        ],
    ),
    (
        "Photorealism and cinematic looks; the largest open image model.",
        [
            "Photoréalisme et rendu cinématographique ; le plus grand modèle d'images ouvert.",
            "Fotorealismo e look cinematografico; il più grande modello di immagini aperto.",
            "Fotorrealismo y aspecto cinematográfico; el mayor modelo de imágenes abierto.",
            "Fotorrealismo e aspeto cinematográfico; o maior modelo de imagens aberto.",
            "Fotorealismus und Kino-Look; das größte offene Bildmodell.",
            "Fotorealizmus és filmes megjelenés; a legnagyobb nyílt képmodell.",
        ],
    ),
    (
        "Plan [DAYS] days in [PLACE] for [PEOPLE], budget [BUDGET]. Per day: morning, afternoon, evening, travel times. Say what to book early.",
        [
            "Planifie [JOURS] jours à [LIEU] pour [PERSONNES], budget [BUDGET]. Par jour : matin, après-midi, soir, temps de trajet. Dis ce qu'il faut réserver tôt.",
            "Pianifica [GIORNI] giorni a [LUOGO] per [PERSONE], budget [BUDGET]. Per ogni giorno: mattina, pomeriggio, sera, tempi di viaggio. Di' cosa prenotare in anticipo.",
            "Planifica [DÍAS] días en [LUGAR] para [PERSONAS], presupuesto [PRESUPUESTO]. Por día: mañana, tarde, noche, tiempos de viaje. Di qué reservar con antelación.",
            "Planeia [DIAS] dias em [LOCAL] para [PESSOAS], orçamento [ORÇAMENTO]. Por dia: manhã, tarde, noite, tempos de viagem. Diz o que reservar com antecedência.",
            "Plane [TAGE] Tage in [ORT] für [PERSONEN], Budget [BUDGET]. Pro Tag: Vormittag, Nachmittag, Abend, Fahrzeiten. Sage, was man früh buchen sollte.",
            "Tervezz [NAPOK] napot ide: [HELY], [SZEMÉLYEK] részére, költségkeret: [KERET]. Naponta: délelőtt, délután, este, utazási idők. Mondd meg, mit érdemes korán lefoglalni.",
        ],
    ),
    (
        "Plan a trip",
        [
            "Planifier un voyage",
            "Pianificare un viaggio",
            "Planificar un viaje",
            "Planear uma viagem",
            "Eine Reise planen",
            "Utazás tervezése",
        ],
    ),
    (
        "Play [CHARACTER] in [SCENE]. Stay in character, keep replies short, and let me lead. Start now.",
        [
            "Joue [PERSONNAGE] dans [SCÈNE]. Reste dans le personnage, garde des réponses courtes et laisse-moi mener. Commence maintenant.",
            "Interpreta [PERSONAGGIO] in [SCENA]. Resta nel personaggio, tieni le risposte brevi e lasciami guidare. Inizia ora.",
            "Interpreta a [PERSONAJE] en [ESCENA]. Mantente en el papel, da respuestas cortas y déjame llevar la iniciativa. Empieza ya.",
            "Interpreta [PERSONAGEM] em [CENA]. Mantém-te na personagem, dá respostas curtas e deixa-me conduzir. Começa agora.",
            "Spiele [FIGUR] in [SZENE]. Bleibe in der Rolle, antworte kurz und lass mich führen. Fang jetzt an.",
            "Játszd [SZEREPLŐ] szerepét ebben: [JELENET]. Maradj a szerepben, röviden válaszolj, és hagyd, hogy én vezessek. Kezdd most.",
        ],
    ),
    (
        "Play a fair but demanding critic of this idea: [IDEA]. Name the weakest assumptions, what could go wrong and how to test it cheaply.",
        [
            "Joue un critique juste mais exigeant de cette idée : [IDÉE]. Nomme les hypothèses les plus fragiles, ce qui pourrait mal tourner et comment la tester à peu de frais.",
            "Fai da critico giusto ma esigente di questa idea: [IDEA]. Indica le ipotesi più deboli, cosa potrebbe andare storto e come testarla a basso costo.",
            "Haz de crítico justo pero exigente de esta idea: [IDEA]. Señala las suposiciones más débiles, qué podría salir mal y cómo probarla con poco costo.",
            "Faz de crítico justo mas exigente desta ideia: [IDEIA]. Aponta os pressupostos mais fracos, o que pode correr mal e como testá-la a baixo custo.",
            "Sei ein fairer, aber anspruchsvoller Kritiker dieser Idee: [IDEE]. Nenne die schwächsten Annahmen, was schiefgehen könnte und wie man es günstig testet.",
            "Légy tisztességes, de igényes kritikusa ennek az ötletnek: [ÖTLET]. Nevezd meg a leggyengébb feltevéseket, mi mehet félre, és hogyan lehet olcsón tesztelni.",
        ],
    ),
    (
        "Polite e-mail",
        [
            "E-mail poli",
            "E-mail cortese",
            "Correo educado",
            "E-mail cortês",
            "Höfliche E-Mail",
            "Udvarias e-mail",
        ],
    ),
    (
        "Post a message to a channel through an incoming webhook.",
        [
            "Publiez un message dans un canal via un webhook entrant.",
            "Pubblica un messaggio in un canale tramite un webhook in entrata.",
            "Publica un mensaje en un canal mediante un webhook de entrada.",
            "Publique uma mensagem num canal através de um webhook de entrada.",
            "Poste eine Nachricht in einen Kanal über einen eingehenden Webhook.",
            "Üzenet küldése egy csatornára bejövő webhookkal.",
        ],
    ),
    (
        "Post to a channel through a webhook.",
        [
            "Publiez dans un canal via un webhook.",
            "Pubblica in un canale tramite un webhook.",
            "Publica en un canal mediante un webhook.",
            "Publique num canal através de um webhook.",
            "In einen Kanal über einen Webhook posten.",
            "Küldés egy csatornára webhookkal.",
        ],
    ),
    (
        "Pros, cons, risks",
        [
            "Pour, contre, risques",
            "Pro, contro, rischi",
            "Pros, contras, riesgos",
            "Prós, contras, riscos",
            "Vorteile, Nachteile, Risiken",
            "Előnyök, hátrányok, kockázatok",
        ],
    ),
    (
        "Public archive of source code; read pages by address.",
        [
            "Archive publique de code source ; lisez les pages par adresse.",
            "Archivio pubblico di codice sorgente; leggi le pagine tramite indirizzo.",
            "Archivo público de código fuente; lee páginas por dirección.",
            "Arquivo público de código-fonte; leia páginas pelo endereço.",
            "Öffentliches Archiv für Quellcode; Seiten über die Adresse lesen.",
            "Forráskód nyilvános archívuma; oldalak olvasása cím alapján.",
        ],
    ),
    (
        "Quick lookups without an account.",
        [
            "Recherches rapides sans compte.",
            "Ricerche rapide senza account.",
            "Consultas rápidas sin cuenta.",
            "Consultas rápidas sem conta.",
            "Schnelle Abfragen ohne Konto.",
            "Gyors keresések fiók nélkül.",
        ],
    ),
    (
        "Quiz me",
        [
            "Interroge-moi",
            "Fammi un quiz",
            "Hazme un cuestionario",
            "Faz-me um questionário",
            "Frag mich ab",
            "Kérdezz ki",
        ],
    ),
    (
        "Quiz me on [TOPIC] with [NUMBER] questions, one at a time. Wait for my answer, say if it is right, explain, then ask the next.",
        [
            "Interroge-moi sur [SUJET] avec [NOMBRE] questions, une à la fois. Attends ma réponse, dis si elle est juste, explique, puis pose la suivante.",
            "Fammi un quiz su [ARGOMENTO] con [NUMERO] domande, una alla volta. Aspetta la mia risposta, dimmi se è giusta, spiega, poi fai la successiva.",
            "Hazme un cuestionario sobre [TEMA] con [NÚMERO] preguntas, una a la vez. Espera mi respuesta, di si es correcta, explica y luego haz la siguiente.",
            "Faz-me um questionário sobre [TEMA] com [NÚMERO] perguntas, uma de cada vez. Espera a minha resposta, diz se está certa, explica e depois faz a seguinte.",
            "Frag mich zu [THEMA] mit [ANZAHL] Fragen ab, eine nach der anderen. Warte auf meine Antwort, sage, ob sie stimmt, erkläre und stelle dann die nächste.",
            "Kérdezz ki a(z) [TÉMA] témából [SZÁM] kérdéssel, egyenként. Várd meg a válaszomat, mondd meg, helyes-e, magyarázd el, aztán jöhet a következő.",
        ],
    ),
    (
        "Read a page you give by address, as clean text.",
        [
            "Lit une page donnée par adresse, en texte propre.",
            "Legge una pagina che indichi tramite indirizzo, come testo pulito.",
            "Lee una página que das por dirección, como texto limpio.",
            "Lê uma página que indicar pelo endereço, como texto limpo.",
            "Liest eine Seite, die du per Adresse angibst, als sauberen Text.",
            "Elolvassa a megadott című oldalt, tiszta szövegként.",
        ],
    ),
    (
        "Read and draft documents (needs a Google sign-in).",
        [
            "Lire et rédiger des documents (nécessite une connexion Google).",
            "Leggere e redigere documenti (richiede l'accesso con Google).",
            "Leer y redactar documentos (requiere iniciar sesión con Google).",
            "Ler e redigir documentos (requer início de sessão Google).",
            "Dokumente lesen und entwerfen (erfordert eine Google-Anmeldung).",
            "Dokumentumok olvasása és fogalmazása (Google-bejelentkezés szükséges).",
        ],
    ),
    (
        "Read and draft issues.",
        [
            "Lire et rédiger des tickets.",
            "Leggere e redigere issue.",
            "Leer y redactar incidencias.",
            "Ler e redigir issues.",
            "Issues lesen und entwerfen.",
            "Hibajegyek olvasása és fogalmazása.",
        ],
    ),
    (
        "Read and draft table rows.",
        [
            "Lire et rédiger des lignes de tableau.",
            "Leggere e redigere righe di tabella.",
            "Leer y redactar filas de tabla.",
            "Ler e redigir linhas de tabela.",
            "Tabellenzeilen lesen und entwerfen.",
            "Táblázatsorok olvasása és fogalmazása.",
        ],
    ),
    (
        "Read and store files in buckets.",
        [
            "Lire et stocker des fichiers dans des buckets.",
            "Leggere e salvare file nei bucket.",
            "Leer y guardar archivos en buckets.",
            "Ler e guardar ficheiros em buckets.",
            "Dateien in Buckets lesen und speichern.",
            "Fájlok olvasása és tárolása gyűjtőkben (bucket).",
        ],
    ),
    (
        "Read boards and cards.",
        [
            "Lire des tableaux et des cartes.",
            "Leggere bacheche e schede.",
            "Leer tableros y tarjetas.",
            "Ler quadros e cartões.",
            "Boards und Karten lesen.",
            "Táblák és kártyák olvasása.",
        ],
    ),
    (
        "Read error reports and group them.",
        [
            "Lire les rapports d'erreurs et les regrouper.",
            "Leggere i rapporti di errore e raggrupparli.",
            "Leer informes de errores y agruparlos.",
            "Ler relatórios de erros e agrupá-los.",
            "Fehlerberichte lesen und gruppieren.",
            "Hibajelentések olvasása és csoportosítása.",
        ],
    ),
    (
        "Read failed build logs and explain the failure.",
        [
            "Lire les journaux d'une compilation en échec et expliquer l'échec.",
            "Leggere i log di una build fallita e spiegare il guasto.",
            "Leer los registros de una compilación fallida y explicar el fallo.",
            "Ler os registos de uma compilação falhada e explicar a falha.",
            "Protokolle fehlgeschlagener Builds lesen und den Fehler erklären.",
            "Sikertelen buildek naplóinak olvasása és a hiba magyarázata.",
        ],
    ),
    (
        "Read logs and container status; start or stop only with your approval.",
        [
            "Lire les journaux et l'état des conteneurs ; démarrer ou arrêter seulement avec votre accord.",
            "Leggere log e stato dei container; avviare o fermare solo con la tua approvazione.",
            "Leer registros y estado de contenedores; iniciar o detener solo con tu aprobación.",
            "Ler registos e estado dos contentores; iniciar ou parar apenas com a sua aprovação.",
            "Protokolle und Container-Status lesen; Starten oder Stoppen nur mit deiner Zustimmung.",
            "Naplók és konténerállapot olvasása; indítás vagy leállítás csak a te jóváhagyásoddal.",
        ],
    ),
    (
        "Read metrics and explain a spike.",
        [
            "Lire des métriques et expliquer un pic.",
            "Leggere le metriche e spiegare un picco.",
            "Leer métricas y explicar un pico.",
            "Ler métricas e explicar um pico.",
            "Metriken lesen und einen Ausreißer erklären.",
            "Metrikák olvasása és egy kiugrás magyarázata.",
        ],
    ),
    (
        "Read news and blog feeds by address.",
        [
            "Lire des flux d'actualités et de blogs par adresse.",
            "Leggere feed di notizie e blog tramite indirizzo.",
            "Leer fuentes de noticias y blogs por dirección.",
            "Ler feeds de notícias e blogues pelo endereço.",
            "Nachrichten- und Blog-Feeds über die Adresse lesen.",
            "Hír- és blogcsatornák olvasása cím alapján.",
        ],
    ),
    (
        "Read open issues of a repository and turn them into a plan.",
        [
            "Lire les tickets ouverts d'un dépôt et en faire un plan.",
            "Leggere le issue aperte di un repository e trasformarle in un piano.",
            "Leer las incidencias abiertas de un repositorio y convertirlas en un plan.",
            "Ler as issues abertas de um repositório e transformá-las num plano.",
            "Offene Issues eines Repositories lesen und in einen Plan verwandeln.",
            "Egy tároló nyitott hibajegyeinek olvasása és tervvé alakítása.",
        ],
    ),
    (
        "Read pages and databases, draft new pages.",
        [
            "Lire des pages et des bases de données, rédiger de nouvelles pages.",
            "Leggere pagine e database, redigere nuove pagine.",
            "Leer páginas y bases de datos, redactar páginas nuevas.",
            "Ler páginas e bases de dados, redigir páginas novas.",
            "Seiten und Datenbanken lesen, neue Seiten entwerfen.",
            "Oldalak és adatbázisok olvasása, új oldalak fogalmazása.",
        ],
    ),
    (
        "Read repositories and tickets.",
        [
            "Lire des dépôts et des tickets.",
            "Leggere repository e ticket.",
            "Leer repositorios y tickets.",
            "Ler repositórios e tickets.",
            "Repositories und Tickets lesen.",
            "Tárolók és hibajegyek olvasása.",
        ],
    ),
    (
        "Read repositories with an app password.",
        [
            "Lire des dépôts avec un mot de passe d'application.",
            "Leggere repository con una password per app.",
            "Leer repositorios con una contraseña de aplicación.",
            "Ler repositórios com uma palavra-passe de aplicação.",
            "Repositories mit einem App-Passwort lesen.",
            "Tárolók olvasása alkalmazásjelszóval.",
        ],
    ),
    (
        "Read tasks and projects.",
        [
            "Lire des tâches et des projets.",
            "Leggere attività e progetti.",
            "Leer tareas y proyectos.",
            "Ler tarefas e projetos.",
            "Aufgaben und Projekte lesen.",
            "Feladatok és projektek olvasása.",
        ],
    ),
    (
        "Read team wiki pages.",
        [
            "Lire les pages wiki de l'équipe.",
            "Leggere le pagine wiki del team.",
            "Leer las páginas wiki del equipo.",
            "Ler as páginas wiki da equipa.",
            "Wiki-Seiten des Teams lesen.",
            "A csapat wikioldalainak olvasása.",
        ],
    ),
    (
        "Read this data",
        [
            "Lire ces données",
            "Leggere questi dati",
            "Leer estos datos",
            "Ler estes dados",
            "Diese Daten lesen",
            "Olvasd el ezeket az adatokat",
        ],
    ),
    (
        "Read tickets, draft comments.",
        [
            "Lire des tickets, rédiger des commentaires.",
            "Leggere ticket, redigere commenti.",
            "Leer tickets, redactar comentarios.",
            "Ler tickets, redigir comentários.",
            "Tickets lesen, Kommentare entwerfen.",
            "Hibajegyek olvasása, megjegyzések fogalmazása.",
        ],
    ),
    (
        "Read your address book to help write messages.",
        [
            "Lire votre carnet d'adresses pour aider à écrire des messages.",
            "Leggere la tua rubrica per aiutare a scrivere messaggi.",
            "Leer tu libreta de direcciones para ayudar a escribir mensajes.",
            "Ler a sua lista de contactos para ajudar a escrever mensagens.",
            "Dein Adressbuch lesen, um beim Schreiben von Nachrichten zu helfen.",
            "A címjegyzéked olvasása az üzenetek írásának segítésére.",
        ],
    ),
    (
        "Read your agenda, propose events.",
        [
            "Lire votre agenda, proposer des événements.",
            "Leggere la tua agenda, proporre eventi.",
            "Leer tu agenda, proponer eventos.",
            "Ler a sua agenda, propor eventos.",
            "Deinen Kalender lesen, Termine vorschlagen.",
            "A naptáradat olvasása, események javaslata.",
        ],
    ),
    (
        "Read-only questions about a database.",
        [
            "Questions en lecture seule sur une base de données.",
            "Domande in sola lettura su un database.",
            "Preguntas de solo lectura sobre una base de datos.",
            "Perguntas só de leitura sobre uma base de dados.",
            "Nur lesende Fragen an eine Datenbank.",
            "Csak olvasási jellegű kérdések egy adatbázisról.",
        ],
    ),
    (
        "Read: you pick the content, the model reads it",
        [
            "Lecture : vous choisissez le contenu, le modèle le lit",
            "Lettura: scegli tu il contenuto, il modello lo legge",
            "Lectura: tú eliges el contenido, el modelo lo lee",
            "Leitura: você escolhe o conteúdo, o modelo lê",
            "Lesen: Du wählst den Inhalt, das Modell liest ihn",
            "Olvasás: te választod ki a tartalmat, a modell elolvassa",
        ],
    ),
    (
        "Reasoning",
        [
            "Raisonnement",
            "Ragionamento",
            "Razonamiento",
            "Raciocínio",
            "Logisches Denken",
            "Gondolkodás",
        ],
    ),
    (
        "Regular expression",
        [
            "Expression régulière",
            "Espressione regolare",
            "Expresión regular",
            "Expressão regular",
            "Regulärer Ausdruck",
            "Reguláris kifejezés",
        ],
    ),
    (
        "Review a file",
        [
            "Relire un fichier",
            "Rivedere un file",
            "Revisar un archivo",
            "Rever um ficheiro",
            "Eine Datei prüfen",
            "Fájl átnézése",
        ],
    ),
    (
        "Review the text: find mistakes and unclear parts and propose concrete improvements.",
        [
            "Relis le texte : trouve les erreurs et les passages peu clairs et propose des améliorations concrètes.",
            "Rivedi il testo: trova errori e parti poco chiare e proponi miglioramenti concreti.",
            "Revisa el texto: encuentra errores y partes poco claras y propón mejoras concretas.",
            "Revê o texto: encontra erros e partes pouco claras e propõe melhorias concretas.",
            "Prüfe den Text: Finde Fehler und unklare Stellen und schlage konkrete Verbesserungen vor.",
            "Nézd át a szöveget: keresd meg a hibákat és a nem világos részeket, és javasolj konkrét javításokat.",
        ],
    ),
    (
        "Role play",
        [
            "Jeu de rôle",
            "Gioco di ruolo",
            "Juego de rol",
            "Jogo de papéis",
            "Rollenspiel",
            "Szerepjáték",
        ],
    ),
    (
        "Romanian",
        [
            "Roumain",
            "Rumeno",
            "Rumano",
            "Romeno",
            "Rumänisch",
            "Román",
        ],
    ),
    (
        "Run saved automations on a timetable while this program is open.",
        [
            "Exécute les automatisations enregistrées selon un horaire tant que ce programme est ouvert.",
            "Esegue le automazioni salvate secondo un orario finché questo programma è aperto.",
            "Ejecuta las automatizaciones guardadas según un horario mientras este programa esté abierto.",
            "Executa as automações guardadas num horário enquanto este programa estiver aberto.",
            "Führt gespeicherte Automatisierungen nach Zeitplan aus, solange dieses Programm geöffnet ist.",
            "Futtatja a mentett automatizálásokat ütemezés szerint, amíg ez a program nyitva van.",
        ],
    ),
    (
        "SQL query",
        [
            "Requête SQL",
            "Query SQL",
            "Consulta SQL",
            "Consulta SQL",
            "SQL-Abfrage",
            "SQL-lekérdezés",
        ],
    ),
    (
        "Safety",
        [
            "Sécurité",
            "Sicurezza",
            "Seguridad",
            "Segurança",
            "Sicherheit",
            "Biztonság",
        ],
    ),
    (
        "Same as GitHub, also on your own GitLab server.",
        [
            "Comme GitHub, y compris sur votre propre serveur GitLab.",
            "Come GitHub, anche sul tuo server GitLab.",
            "Igual que GitHub, también en tu propio servidor GitLab.",
            "Como o GitHub, também no seu próprio servidor GitLab.",
            "Wie GitHub, auch auf deinem eigenen GitLab-Server.",
            "Ugyanaz, mint a GitHub, saját GitLab-szerveren is.",
        ],
    ),
    (
        "Samples, loops and sound effects up to about 47 seconds; not for sung songs.",
        [
            "Échantillons, boucles et effets sonores jusqu'à environ 47 secondes ; pas pour les chansons chantées.",
            "Campioni, loop ed effetti sonori fino a circa 47 secondi; non per canzoni cantate.",
            "Muestras, bucles y efectos de sonido de hasta unos 47 segundos; no para canciones cantadas.",
            "Amostras, loops e efeitos sonoros até cerca de 47 segundos; não serve para canções cantadas.",
            "Samples, Loops und Soundeffekte bis etwa 47 Sekunden; nicht für gesungene Lieder.",
            "Minták, hurkok és hanghatások legfeljebb kb. 47 másodpercig; énekelt dalokhoz nem alkalmas.",
        ],
    ),
    (
        "Say briefly what each file holds and how they relate.",
        [
            "Dis brièvement ce que contient chaque fichier et comment ils sont liés.",
            "Di' brevemente cosa contiene ogni file e come sono collegati.",
            "Di brevemente qué contiene cada archivo y cómo se relacionan.",
            "Diz brevemente o que contém cada ficheiro e como se relacionam.",
            "Sage kurz, was jede Datei enthält und wie sie zusammenhängen.",
            "Mondd el röviden, mit tartalmaz az egyes fájl, és hogyan kapcsolódnak egymáshoz.",
        ],
    ),
    (
        "Search: the model looks things up and browses by itself",
        [
            "Recherche : le modèle cherche et navigue tout seul",
            "Ricerca: il modello cerca e naviga da solo",
            "Búsqueda: el modelo busca y navega por sí mismo",
            "Pesquisa: o modelo procura e navega sozinho",
            "Suchen: Das Modell recherchiert und surft selbstständig",
            "Keresés: a modell maga néz utána és böngészik",
        ],
    ),
    (
        "Self-hosted push notifications.",
        [
            "Notifications push auto-hébergées.",
            "Notifiche push self-hosted.",
            "Notificaciones push autoalojadas.",
            "Notificações push auto-hospedadas.",
            "Selbst gehostete Push-Benachrichtigungen.",
            "Saját üzemeltetésű push-értesítések.",
        ],
    ),
    (
        "Self-hosted team chat; post through a webhook.",
        [
            "Messagerie d'équipe auto-hébergée ; publiez via un webhook.",
            "Chat di team self-hosted; pubblica tramite un webhook.",
            "Chat de equipo autoalojado; publica mediante un webhook.",
            "Chat de equipa auto-hospedado; publique através de um webhook.",
            "Selbst gehosteter Team-Chat; Beiträge über einen Webhook.",
            "Saját üzemeltetésű csapatchat; küldés webhookkal.",
        ],
    ),
    (
        "Serves Qwen-Image-2.1 at /v1/images/generations next to the chat models.",
        [
            "Sert Qwen-Image-2.1 à /v1/images/generations, à côté des modèles de chat.",
            "Serve Qwen-Image-2.1 su /v1/images/generations, accanto ai modelli di chat.",
            "Sirve Qwen-Image-2.1 en /v1/images/generations, junto a los modelos de chat.",
            "Serve o Qwen-Image-2.1 em /v1/images/generations, ao lado dos modelos de chat.",
            "Stellt Qwen-Image-2.1 unter /v1/images/generations bereit, neben den Chat-Modellen.",
            "A Qwen-Image-2.1-et a /v1/images/generations címen szolgálja ki a csevegőmodellek mellett.",
        ],
    ),
    (
        "Sharp pictures on small graphics cards.",
        [
            "Images nettes sur de petites cartes graphiques.",
            "Immagini nitide su schede grafiche piccole.",
            "Imágenes nítidas en tarjetas gráficas pequeñas.",
            "Imagens nítidas em placas gráficas pequenas.",
            "Scharfe Bilder auf kleinen Grafikkarten.",
            "Éles képek kisebb videokártyákon.",
        ],
    ),
    (
        "Short story",
        [
            "Nouvelle",
            "Racconto breve",
            "Cuento corto",
            "Conto curto",
            "Kurzgeschichte",
            "Novella",
        ],
    ),
    (
        "Simple triggers and actions through webhooks.",
        [
            "Déclencheurs et actions simples via webhooks.",
            "Trigger e azioni semplici tramite webhook.",
            "Disparadores y acciones simples mediante webhooks.",
            "Gatilhos e ações simples através de webhooks.",
            "Einfache Auslöser und Aktionen über Webhooks.",
            "Egyszerű eseményindítók és műveletek webhookokon át.",
        ],
    ),
    (
        "Sings lyrics; usable commercially with credit.",
        [
            "Chante les paroles ; utilisable commercialement avec mention de la source.",
            "Canta i testi; utilizzabile commercialmente con attribuzione.",
            "Canta letras; se puede usar comercialmente con atribución.",
            "Canta letras; utilizável comercialmente com atribuição.",
            "Singt Liedtexte; kommerziell nutzbar mit Namensnennung.",
            "Dalszöveget énekel; forrásmegjelöléssel kereskedelmileg is használható.",
        ],
    ),
    (
        "Smart home: a webhook can switch lights, scenes and scripts. Keep the model away from locks and alarms.",
        [
            "Maison connectée : un webhook peut allumer des lumières, des scènes et des scripts. Tenez le modèle éloigné des serrures et des alarmes.",
            "Casa intelligente: un webhook può accendere luci, scene e script. Tieni il modello lontano da serrature e allarmi.",
            "Casa inteligente: un webhook puede encender luces, escenas y scripts. Mantén al modelo alejado de cerraduras y alarmas.",
            "Casa inteligente: um webhook pode ligar luzes, cenas e scripts. Mantenha o modelo longe de fechaduras e alarmes.",
            "Smart Home: Ein Webhook kann Lichter, Szenen und Skripte schalten. Halte das Modell von Schlössern und Alarmanlagen fern.",
            "Okosotthon: egy webhook képes lámpákat, jeleneteket és szkripteket kapcsolni. A modellt tartsd távol a záraktól és a riasztóktól.",
        ],
    ),
    (
        "Structured facts about people, places and things.",
        [
            "Faits structurés sur les personnes, les lieux et les choses.",
            "Fatti strutturati su persone, luoghi e cose.",
            "Datos estructurados sobre personas, lugares y cosas.",
            "Factos estruturados sobre pessoas, lugares e coisas.",
            "Strukturierte Fakten über Personen, Orte und Dinge.",
            "Strukturált tények személyekről, helyekről és dolgokról.",
        ],
    ),
    (
        "Study",
        [
            "Étude",
            "Studio",
            "Estudio",
            "Estudo",
            "Lernen",
            "Tanulás",
        ],
    ),
    (
        "Study plan",
        [
            "Plan d'étude",
            "Piano di studio",
            "Plan de estudio",
            "Plano de estudo",
            "Lernplan",
            "Tanulási terv",
        ],
    ),
    (
        "Summarise",
        [
            "Résumer",
            "Riassumere",
            "Resumir",
            "Resumir",
            "Zusammenfassen",
            "Összefoglalás",
        ],
    ),
    (
        "Summarise a web page",
        [
            "Résumer une page web",
            "Riassumere una pagina web",
            "Resumir una página web",
            "Resumir uma página web",
            "Eine Webseite zusammenfassen",
            "Weboldal összefoglalása",
        ],
    ),
    (
        "Summarise in one sentence: \"Mihai left early for the station, but the train was already an hour late, so he waited in a cafe.\"",
        [
            "Résume en une phrase : « Mihai est parti tôt pour la gare, mais le train avait déjà une heure de retard, alors il a attendu dans un café. »",
            "Riassumi in una frase: «Mihai è andato presto alla stazione, ma il treno aveva già un'ora di ritardo, così ha aspettato in un caffè.»",
            "Resume en una frase: «Mihai salió temprano hacia la estación, pero el tren ya llevaba una hora de retraso, así que esperó en una cafetería.»",
            "Resume numa frase: «O Mihai saiu cedo para a estação, mas o comboio já tinha uma hora de atraso, por isso esperou num café.»",
            "Fasse in einem Satz zusammen: „Mihai ist früh zum Bahnhof gegangen, aber der Zug hatte schon eine Stunde Verspätung, also hat er in einem Café gewartet.“",
            "Foglald össze egy mondatban: „Mihai korán indult az állomásra, de a vonat már egy órát késett, ezért egy kávézóban várt.”",
        ],
    ),
    (
        "Summarise the page in 5 clear points, then say who it is useful to.",
        [
            "Résume la page en 5 points clairs, puis dis à qui elle est utile.",
            "Riassumi la pagina in 5 punti chiari, poi di' a chi è utile.",
            "Resume la página en 5 puntos claros y luego di a quién le resulta útil.",
            "Resume a página em 5 pontos claros e depois diz a quem é útil.",
            "Fasse die Seite in 5 klaren Punkten zusammen und sage dann, wem sie nützt.",
            "Foglald össze az oldalt 5 világos pontban, majd mondd meg, kinek hasznos.",
        ],
    ),
    (
        "Summarise the text below in [NUMBER] bullet points for someone who has two minutes. Then give one sentence on what to do next.

[TEXT]",
        [
            "Résume le texte ci-dessous en [NOMBRE] points pour quelqu'un qui a deux minutes. Puis donne une phrase sur ce qu'il faut faire ensuite.

[TEXTE]",
            "Riassumi il testo qui sotto in [NUMERO] punti per chi ha due minuti. Poi dai una frase su cosa fare dopo.

[TESTO]",
            "Resume el texto de abajo en [NÚMERO] puntos para alguien que tiene dos minutos. Luego da una frase sobre qué hacer a continuación.

[TEXTO]",
            "Resume o texto abaixo em [NÚMERO] pontos para alguém que tem dois minutos. Depois dá uma frase sobre o que fazer a seguir.

[TEXTO]",
            "Fasse den folgenden Text in [ANZAHL] Stichpunkten für jemanden zusammen, der zwei Minuten hat. Gib dann einen Satz dazu, was als Nächstes zu tun ist.

[TEXT]",
            "Foglald össze az alábbi szöveget [SZÁM] pontban valakinek, akinek két perce van. Utána egy mondatban mondd meg, mi a következő teendő.

[SZÖVEG]",
        ],
    ),
    (
        "Summarise your inbox, draft replies you approve before sending.",
        [
            "Résumez votre boîte de réception, rédigez des réponses que vous validez avant l'envoi.",
            "Riassumi la tua posta in arrivo, redigi risposte che approvi prima dell'invio.",
            "Resume tu bandeja de entrada y redacta respuestas que apruebas antes de enviar.",
            "Resuma a sua caixa de entrada e redija respostas que aprova antes de enviar.",
            "Fasse dein Postfach zusammen und entwirf Antworten, die du vor dem Senden freigibst.",
            "Foglalja össze a postaládádat, és fogalmazzon válaszokat, amelyeket küldés előtt jóváhagysz.",
        ],
    ),
    (
        "Tables: [TABLES]. Write a SQL query that [GOAL]. Explain it and say how to make it fast.",
        [
            "Tables : [TABLES]. Écris une requête SQL qui [OBJECTIF]. Explique-la et dis comment la rendre rapide.",
            "Tabelle: [TABELLE]. Scrivi una query SQL che [OBIETTIVO]. Spiegala e dì come renderla veloce.",
            "Tablas: [TABLAS]. Escribe una consulta SQL que [OBJETIVO]. Explícala y di cómo hacerla rápida.",
            "Tabelas: [TABELAS]. Escreve uma consulta SQL que [OBJETIVO]. Explica-a e diz como a tornar rápida.",
            "Tabellen: [TABELLEN]. Schreibe eine SQL-Abfrage, die [ZIEL]. Erkläre sie und sage, wie man sie schnell macht.",
            "Táblák: [TÁBLÁK]. Írj egy SQL-lekérdezést, amely [CÉL]. Magyarázd el, és mondd meg, hogyan lehet gyorssá tenni.",
        ],
    ),
    (
        "Tailor my CV",
        [
            "Adapter mon CV",
            "Adattare il mio CV",
            "Adaptar mi CV",
            "Adaptar o meu CV",
            "Meinen Lebenslauf anpassen",
            "Szabd személyre az önéletrajzomat",
        ],
    ),
    (
        "Ten titles",
        [
            "Dix titres",
            "Dieci titoli",
            "Diez títulos",
            "Dez títulos",
            "Zehn Titel",
            "Tíz cím",
        ],
    ),
    (
        "The answer is too big (over 8 MB).",
        [
            "La réponse est trop grande (plus de 8 Mo).",
            "La risposta è troppo grande (oltre 8 MB).",
            "La respuesta es demasiado grande (más de 8 MB).",
            "A resposta é grande demais (mais de 8 MB).",
            "Die Antwort ist zu groß (über 8 MB).",
            "A válasz túl nagy (8 MB felett).",
        ],
    ),
    (
        "The best all-round open video model in several 2026 comparisons, with no licence limits; text-to-video and picture-to-video. About nine minutes for five seconds on a big card.",
        [
            "Le meilleur modèle vidéo ouvert polyvalent dans plusieurs comparatifs de 2026, sans limite de licence ; texte vers vidéo et image vers vidéo. Environ neuf minutes pour cinq secondes sur une grosse carte.",
            "Il miglior modello video aperto tuttofare in diversi confronti del 2026, senza limiti di licenza; da testo a video e da immagine a video. Circa nove minuti per cinque secondi su una scheda grande.",
            "El mejor modelo de vídeo abierto en general en varias comparativas de 2026, sin límites de licencia; de texto a vídeo y de imagen a vídeo. Unos nueve minutos para cinco segundos en una tarjeta grande.",
            "O melhor modelo de vídeo aberto geral em várias comparações de 2026, sem limites de licença; de texto para vídeo e de imagem para vídeo. Cerca de nove minutos para cinco segundos numa placa grande.",
            "Das beste offene Allround-Videomodell in mehreren Vergleichen von 2026, ohne Lizenzbeschränkungen; Text-zu-Video und Bild-zu-Video. Etwa neun Minuten für fünf Sekunden auf einer großen Karte.",
            "A legjobb nyílt, általános célú videómodell több 2026-os összehasonlításban, licenckorlátok nélkül; szövegből és képből videó. Nagy kártyán kb. kilenc perc öt másodpercnyi videóhoz.",
        ],
    ),
    (
        "The biggest ecosystem of styles, add-ons and tutorials, and the lightest to run.",
        [
            "Le plus grand écosystème de styles, d'extensions et de tutoriels, et le plus léger à exécuter.",
            "Il più grande ecosistema di stili, componenti aggiuntivi e tutorial, e il più leggero da eseguire.",
            "El mayor ecosistema de estilos, complementos y tutoriales, y el más ligero de ejecutar.",
            "O maior ecossistema de estilos, extensões e tutoriais, e o mais leve de executar.",
            "Das größte Ökosystem an Stilen, Erweiterungen und Tutorials und das leichteste im Betrieb.",
            "A stílusok, kiegészítők és oktatóanyagok legnagyobb ökoszisztémája, és a legkönnyebben futtatható.",
        ],
    ),
    (
        "The most practical open song model: full tracks with sung lyrics in seconds, with editing tools. The only open music family that cleared every test in one October 2026 review.",
        [
            "Le modèle de chansons ouvert le plus pratique : des morceaux complets avec paroles chantées en quelques secondes, avec outils d'édition. La seule famille musicale ouverte à avoir réussi tous les tests d'un comparatif d'octobre 2026.",
            "Il modello di canzoni aperto più pratico: brani completi con testi cantati in pochi secondi, con strumenti di editing. L'unica famiglia musicale aperta che ha superato tutti i test in una rassegna di ottobre 2026.",
            "El modelo de canciones abierto más práctico: temas completos con letra cantada en segundos, con herramientas de edición. La única familia musical abierta que superó todas las pruebas en una reseña de octubre de 2026.",
            "O modelo de canções aberto mais prático: faixas completas com letra cantada em segundos, com ferramentas de edição. A única família musical aberta que passou todos os testes numa análise de outubro de 2026.",
            "Das praktischste offene Songmodell: komplette Titel mit gesungenem Text in Sekunden, mit Bearbeitungswerkzeugen. Die einzige offene Musikfamilie, die in einem Test im Oktober 2026 alle Prüfungen bestand.",
            "A legpraktikusabb nyílt dalmodell: másodpercek alatt teljes, énekelt szövegű számok, szerkesztőeszközökkel. Az egyetlen nyílt zenei család, amely egy 2026. októberi tesztben minden próbát kiállt.",
        ],
    ),
    (
        "The server answered: {}",
        [
            "Le serveur a répondu : {}",
            "Il server ha risposto: {}",
            "El servidor respondió: {}",
            "O servidor respondeu: {}",
            "Der Server antwortete: {}",
            "A szerver válasza: {}",
        ],
    ),
    (
        "This [LANGUAGE] code should [EXPECTED] but instead [ACTUAL]. Find the cause, show the smallest fix and explain why it works.

[CODE]",
        [
            "Ce code [LANGAGE] devrait [ATTENDU] mais fait [OBTENU]. Trouve la cause, montre la plus petite correction et explique pourquoi elle marche.

[CODE]",
            "Questo codice [LINGUAGGIO] dovrebbe [ATTESO] ma invece [OTTENUTO]. Trova la causa, mostra la correzione più piccola e spiega perché funziona.

[CODICE]",
            "Este código en [LENGUAJE] debería [ESPERADO] pero en cambio [OBTENIDO]. Encuentra la causa, muestra la corrección más pequeña y explica por qué funciona.

[CÓDIGO]",
            "Este código em [LINGUAGEM] devia [ESPERADO] mas em vez disso [OBTIDO]. Encontra a causa, mostra a correção mais pequena e explica por que funciona.

[CÓDIGO]",
            "Dieser [SPRACHE]-Code sollte [ERWARTET], tut aber stattdessen [TATSÄCHLICH]. Finde die Ursache, zeige die kleinste Korrektur und erkläre, warum sie funktioniert.

[CODE]",
            "Ennek a [NYELV] kódnak [VÁRT] kellene, de ehelyett [TÉNYLEGES]. Keresd meg az okot, mutasd a legkisebb javítást, és magyarázd el, miért működik.

[KÓD]",
        ],
    ),
    (
        "Thousands of apps behind one webhook (paid service).",
        [
            "Des milliers d'applications derrière un seul webhook (service payant).",
            "Migliaia di app dietro un solo webhook (servizio a pagamento).",
            "Miles de aplicaciones detrás de un solo webhook (servicio de pago).",
            "Milhares de aplicações atrás de um só webhook (serviço pago).",
            "Tausende Apps hinter einem Webhook (kostenpflichtiger Dienst).",
            "Több ezer alkalmazás egyetlen webhook mögött (fizetős szolgáltatás).",
        ],
    ),
    (
        "Tiny, fast, natural narration with preset voices; cannot copy a voice.",
        [
            "Narration minuscule, rapide et naturelle avec des voix prédéfinies ; ne peut pas copier une voix.",
            "Narrazione minuscola, veloce e naturale con voci predefinite; non può copiare una voce.",
            "Narración diminuta, rápida y natural con voces predefinidas; no puede copiar una voz.",
            "Narração minúscula, rápida e natural com vozes predefinidas; não consegue copiar uma voz.",
            "Winzige, schnelle, natürliche Sprachausgabe mit voreingestellten Stimmen; kann keine Stimme kopieren.",
            "Apró, gyors, természetes felolvasás előre beállított hangokkal; hangot nem tud másolni.",
        ],
    ),
    (
        "Too many requests: wait a little and try again.",
        [
            "Trop de requêtes : attendez un peu et réessayez.",
            "Troppe richieste: aspetta un po' e riprova.",
            "Demasiadas solicitudes: espera un poco e inténtalo de nuevo.",
            "Demasiados pedidos: espere um pouco e tente de novo.",
            "Zu viele Anfragen: Warte kurz und versuche es erneut.",
            "Túl sok kérés: várj egy kicsit, és próbáld újra.",
        ],
    ),
    (
        "Top of the open-weight text-to-image arena in October 2026; one model for making and editing pictures, strong at readable text. The only picture model colibri serves.",
        [
            "En tête de l'arène ouverte texte-vers-image en octobre 2026 ; un seul modèle pour créer et retoucher des images, fort pour le texte lisible. Le seul modèle d'images que sert colibri.",
            "In cima all'arena aperta text-to-image a ottobre 2026; un solo modello per creare e modificare immagini, forte nel testo leggibile. L'unico modello di immagini servito da colibri.",
            "En lo más alto de la arena abierta de texto a imagen en octubre de 2026; un solo modelo para crear y editar imágenes, fuerte en texto legible. El único modelo de imágenes que sirve colibri.",
            "No topo da arena aberta de texto para imagem em outubro de 2026; um só modelo para criar e editar imagens, forte em texto legível. O único modelo de imagens que o colibri serve.",
            "Spitze der offenen Text-zu-Bild-Arena im Oktober 2026; ein Modell zum Erzeugen und Bearbeiten von Bildern, stark bei lesbarem Text. Das einzige Bildmodell, das colibri bereitstellt.",
            "A nyílt szöveg-kép aréna élén 2026 októberében; egyetlen modell képek készítésére és szerkesztésére, erős az olvasható szövegben. Az egyetlen képmodell, amelyet a colibri kiszolgál.",
        ],
    ),
    (
        "Translate",
        [
            "Traduire",
            "Tradurre",
            "Traducir",
            "Traduzir",
            "Übersetzen",
            "Fordítás",
        ],
    ),
    (
        "Translate into Romanian: \"No worries, we will manage.\"",
        [
            "Traduis en roumain : « Pas de souci, on s'en sortira. »",
            "Traduci in rumeno: «Nessun problema, ce la caveremo.»",
            "Traduce al rumano: «No te preocupes, nos las arreglaremos.»",
            "Traduz para romeno: «Não faz mal, nós arranjamo-nos.»",
            "Übersetze ins Rumänische: „Keine Sorge, wir schaffen das.“",
            "Fordítsd románra: „Semmi baj, megoldjuk.”",
        ],
    ),
    (
        "Translate the text into [LANGUAGE]. Keep the tone and formatting. Where a phrase has no direct equivalent, add a short note.

[TEXT]",
        [
            "Traduis le texte en [LANGUE]. Garde le ton et la mise en forme. Quand une expression n'a pas d'équivalent direct, ajoute une courte note.

[TEXTE]",
            "Traduci il testo in [LINGUA]. Mantieni tono e formattazione. Dove una frase non ha equivalente diretto, aggiungi una breve nota.

[TESTO]",
            "Traduce el texto al [IDIOMA]. Mantén el tono y el formato. Donde una expresión no tenga equivalente directo, añade una nota breve.

[TEXTO]",
            "Traduz o texto para [LÍNGUA]. Mantém o tom e a formatação. Onde uma expressão não tiver equivalente direto, acrescenta uma nota breve.

[TEXTO]",
            "Übersetze den Text ins [SPRACHE]. Behalte Ton und Formatierung bei. Wo ein Ausdruck keine direkte Entsprechung hat, füge eine kurze Anmerkung hinzu.

[TEXT]",
            "Fordítsd le a szöveget [NYELV] nyelvre. Tartsd meg a hangnemet és a formázást. Ahol egy kifejezésnek nincs közvetlen megfelelője, fűzz hozzá egy rövid megjegyzést.

[SZÖVEG]",
        ],
    ),
    (
        "Translation",
        [
            "Traduction",
            "Traduzione",
            "Traducción",
            "Tradução",
            "Übersetzung",
            "Fordítás",
        ],
    ),
    (
        "Triage a repository's issues",
        [
            "Trier les tickets d'un dépôt",
            "Fare il triage delle issue di un repository",
            "Clasificar las incidencias de un repositorio",
            "Triar as issues de um repositório",
            "Issues eines Repositories sichten",
            "Egy tároló hibajegyeinek átvizsgálása",
        ],
    ),
    (
        "Turn these rough notes into minutes: decisions, tasks with owner and date, open questions.

[NOTES]",
        [
            "Transforme ces notes brutes en compte rendu : décisions, tâches avec responsable et date, questions ouvertes.

[NOTES]",
            "Trasforma questi appunti grezzi in un verbale: decisioni, compiti con responsabile e data, questioni aperte.

[APPUNTI]",
            "Convierte estas notas en bruto en un acta: decisiones, tareas con responsable y fecha, preguntas abiertas.

[NOTAS]",
            "Transforma estas notas em bruto numa ata: decisões, tarefas com responsável e data, questões em aberto.

[NOTAS]",
            "Mache aus diesen rohen Notizen ein Protokoll: Beschlüsse, Aufgaben mit Verantwortlichem und Datum, offene Fragen.

[NOTIZEN]",
            "Alakítsd ezeket a nyers jegyzeteket jegyzőkönyvvé: döntések, felelőssel és határidővel ellátott feladatok, nyitott kérdések.

[JEGYZETEK]",
        ],
    ),
    (
        "Typography, posters and branding.",
        [
            "Typographie, affiches et identité visuelle.",
            "Tipografia, poster e identità visiva.",
            "Tipografía, carteles e identidad de marca.",
            "Tipografia, cartazes e identidade visual.",
            "Typografie, Plakate und Markenauftritt.",
            "Tipográfia, plakátok és arculat.",
        ],
    ),
    (
        "Uncertainty",
        [
            "Incertitude",
            "Incertezza",
            "Incertidumbre",
            "Incerteza",
            "Unsicherheit",
            "Bizonytalanság",
        ],
    ),
    (
        "Unexpected answer: {}",
        [
            "Réponse inattendue : {}",
            "Risposta inattesa: {}",
            "Respuesta inesperada: {}",
            "Resposta inesperada: {}",
            "Unerwartete Antwort: {}",
            "Váratlan válasz: {}",
        ],
    ),
    (
        "Very small and fast, for devices with little power.",
        [
            "Très petit et rapide, pour les appareils peu puissants.",
            "Molto piccolo e veloce, per dispositivi con poca potenza.",
            "Muy pequeño y rápido, para dispositivos con poca potencia.",
            "Muito pequeno e rápido, para dispositivos com pouca potência.",
            "Sehr klein und schnell, für Geräte mit wenig Leistung.",
            "Nagyon kicsi és gyors, kis teljesítményű eszközökhöz.",
        ],
    ),
    (
        "Visual automations behind a webhook (paid service).",
        [
            "Automatisations visuelles derrière un webhook (service payant).",
            "Automazioni visuali dietro un webhook (servizio a pagamento).",
            "Automatizaciones visuales detrás de un webhook (servicio de pago).",
            "Automações visuais atrás de um webhook (serviço pago).",
            "Visuelle Automatisierungen hinter einem Webhook (kostenpflichtiger Dienst).",
            "Vizuális automatizálások egy webhook mögött (fizetős szolgáltatás).",
        ],
    ),
    (
        "Web search with an API key.",
        [
            "Recherche web avec une clé d'API.",
            "Ricerca web con una chiave API.",
            "Búsqueda web con una clave de API.",
            "Pesquisa na web com uma chave de API.",
            "Websuche mit einem API-Schlüssel.",
            "Webes keresés API-kulccsal.",
        ],
    ),
    (
        "What are the signs that a text message on your phone is a scam?",
        [
            "Quels sont les signes qu'un SMS reçu sur votre téléphone est une arnaque ?",
            "Quali sono i segnali che un messaggio ricevuto sul telefono è una truffa?",
            "¿Cuáles son las señales de que un mensaje de texto en tu teléfono es una estafa?",
            "Quais são os sinais de que uma mensagem recebida no telemóvel é uma burla?",
            "Woran erkennt man, dass eine SMS auf dem Handy ein Betrug ist?",
            "Miről ismerhető fel, hogy a telefonra érkezett szöveges üzenet csalás?",
        ],
    ),
    (
        "What is the capital of Australia, and why is it not Sydney?",
        [
            "Quelle est la capitale de l'Australie, et pourquoi n'est-ce pas Sydney ?",
            "Qual è la capitale dell'Australia e perché non è Sydney?",
            "¿Cuál es la capital de Australia y por qué no es Sídney?",
            "Qual é a capital da Austrália e por que não é Sydney?",
            "Was ist die Hauptstadt Australiens, und warum ist es nicht Sydney?",
            "Mi Ausztrália fővárosa, és miért nem Sydney?",
        ],
    ),
    (
        "What is wrong with this regular expression for a 6-digit Romanian postal code: ^\\d{5}$ ?",
        [
            "Qu'est-ce qui ne va pas dans cette expression régulière pour un code postal roumain à 6 chiffres : ^\\d{5}$ ?",
            "Cosa c'è che non va in questa espressione regolare per un codice postale rumeno a 6 cifre: ^\\d{5}$ ?",
            "¿Qué falla en esta expresión regular para un código postal rumano de 6 dígitos: ^\\d{5}$ ?",
            "O que está errado nesta expressão regular para um código postal romeno de 6 dígitos: ^\\d{5}$ ?",
            "Was ist an diesem regulären Ausdruck für eine sechsstellige rumänische Postleitzahl falsch: ^\\d{5}$ ?",
            "Mi a hiba ebben a reguláris kifejezésben egy 6 számjegyű román irányítószámra: ^\\d{5}$ ?",
        ],
    ),
    (
        "Who won the football World Cup in the year 2150?",
        [
            "Qui a gagné la Coupe du monde de football en l'an 2150 ?",
            "Chi ha vinto i Mondiali di calcio nell'anno 2150?",
            "¿Quién ganó el Mundial de fútbol en el año 2150?",
            "Quem ganhou o Mundial de futebol no ano 2150?",
            "Wer hat im Jahr 2150 die Fußball-Weltmeisterschaft gewonnen?",
            "Ki nyerte a labdarúgó-világbajnokságot 2150-ben?",
        ],
    ),
    (
        "Wire devices and services; an HTTP-in node receives the model's calls.",
        [
            "Reliez des appareils et des services ; un nœud HTTP-in reçoit les appels du modèle.",
            "Collega dispositivi e servizi; un nodo HTTP-in riceve le chiamate del modello.",
            "Conecta dispositivos y servicios; un nodo HTTP-in recibe las llamadas del modelo.",
            "Ligue dispositivos e serviços; um nó HTTP-in recebe as chamadas do modelo.",
            "Verbinde Geräte und Dienste; ein HTTP-in-Knoten empfängt die Aufrufe des Modells.",
            "Eszközök és szolgáltatások összekötése; egy HTTP-in csomópont fogadja a modell hívásait.",
        ],
    ),
    (
        "Work",
        [
            "Travail",
            "Lavoro",
            "Trabajo",
            "Trabalho",
            "Arbeit",
            "Munka",
        ],
    ),
    (
        "Write a 5-line brief of what is new on the page.",
        [
            "Rédige un résumé de 5 lignes de ce qui est nouveau sur la page.",
            "Scrivi un riassunto di 5 righe delle novità della pagina.",
            "Escribe un resumen de 5 líneas con las novedades de la página.",
            "Escreve um resumo de 5 linhas do que há de novo na página.",
            "Schreibe eine Kurzfassung in 5 Zeilen, was auf der Seite neu ist.",
            "Írj 5 soros összefoglalót arról, mi új az oldalon.",
        ],
    ),
    (
        "Write a Python function that checks whether a string is a palindrome, ignoring spaces and case.",
        [
            "Écris une fonction Python qui vérifie si une chaîne est un palindrome, en ignorant les espaces et la casse.",
            "Scrivi una funzione Python che verifica se una stringa è palindroma, ignorando spazi e maiuscole.",
            "Escribe una función en Python que compruebe si una cadena es un palíndromo, ignorando espacios y mayúsculas.",
            "Escreve uma função em Python que verifique se uma cadeia é um palíndromo, ignorando espaços e maiúsculas.",
            "Schreibe eine Python-Funktion, die prüft, ob eine Zeichenkette ein Palindrom ist, ohne Leerzeichen und Groß-/Kleinschreibung zu beachten.",
            "Írj egy Python-függvényt, amely ellenőrzi, hogy egy szöveg palindrom-e, a szóközöket és a kis-nagybetűket figyelmen kívül hagyva.",
        ],
    ),
    (
        "Write a formal letter in Romanian to [INSTITUTION] requesting [REQUEST]. Include the usual opening and closing formulas and a place for the registration number.",
        [
            "Rédige une lettre formelle en roumain à [INSTITUTION] pour demander [DEMANDE]. Inclus les formules d'ouverture et de clôture habituelles et un emplacement pour le numéro d'enregistrement.",
            "Scrivi una lettera formale in rumeno a [ISTITUZIONE] per richiedere [RICHIESTA]. Includi le consuete formule di apertura e chiusura e uno spazio per il numero di registrazione.",
            "Escribe una carta formal en rumano a [INSTITUCIÓN] solicitando [SOLICITUD]. Incluye las fórmulas habituales de apertura y cierre y un espacio para el número de registro.",
            "Escreve uma carta formal em romeno a [INSTITUIÇÃO] a solicitar [PEDIDO]. Inclui as fórmulas habituais de abertura e fecho e um espaço para o número de registo.",
            "Schreibe einen formellen Brief auf Rumänisch an [INSTITUTION], in dem du [ANLIEGEN] beantragst. Nutze die üblichen Anrede- und Schlussformeln und lasse Platz für die Registriernummer.",
            "Írj hivatalos levelet románul a(z) [INTÉZMÉNY] részére, amelyben [KÉRELEM] iránt érdeklődsz. Használd a szokásos megszólítást és záró formulát, és hagyj helyet az iktatószámnak.",
        ],
    ),
    (
        "Write a regular expression that matches [WHAT], with three examples that match and three that do not. Explain each part.",
        [
            "Écris une expression régulière qui reconnaît [QUOI], avec trois exemples qui correspondent et trois qui ne correspondent pas. Explique chaque partie.",
            "Scrivi un'espressione regolare che corrisponde a [COSA], con tre esempi che corrispondono e tre che no. Spiega ogni parte.",
            "Escribe una expresión regular que coincida con [QUÉ], con tres ejemplos que coinciden y tres que no. Explica cada parte.",
            "Escreve uma expressão regular que corresponda a [O QUÊ], com três exemplos que correspondem e três que não. Explica cada parte.",
            "Schreibe einen regulären Ausdruck, der [WAS] erkennt, mit drei passenden und drei nicht passenden Beispielen. Erkläre jeden Teil.",
            "Írj egy reguláris kifejezést, amely illeszkedik erre: [MIRE], három illeszkedő és három nem illeszkedő példával. Magyarázd el az egyes részeket.",
        ],
    ),
    (
        "Write a short, polite e-mail to [RECIPIENT] about [SUBJECT]. Tone: [TONE]. End with a clear next step.",
        [
            "Écris un e-mail court et poli à [DESTINATAIRE] au sujet de [SUJET]. Ton : [TON]. Termine par une prochaine étape claire.",
            "Scrivi una breve e-mail cortese a [DESTINATARIO] su [OGGETTO]. Tono: [TONO]. Concludi con un prossimo passo chiaro.",
            "Escribe un correo breve y educado a [DESTINATARIO] sobre [ASUNTO]. Tono: [TONO]. Termina con un siguiente paso claro.",
            "Escreve um e-mail curto e cortês para [DESTINATÁRIO] sobre [ASSUNTO]. Tom: [TOM]. Termina com um próximo passo claro.",
            "Schreibe eine kurze, höfliche E-Mail an [EMPFÄNGER] zum Thema [BETREFF]. Ton: [TON]. Schließe mit einem klaren nächsten Schritt.",
            "Írj egy rövid, udvarias e-mailt [CÍMZETT] részére a következőről: [TÁRGY]. Hangnem: [HANGNEM]. Zárd egyértelmű következő lépéssel.",
        ],
    ),
    (
        "Write a story of about [LENGTH] words. Genre: [GENRE]. Main character: [CHARACTER]. It must end with a surprise that fits the beginning.",
        [
            "Écris une histoire d'environ [LONGUEUR] mots. Genre : [GENRE]. Personnage principal : [PERSONNAGE]. Elle doit finir par une surprise qui s'accorde au début.",
            "Scrivi una storia di circa [LUNGHEZZA] parole. Genere: [GENERE]. Protagonista: [PERSONAGGIO]. Deve finire con una sorpresa coerente con l'inizio.",
            "Escribe un cuento de unas [LONGITUD] palabras. Género: [GÉNERO]. Personaje principal: [PERSONAJE]. Debe terminar con una sorpresa que encaje con el comienzo.",
            "Escreve uma história com cerca de [COMPRIMENTO] palavras. Género: [GÉNERO]. Personagem principal: [PERSONAGEM]. Tem de acabar com uma surpresa que combine com o início.",
            "Schreibe eine Geschichte von etwa [LÄNGE] Wörtern. Genre: [GENRE]. Hauptfigur: [FIGUR]. Sie muss mit einer Überraschung enden, die zum Anfang passt.",
            "Írj egy körülbelül [HOSSZ] szavas történetet. Műfaj: [MŰFAJ]. Főszereplő: [SZEREPLŐ]. Olyan meglepetéssel végződjön, amely illik a kezdethez.",
        ],
    ),
    (
        "Write a three-line e-mail postponing a meeting by one day.",
        [
            "Écris un e-mail de trois lignes reportant une réunion d'un jour.",
            "Scrivi un'e-mail di tre righe che rimanda una riunione di un giorno.",
            "Escribe un correo de tres líneas que aplace una reunión un día.",
            "Escreve um e-mail de três linhas a adiar uma reunião por um dia.",
            "Schreibe eine dreizeilige E-Mail, die eine Besprechung um einen Tag verschiebt.",
            "Írj háromsoros e-mailt, amely egy nappal elhalaszt egy megbeszélést.",
        ],
    ),
    (
        "Write four lines about autumn without using the word \"leaves\".",
        [
            "Écris quatre vers sur l'automne sans utiliser le mot « feuilles ».",
            "Scrivi quattro versi sull'autunno senza usare la parola «foglie».",
            "Escribe cuatro versos sobre el otoño sin usar la palabra «hojas».",
            "Escreve quatro versos sobre o outono sem usar a palavra «folhas».",
            "Schreibe vier Zeilen über den Herbst, ohne das Wort „Blätter“ zu benutzen.",
            "Írj négy sort az őszről anélkül, hogy a „levelek” szót használnád.",
        ],
    ),
    (
        "Write tests",
        [
            "Écrire des tests",
            "Scrivere test",
            "Escribir pruebas",
            "Escrever testes",
            "Tests schreiben",
            "Tesztek írása",
        ],
    ),
    (
        "Write tests for the code below using [FRAMEWORK]. Cover normal use, edge cases and errors. Explain what each test protects.

[CODE]",
        [
            "Écris des tests pour le code ci-dessous avec [FRAMEWORK]. Couvre l'usage normal, les cas limites et les erreurs. Explique ce que protège chaque test.

[CODE]",
            "Scrivi test per il codice qui sotto usando [FRAMEWORK]. Copri uso normale, casi limite ed errori. Spiega cosa protegge ogni test.

[CODICE]",
            "Escribe pruebas para el código de abajo usando [FRAMEWORK]. Cubre el uso normal, los casos límite y los errores. Explica qué protege cada prueba.

[CÓDIGO]",
            "Escreve testes para o código abaixo usando [FRAMEWORK]. Cobre o uso normal, os casos-limite e os erros. Explica o que cada teste protege.

[CÓDIGO]",
            "Schreibe Tests für den folgenden Code mit [FRAMEWORK]. Decke normale Nutzung, Grenzfälle und Fehler ab. Erkläre, was jeder Test absichert.

[CODE]",
            "Írj teszteket az alábbi kódhoz a(z) [KERETRENDSZER] használatával. Fedd le a normál használatot, a határeseteket és a hibákat. Magyarázd el, mit véd az egyes teszt.

[KÓD]",
        ],
    ),
    (
        "Write: it drafts or changes things; you approve each change",
        [
            "Écrire : il rédige ou modifie des choses ; vous approuvez chaque changement",
            "Scrivere: redige o modifica; approvi ogni modifica",
            "Escribir: redacta o cambia cosas; tú apruebas cada cambio",
            "Escrever: redige ou altera coisas; você aprova cada alteração",
            "Schreiben: entwirft oder ändert Dinge; du genehmigst jede Änderung",
            "Írás: fogalmaz vagy módosít; minden változtatást te hagysz jóvá",
        ],
    ),
    (
        "Writing",
        [
            "Écriture",
            "Scrittura",
            "Escritura",
            "Escrita",
            "Schreiben",
            "Írás",
        ],
    ),
    (
        "Your own cloud files: read and save documents.",
        [
            "Vos propres fichiers dans le cloud : lire et enregistrer des documents.",
            "I tuoi file nel cloud: leggere e salvare documenti.",
            "Tus propios archivos en la nube: leer y guardar documentos.",
            "Os seus próprios ficheiros na nuvem: ler e guardar documentos.",
            "Deine eigenen Cloud-Dateien: Dokumente lesen und speichern.",
            "Saját felhőfájljaid: dokumentumok olvasása és mentése.",
        ],
    ),
    (
        "Your reference library and notes.",
        [
            "Votre bibliothèque de références et vos notes.",
            "La tua biblioteca di riferimenti e i tuoi appunti.",
            "Tu biblioteca de referencias y tus notas.",
            "A sua biblioteca de referências e as suas notas.",
            "Deine Literatursammlung und deine Notizen.",
            "A hivatkozásgyűjteményed és a jegyzeteid.",
        ],
    ),
    (
        "[dev] is the 32B flagship; the small [klein] models make a picture in under half a second on a consumer GPU. Pictures from the non-commercial model may still be usable commercially, per one source: check.",
        [
            "[dev] est le modèle phare de 32B ; les petits modèles [klein] créent une image en moins d'une demi-seconde sur un GPU grand public. Les images du modèle non commercial peuvent rester utilisables commercialement, selon une source : vérifiez.",
            "[dev] è il modello di punta da 32B; i piccoli modelli [klein] creano un'immagine in meno di mezzo secondo su una GPU consumer. Le immagini del modello non commerciale potrebbero comunque essere utilizzabili commercialmente, secondo una fonte: verifica.",
            "[dev] es el modelo estrella de 32B; los modelos pequeños [klein] crean una imagen en menos de medio segundo en una GPU de consumo. Las imágenes del modelo no comercial aún podrían usarse comercialmente, según una fuente: compruébalo.",
            "[dev] é o modelo de topo de 32B; os modelos pequenos [klein] criam uma imagem em menos de meio segundo numa GPU de consumo. As imagens do modelo não comercial podem ainda ser utilizáveis comercialmente, segundo uma fonte: confirme.",
            "[dev] ist das 32B-Flaggschiff; die kleinen [klein]-Modelle erzeugen auf einer Consumer-GPU ein Bild in unter einer halben Sekunde. Bilder des nichtkommerziellen Modells sind laut einer Quelle womöglich trotzdem kommerziell nutzbar: bitte prüfen.",
            "A [dev] a 32B-s zászlóshajó; a kis [klein] modellek fogyasztói GPU-n fél másodpercnél rövidebb idő alatt készítenek képet. A nem kereskedelmi modell képei egy forrás szerint kereskedelmileg is használhatók lehetnek: ellenőrizd.",
        ],
    ),
    (
        "a header (the token?) contains a line break or control character",
        [
            "un en-tête (le jeton ?) contient un saut de ligne ou un caractère de contrôle",
            "un'intestazione (il token?) contiene un'interruzione di riga o un carattere di controllo",
            "un encabezado (¿el token?) contiene un salto de línea o un carácter de control",
            "um cabeçalho (o token?) contém uma quebra de linha ou um carácter de controlo",
            "ein Header (das Token?) enthält einen Zeilenumbruch oder ein Steuerzeichen",
            "egy fejléc (a token?) sortörést vagy vezérlőkaraktert tartalmaz",
        ],
    ),
    (
        "an address with a user name or password in it is refused",
        [
            "une adresse contenant un nom d'utilisateur ou un mot de passe est refusée",
            "un indirizzo con nome utente o password al suo interno viene rifiutato",
            "se rechaza una dirección que contenga nombre de usuario o contraseña",
            "um endereço com nome de utilizador ou palavra-passe é recusado",
            "eine Adresse mit Benutzername oder Passwort wird abgelehnt",
            "a felhasználónevet vagy jelszót tartalmazó címet elutasítjuk",
        ],
    ),
    (
        "not a list",
        [
            "ce n'est pas une liste",
            "non è un elenco",
            "no es una lista",
            "não é uma lista",
            "keine Liste",
            "nem lista",
        ],
    ),
    (
        "plain http is only allowed for servers on this computer or network",
        [
            "le http simple n'est autorisé que pour les serveurs de cet ordinateur ou de ce réseau",
            "l'http semplice è consentito solo per server su questo computer o rete",
            "el http simple solo se permite para servidores de este ordenador o red",
            "o http simples só é permitido para servidores deste computador ou rede",
            "einfaches http ist nur für Server auf diesem Computer oder Netzwerk erlaubt",
            "az egyszerű http csak ezen a gépen vagy hálózaton lévő szerverekhez engedélyezett",
        ],
    ),
    (
        "the address has no host",
        [
            "l'adresse n'a pas d'hôte",
            "l'indirizzo non ha un host",
            "la dirección no tiene host",
            "o endereço não tem anfitrião",
            "die Adresse hat keinen Host",
            "a címben nincs hoszt",
        ],
    ),
    (
        "the address must start with https://",
        [
            "l'adresse doit commencer par https://",
            "l'indirizzo deve iniziare con https://",
            "la dirección debe empezar por https://",
            "o endereço tem de começar por https://",
            "die Adresse muss mit https:// beginnen",
            "a címnek https://-sel kell kezdődnie",
        ],
    ),
    (
        "this is a file, not a folder",
        [
            "c'est un fichier, pas un dossier",
            "è un file, non una cartella",
            "es un archivo, no una carpeta",
            "é um ficheiro, não uma pasta",
            "das ist eine Datei, kein Ordner",
            "ez egy fájl, nem mappa",
        ],
    ),
    (
        "this is not a text page",
        [
            "ce n'est pas une page de texte",
            "non è una pagina di testo",
            "no es una página de texto",
            "não é uma página de texto",
            "das ist keine Textseite",
            "ez nem szöveges oldal",
        ],
    ),
    (
        "Model duels",
        [
            "Duels de modèles",
            "Duelli tra modelli",
            "Duelos de modelos",
            "Duelos de modelos",
            "Modell-Duelle",
            "Modellpárbajok",
        ],
    ),
    (
        "Arena",
        [
            "Arène",
            "Arena",
            "Arena",
            "Arena",
            "Arena",
            "Aréna",
        ],
    ),
    (
        "Leaderboard",
        [
            "Classement",
            "Classifica",
            "Clasificación",
            "Classificação",
            "Rangliste",
            "Ranglista",
        ],
    ),
    (
        "The same question goes to two anonymous models. You vote for the better answer and only then see who was who. Everything stays on this computer; the models can only be on this computer or on your network.",
        [
            "La même question va à deux modèles anonymes. Vous votez pour la meilleure réponse et ce n'est qu'ensuite que vous voyez qui était qui. Tout reste sur cet ordinateur ; les modèles ne peuvent être que sur cet ordinateur ou sur votre réseau.",
            "La stessa domanda va a due modelli anonimi. Voti la risposta migliore e solo dopo vedi chi era chi. Tutto resta su questo computer; i modelli possono essere solo su questo computer o sulla tua rete.",
            "La misma pregunta va a dos modelos anónimos. Votas por la mejor respuesta y solo después ves quién era quién. Todo se queda en este ordenador; los modelos solo pueden estar en este ordenador o en tu red.",
            "A mesma pergunta vai para dois modelos anónimos. Votas na melhor resposta e só depois vês quem era quem. Tudo fica neste computador; os modelos só podem estar neste computador ou na tua rede.",
            "Dieselbe Frage geht an zwei anonyme Modelle. Du stimmst für die bessere Antwort und siehst erst danach, wer wer war. Alles bleibt auf diesem Computer; die Modelle dürfen nur auf diesem Computer oder in deinem Netzwerk laufen.",
            "Ugyanaz a kérdés két névtelen modellhez megy. A jobb válaszra szavazol, és csak utána látod, ki kicsoda. Minden ezen a gépen marad; a modellek csak ezen a gépen vagy a saját hálózatodon lehetnek.",
        ],
    ),
    (
        "You need at least two active models: add them in the “Models” tab.",
        [
            "Il faut au moins deux modèles actifs : ajoutez-les dans l'onglet « Modèles ».",
            "Servono almeno due modelli attivi: aggiungili nella scheda «Modelli».",
            "Necesitas al menos dos modelos activos: añádelos en la pestaña «Modelos».",
            "Precisas de pelo menos dois modelos ativos: adiciona-os no separador «Modelos».",
            "Du brauchst mindestens zwei aktive Modelle: Füge sie im Tab „Modelle“ hinzu.",
            "Legalább két aktív modell kell: add hozzá őket a „Modellek” fülön.",
        ],
    ),
    (
        "Go to Models",
        [
            "Aller aux modèles",
            "Vai ai modelli",
            "Ir a Modelos",
            "Ir para Modelos",
            "Zu den Modellen",
            "Ugrás a modellekhez",
        ],
    ),
    (
        "Write the question…",
        [
            "Écrivez la question…",
            "Scrivi la domanda…",
            "Escribe la pregunta…",
            "Escreve a pergunta…",
            "Schreibe die Frage…",
            "Írd meg a kérdést…",
        ],
    ),
    (
        "Load the sample questions",
        [
            "Charger les questions d'exemple",
            "Carica le domande di esempio",
            "Cargar las preguntas de ejemplo",
            "Carregar as perguntas de exemplo",
            "Beispielfragen laden",
            "Mintakérdések betöltése",
        ],
    ),
    (
        "Next question from the set",
        [
            "Question suivante du lot",
            "Prossima domanda del set",
            "Siguiente pregunta del conjunto",
            "Pergunta seguinte do conjunto",
            "Nächste Frage aus dem Set",
            "Következő kérdés a készletből",
        ],
    ),
    (
        "Start the duel",
        [
            "Lancer le duel",
            "Avvia il duello",
            "Iniciar el duelo",
            "Iniciar o duelo",
            "Duell starten",
            "Párbaj indítása",
        ],
    ),
    (
        "My own set of questions",
        [
            "Mon propre lot de questions",
            "Il mio set di domande",
            "Mi propio conjunto de preguntas",
            "O meu conjunto de perguntas",
            "Mein eigenes Fragenset",
            "Saját kérdéskészletem",
        ],
    ),
    (
        "One question per line, or “category | question”",
        [
            "Une question par ligne, ou « catégorie | question »",
            "Una domanda per riga, oppure «categoria | domanda»",
            "Una pregunta por línea, o «categoría | pregunta»",
            "Uma pergunta por linha, ou «categoria | pergunta»",
            "Eine Frage pro Zeile oder „Kategorie | Frage“",
            "Soronként egy kérdés, vagy „kategória | kérdés”",
        ],
    ),
    (
        "Use these questions",
        [
            "Utiliser ces questions",
            "Usa queste domande",
            "Usar estas preguntas",
            "Usar estas perguntas",
            "Diese Fragen verwenden",
            "Ezek a kérdések legyenek",
        ],
    ),
    (
        "1 · A is better",
        [
            "1 · A est meilleur",
            "1 · A è meglio",
            "1 · A es mejor",
            "1 · A é melhor",
            "1 · A ist besser",
            "1 · A jobb",
        ],
    ),
    (
        "2 · B is better",
        [
            "2 · B est meilleur",
            "2 · B è meglio",
            "2 · B es mejor",
            "2 · B é melhor",
            "2 · B ist besser",
            "2 · B jobb",
        ],
    ),
    (
        "3 · A tie",
        [
            "3 · Égalité",
            "3 · Pareggio",
            "3 · Empate",
            "3 · Empate",
            "3 · Unentschieden",
            "3 · Döntetlen",
        ],
    ),
    (
        "4 · Both are bad",
        [
            "4 · Les deux sont mauvais",
            "4 · Entrambi scarsi",
            "4 · Ambos son malos",
            "4 · Ambos são maus",
            "4 · Beide sind schlecht",
            "4 · Mindkettő gyenge",
        ],
    ),
    (
        "You chose",
        [
            "Vous avez choisi",
            "Hai scelto",
            "Elegiste",
            "Escolheste",
            "Du hast gewählt",
            "Ezt választottad",
        ],
    ),
    (
        "A tie",
        [
            "Égalité",
            "Pareggio",
            "Empate",
            "Empate",
            "Unentschieden",
            "Döntetlen",
        ],
    ),
    (
        "Both are bad",
        [
            "Les deux sont mauvais",
            "Entrambi scarsi",
            "Ambos son malos",
            "Ambos são maus",
            "Beide sind schlecht",
            "Mindkettő gyenge",
        ],
    ),
    (
        "All categories",
        [
            "Toutes les catégories",
            "Tutte le categorie",
            "Todas las categorías",
            "Todas as categorias",
            "Alle Kategorien",
            "Minden kategória",
        ],
    ),
    (
        "Export CSV",
        [
            "Exporter en CSV",
            "Esporta CSV",
            "Exportar CSV",
            "Exportar CSV",
            "CSV exportieren",
            "CSV exportálása",
        ],
    ),
    (
        "Export Markdown",
        [
            "Exporter en Markdown",
            "Esporta Markdown",
            "Exportar Markdown",
            "Exportar Markdown",
            "Markdown exportieren",
            "Markdown exportálása",
        ],
    ),
    (
        "Export the votes (JSONL)",
        [
            "Exporter les votes (JSONL)",
            "Esporta i voti (JSONL)",
            "Exportar los votos (JSONL)",
            "Exportar os votos (JSONL)",
            "Stimmen exportieren (JSONL)",
            "Szavazatok exportálása (JSONL)",
        ],
    ),
    (
        "Keep the question text in the history",
        [
            "Garder le texte des questions dans l'historique",
            "Conserva il testo delle domande nella cronologia",
            "Guardar el texto de las preguntas en el historial",
            "Guardar o texto das perguntas no histórico",
            "Fragetext im Verlauf behalten",
            "A kérdések szövegének megőrzése az előzményekben",
        ],
    ),
    (
        "There are no votes yet. Run a duel in the “Arena” tab.",
        [
            "Il n'y a pas encore de vote. Lancez un duel dans l'onglet « Arène ».",
            "Non ci sono ancora voti. Fai un duello nella scheda «Arena».",
            "Todavía no hay votos. Haz un duelo en la pestaña «Arena».",
            "Ainda não há votos. Faz um duelo no separador «Arena».",
            "Es gibt noch keine Stimmen. Starte ein Duell im Tab „Arena“.",
            "Még nincs szavazat. Indíts egy párbajt az „Aréna” fülön.",
        ],
    ),
    (
        "Rank",
        [
            "Rang",
            "Posizione",
            "Puesto",
            "Posição",
            "Platz",
            "Helyezés",
        ],
    ),
    (
        "Model",
        [
            "Modèle",
            "Modello",
            "Modelo",
            "Modelo",
            "Modell",
            "Modell",
        ],
    ),
    (
        "Rating",
        [
            "Score",
            "Punteggio",
            "Puntuación",
            "Pontuação",
            "Wertung",
            "Pontszám",
        ],
    ),
    (
        "Duels",
        [
            "Duels",
            "Duelli",
            "Duelos",
            "Duelos",
            "Duelle",
            "Párbajok",
        ],
    ),
    (
        "Wins",
        [
            "Victoires",
            "Vittorie",
            "Victorias",
            "Vitórias",
            "Siege",
            "Győzelmek",
        ],
    ),
    (
        "Draws",
        [
            "Égalités",
            "Pareggi",
            "Empates",
            "Empates",
            "Unentschieden",
            "Döntetlenek",
        ],
    ),
    (
        "Losses",
        [
            "Défaites",
            "Sconfitte",
            "Derrotas",
            "Derrotas",
            "Niederlagen",
            "Vereségek",
        ],
    ),
    (
        "Ratings start at 1000 and are recomputed from the history (Elo, K = 32). With few duels, small differences mean nothing.",
        [
            "Les scores partent de 1000 et sont recalculés à partir de l'historique (Elo, K = 32). Avec peu de duels, les petits écarts ne signifient rien.",
            "I punteggi partono da 1000 e si ricalcolano dalla cronologia (Elo, K = 32). Con pochi duelli, le piccole differenze non significano nulla.",
            "Las puntuaciones parten de 1000 y se recalculan desde el historial (Elo, K = 32). Con pocos duelos, las diferencias pequeñas no significan nada.",
            "As pontuações partem de 1000 e são recalculadas a partir do histórico (Elo, K = 32). Com poucos duelos, as diferenças pequenas não significam nada.",
            "Die Wertungen beginnen bei 1000 und werden aus dem Verlauf neu berechnet (Elo, K = 32). Bei wenigen Duellen bedeuten kleine Unterschiede nichts.",
            "A pontszámok 1000-ről indulnak, és az előzményekből számolódnak újra (Elo, K = 32). Kevés párbajnál a kis különbségek semmit sem jelentenek.",
        ],
    ),
    (
        "Any server that speaks the OpenAI chat protocol, on this computer or on your network (colibri, Ollama, LM Studio, llama.cpp…). The address starts with http://.",
        [
            "Tout serveur parlant le protocole de chat OpenAI, sur cet ordinateur ou sur votre réseau (colibri, Ollama, LM Studio, llama.cpp…). L'adresse commence par http://.",
            "Qualsiasi server che parli il protocollo chat OpenAI, su questo computer o sulla tua rete (colibri, Ollama, LM Studio, llama.cpp…). L'indirizzo inizia con http://.",
            "Cualquier servidor que hable el protocolo de chat de OpenAI, en este ordenador o en tu red (colibri, Ollama, LM Studio, llama.cpp…). La dirección empieza por http://.",
            "Qualquer servidor que fale o protocolo de chat da OpenAI, neste computador ou na tua rede (colibri, Ollama, LM Studio, llama.cpp…). O endereço começa por http://.",
            "Jeder Server, der das OpenAI-Chatprotokoll spricht, auf diesem Computer oder in deinem Netzwerk (colibri, Ollama, LM Studio, llama.cpp…). Die Adresse beginnt mit http://.",
            "Bármely, az OpenAI csevegőprotokollt beszélő szerver ezen a gépen vagy a hálózatodon (colibri, Ollama, LM Studio, llama.cpp…). A cím http://-sel kezdődik.",
        ],
    ),
    (
        "Remove",
        [
            "Retirer",
            "Rimuovi",
            "Quitar",
            "Remover",
            "Entfernen",
            "Eltávolítás",
        ],
    ),
    (
        "Add a model",
        [
            "Ajouter un modèle",
            "Aggiungi un modello",
            "Añadir un modelo",
            "Adicionar um modelo",
            "Modell hinzufügen",
            "Modell hozzáadása",
        ],
    ),
    (
        "Display name",
        [
            "Nom affiché",
            "Nome visualizzato",
            "Nombre mostrado",
            "Nome apresentado",
            "Anzeigename",
            "Megjelenített név",
        ],
    ),
    (
        "Server address, e.g. http://127.0.0.1:8000",
        [
            "Adresse du serveur, p. ex. http://127.0.0.1:8000",
            "Indirizzo del server, ad es. http://127.0.0.1:8000",
            "Dirección del servidor, p. ej. http://127.0.0.1:8000",
            "Endereço do servidor, p. ex. http://127.0.0.1:8000",
            "Serveradresse, z. B. http://127.0.0.1:8000",
            "A szerver címe, pl. http://127.0.0.1:8000",
        ],
    ),
    (
        "Model name",
        [
            "Nom du modèle",
            "Nome del modello",
            "Nombre del modelo",
            "Nome do modelo",
            "Modellname",
            "A modell neve",
        ],
    ),
    (
        "Add the model running now",
        [
            "Ajouter le modèle en cours d'exécution",
            "Aggiungi il modello in esecuzione ora",
            "Añadir el modelo que se ejecuta ahora",
            "Adicionar o modelo em execução agora",
            "Das jetzt laufende Modell hinzufügen",
            "A most futó modell hozzáadása",
        ],
    ),
    (
        "The address is not valid: only http:// servers on this computer or your own network.",
        [
            "L'adresse n'est pas valide : uniquement des serveurs http:// sur cet ordinateur ou votre réseau.",
            "L'indirizzo non è valido: solo server http:// su questo computer o sulla tua rete.",
            "La dirección no es válida: solo servidores http:// de este ordenador o de tu red.",
            "O endereço não é válido: apenas servidores http:// neste computador ou na tua rede.",
            "Die Adresse ist ungültig: nur http://-Server auf diesem Computer oder in deinem Netzwerk.",
            "A cím érvénytelen: csak ezen a gépen vagy a saját hálózatodon lévő http:// szerverek.",
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

/// The ready-made questions for comparing models, in the given language: Romanian and
/// Hungarian have their own sets; the other languages translate the English set line by line.
pub fn demo_set(lang: Lang) -> String {
    use companion_core::duel;
    match lang {
        Lang::Ro => return duel::demo_set("ro").to_string(),
        Lang::Hu => return duel::demo_set("hu").to_string(),
        Lang::En => return duel::demo_set("en").to_string(),
        _ => {}
    }
    let tr = |en: &str| lookup(lang, en).unwrap_or(en).to_string();
    duel::demo_set("en")
        .lines()
        .map(|line| match line.split_once(" | ") {
            Some((c, q)) => format!("{} | {}", tr(c.trim()), tr(q.trim())),
            None => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
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
    (
        "not started: {}",
        [
            "nu a pornit: {}",
            "non démarrée : {}",
            "non avviata: {}",
            "no se inició: {}",
            "não iniciada: {}",
            "nicht gestartet: {}",
            "nem indult el: {}",
        ],
    ),
    (
        "stopped by the person",
        [
            "oprită de persoană",
            "arrêtée par la personne",
            "fermata dalla persona",
            "detenida por la persona",
            "parada pela pessoa",
            "von der Person gestoppt",
            "a felhasználó leállította",
        ],
    ),
    (
        "step {}/{}: fetch {}",
        [
            "pasul {}/{}: preia {}",
            "étape {}/{} : récupérer {}",
            "passo {}/{}: scarica {}",
            "paso {}/{}: obtener {}",
            "passo {}/{}: obter {}",
            "Schritt {}/{}: {} abrufen",
            "{}/{}. lépés: letöltés: {}",
        ],
    ),
    (
        "step {}/{}: read folder {}",
        [
            "pasul {}/{}: citește dosarul {}",
            "étape {}/{} : lire le dossier {}",
            "passo {}/{}: leggi la cartella {}",
            "paso {}/{}: leer la carpeta {}",
            "passo {}/{}: ler a pasta {}",
            "Schritt {}/{}: Ordner {} lesen",
            "{}/{}. lépés: mappa olvasása: {}",
        ],
    ),
    (
        "step {}/{}: read file {}",
        [
            "pasul {}/{}: citește fișierul {}",
            "étape {}/{} : lire le fichier {}",
            "passo {}/{}: leggi il file {}",
            "paso {}/{}: leer el archivo {}",
            "passo {}/{}: ler o ficheiro {}",
            "Schritt {}/{}: Datei {} lesen",
            "{}/{}. lépés: fájl olvasása: {}",
        ],
    ),
    (
        "step {}/{}: repository file {}",
        [
            "pasul {}/{}: fișier din depozit {}",
            "étape {}/{} : fichier du dépôt {}",
            "passo {}/{}: file del repository {}",
            "paso {}/{}: archivo del repositorio {}",
            "passo {}/{}: ficheiro do repositório {}",
            "Schritt {}/{}: Repository-Datei {}",
            "{}/{}. lépés: tárolófájl: {}",
        ],
    ),
    (
        "step {}/{}: ask the model: {}",
        [
            "pasul {}/{}: întreabă modelul: {}",
            "étape {}/{} : demander au modèle : {}",
            "passo {}/{}: chiedi al modello: {}",
            "paso {}/{}: preguntar al modelo: {}",
            "passo {}/{}: perguntar ao modelo: {}",
            "Schritt {}/{}: Modell fragen: {}",
            "{}/{}. lépés: kérdés a modellhez: {}",
        ],
    ),
    (
        "step {}/{}: save as {}",
        [
            "pasul {}/{}: salvează ca {}",
            "étape {}/{} : enregistrer sous {}",
            "passo {}/{}: salva come {}",
            "paso {}/{}: guardar como {}",
            "passo {}/{}: guardar como {}",
            "Schritt {}/{}: speichern als {}",
            "{}/{}. lépés: mentés ezen a néven: {}",
        ],
    ),
    (
        "step {}/{}: send to {}",
        [
            "pasul {}/{}: trimite la {}",
            "étape {}/{} : envoyer à {}",
            "passo {}/{}: invia a {}",
            "paso {}/{}: enviar a {}",
            "passo {}/{}: enviar para {}",
            "Schritt {}/{}: senden an {}",
            "{}/{}. lépés: küldés ide: {}",
        ],
    ),
    (
        "step {}/{}: show the result",
        [
            "pasul {}/{}: arată rezultatul",
            "étape {}/{} : afficher le résultat",
            "passo {}/{}: mostra il risultato",
            "paso {}/{}: mostrar el resultado",
            "passo {}/{}: mostrar o resultado",
            "Schritt {}/{}: Ergebnis anzeigen",
            "{}/{}. lépés: az eredmény megjelenítése",
        ],
    ),
    (
        "step {} failed: {}",
        [
            "pasul {} a eșuat: {}",
            "l'étape {} a échoué : {}",
            "il passo {} è fallito: {}",
            "el paso {} falló: {}",
            "o passo {} falhou: {}",
            "Schritt {} ist fehlgeschlagen: {}",
            "a(z) {}. lépés nem sikerült: {}",
        ],
    ),
    (
        "the limit of {} model calls is reached",
        [
            "limita de {} apeluri către model a fost atinsă",
            "la limite de {} appels au modèle est atteinte",
            "il limite di {} chiamate al modello è stato raggiunto",
            "se alcanzó el límite de {} llamadas al modelo",
            "o limite de {} chamadas ao modelo foi atingido",
            "das Limit von {} Modellaufrufen ist erreicht",
            "elérte a(z) {} modellhívásos korlátot",
        ],
    ),
    (
        "the model answered nothing",
        [
            "modelul nu a răspuns nimic",
            "le modèle n'a rien répondu",
            "il modello non ha risposto nulla",
            "el modelo no respondió nada",
            "o modelo não respondeu nada",
            "das Modell hat nichts geantwortet",
            "a modell nem válaszolt semmit",
        ],
    ),
    (
        "model call failed: {}",
        [
            "apelul către model a eșuat: {}",
            "l'appel au modèle a échoué : {}",
            "la chiamata al modello è fallita: {}",
            "la llamada al modelo falló: {}",
            "a chamada ao modelo falhou: {}",
            "Modellaufruf fehlgeschlagen: {}",
            "a modellhívás nem sikerült: {}",
        ],
    ),
    (
        "the model did not answer after {} retries",
        [
            "modelul nu a răspuns după {} reîncercări",
            "le modèle n'a pas répondu après {} nouvelles tentatives",
            "il modello non ha risposto dopo {} tentativi",
            "el modelo no respondió tras {} reintentos",
            "o modelo não respondeu após {} novas tentativas",
            "das Modell hat nach {} Wiederholungen nicht geantwortet",
            "a modell {} újrapróbálkozás után sem válaszolt",
        ],
    ),
    (
        "retry {}/{}",
        [
            "reîncercare {}/{}",
            "nouvelle tentative {}/{}",
            "nuovo tentativo {}/{}",
            "reintento {}/{}",
            "nova tentativa {}/{}",
            "Wiederholung {}/{}",
            "újrapróbálkozás {}/{}",
        ],
    ),
    (
        "not a usable file name: {}",
        [
            "nu e un nume de fișier utilizabil: {}",
            "nom de fichier inutilisable : {}",
            "nome di file non utilizzabile: {}",
            "nombre de archivo no utilizable: {}",
            "nome de ficheiro não utilizável: {}",
            "kein verwendbarer Dateiname: {}",
            "nem használható fájlnév: {}",
        ],
    ),
    (
        "saved: {}",
        [
            "salvat: {}",
            "enregistré : {}",
            "salvato: {}",
            "guardado: {}",
            "guardado: {}",
            "gespeichert: {}",
            "mentve: {}",
        ],
    ),
    (
        "sending is never done without a person present; run this by hand",
        [
            "nu se trimite nimic fără o persoană de față; rulează asta manual",
            "l'envoi ne se fait jamais sans personne présente ; lancez ceci à la main",
            "l'invio non avviene mai senza una persona presente; avvialo a mano",
            "nunca se envía nada sin una persona presente; ejecútalo a mano",
            "nunca se envia nada sem uma pessoa presente; execute isto manualmente",
            "Gesendet wird nie ohne anwesende Person; starte das von Hand",
            "küldés soha nem történik jelenlévő személy nélkül; futtasd ezt kézzel",
        ],
    ),
    (
        "the address is not allowed: {}",
        [
            "adresa nu este permisă: {}",
            "l'adresse n'est pas autorisée : {}",
            "l'indirizzo non è consentito: {}",
            "la dirección no está permitida: {}",
            "o endereço não é permitido: {}",
            "die Adresse ist nicht erlaubt: {}",
            "a cím nem engedélyezett: {}",
        ],
    ),
    (
        "not sent: the person said no",
        [
            "netrimis: persoana a spus nu",
            "non envoyé : la personne a refusé",
            "non inviato: la persona ha detto no",
            "no enviado: la persona dijo que no",
            "não enviado: a pessoa disse não",
            "nicht gesendet: Die Person hat abgelehnt",
            "nem lett elküldve: a felhasználó nemet mondott",
        ],
    ),
    (
        "done",
        [
            "gata",
            "terminé",
            "fatto",
            "hecho",
            "concluído",
            "fertig",
            "kész",
        ],
    ),
    (
        "the automation has no name",
        [
            "automatizarea nu are nume",
            "l'automatisation n'a pas de nom",
            "l'automazione non ha un nome",
            "la automatización no tiene nombre",
            "a automação não tem nome",
            "die Automatisierung hat keinen Namen",
            "az automatizálásnak nincs neve",
        ],
    ),
    (
        "the automation has no steps",
        [
            "automatizarea nu are pași",
            "l'automatisation n'a aucune étape",
            "l'automazione non ha passaggi",
            "la automatización no tiene pasos",
            "a automação não tem passos",
            "die Automatisierung hat keine Schritte",
            "az automatizálásnak nincsenek lépései",
        ],
    ),
    (
        "at most {} steps",
        [
            "cel mult {} pași",
            "au plus {} étapes",
            "al massimo {} passaggi",
            "como máximo {} pasos",
            "no máximo {} passos",
            "höchstens {} Schritte",
            "legfeljebb {} lépés",
        ],
    ),
    (
        "at most {} model calls",
        [
            "cel mult {} apeluri către model",
            "au plus {} appels au modèle",
            "al massimo {} chiamate al modello",
            "como máximo {} llamadas al modelo",
            "no máximo {} chamadas ao modelo",
            "höchstens {} Modellaufrufe",
            "legfeljebb {} modellhívás",
        ],
    ),
    (
        "step {}: no path",
        [
            "pasul {}: nicio cale",
            "étape {} : aucun chemin",
            "passo {}: nessun percorso",
            "paso {}: ninguna ruta",
            "passo {}: nenhum caminho",
            "Schritt {}: kein Pfad",
            "{}. lépés: nincs útvonal",
        ],
    ),
    (
        "step {}: empty question",
        [
            "pasul {}: întrebare goală",
            "étape {} : question vide",
            "passo {}: domanda vuota",
            "paso {}: pregunta vacía",
            "passo {}: pergunta vazia",
            "Schritt {}: leere Frage",
            "{}. lépés: üres kérdés",
        ],
    ),
    (
        "step {}: not a usable file name",
        [
            "pasul {}: nu e un nume de fișier utilizabil",
            "étape {} : nom de fichier inutilisable",
            "passo {}: nome di file non utilizzabile",
            "paso {}: nombre de archivo no utilizable",
            "passo {}: nome de ficheiro não utilizável",
            "Schritt {}: kein verwendbarer Dateiname",
            "{}. lépés: nem használható fájlnév",
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
            include_str!("models_ui.rs"),
            include_str!("resources_ui.rs"),
            include_str!("guide_ui.rs"),
            include_str!("prompts_ui.rs"),
            include_str!("online_ui.rs"),
            include_str!("studio_ui.rs"),
            include_str!("duel_ui.rs"),
            include_str!("access_ui.rs"),
            include_str!("flows_ui.rs"),
            include_str!("memory_ui.rs"),
            include_str!("sheets_ui.rs"),
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

    /// Every English text that the core hands over (platforms, prompts, catalogue, automations,
    /// connection errors, demo prompts), so none is left in two languages only.
    fn core_texts() -> Vec<String> {
        use companion_core::{connect, duel, flows, platforms, prompts, studio};
        let mut v: Vec<String> = Vec::new();
        for p in platforms::ALL {
            v.push(p.en.to_string());
        }
        for l in platforms::Level::ALL {
            v.push(platforms::level_text(l, false).to_string());
        }
        for m in studio::ALL {
            v.push(m.en.to_string());
        }
        for r in studio::RUNNERS {
            v.push(r.en.to_string());
        }
        for p in prompts::builtin(false) {
            v.push(p.category);
            v.push(p.title);
            v.push(p.text);
        }
        for f in flows::templates(false) {
            v.push(f.name.clone());
            for st in &f.steps {
                if let flows::Step::Ask { prompt } = st {
                    v.push(prompt.clone());
                }
            }
        }
        v.extend(connect::EXPLAIN_KEYS.iter().map(|s| s.to_string()));
        v.extend(connect::DETAIL_KEYS.iter().map(|s| s.to_string()));
        for line in duel::demo_set("en").lines() {
            if let Some((c, q)) = line.split_once(" | ") {
                v.push(c.trim().to_string());
                v.push(q.trim().to_string());
            }
        }
        v.sort();
        v.dedup();
        v.retain(|s| !s.trim().is_empty());
        v
    }

    #[test]
    fn core_texts_are_translated() {
        let all = core_texts();
        assert!(
            all.len() > 200,
            "{} texts; the collector is broken",
            all.len()
        );
        let mut missing: Vec<&String> = Vec::new();
        for t in &all {
            if lookup(Lang::Fr, t).is_none() {
                missing.push(t);
            }
        }
        if let Ok(path) = std::env::var("DUMP_MISSING") {
            std::fs::write(path, serde_json::to_string_pretty(&missing).unwrap()).unwrap();
        }
        assert!(
            missing.is_empty(),
            "{} untranslated core texts, first: {:?}",
            missing.len(),
            missing.iter().take(3).collect::<Vec<_>>()
        );
    }

    #[test]
    fn demo_sets_and_journal_lines_follow_the_language() {
        let fr = demo_set(Lang::Fr);
        assert!(fr.contains("Raisonnement | "), "{fr}");
        assert_eq!(
            fr.lines().count(),
            companion_core::duel::demo_set("en").lines().count()
        );
        assert!(demo_set(Lang::Ro).contains("Cunoștințe"));
        assert_eq!(
            engine_text(Lang::De, "step 2/5: ask the model: Hello"),
            "Schritt 2/5: Modell fragen: Hello"
        );
        assert_eq!(engine_text(Lang::Hu, "done"), "kész");
        assert_eq!(engine_text(Lang::Es, "retry 2/3"), "reintento 2/3");
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
