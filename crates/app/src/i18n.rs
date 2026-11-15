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
            include_str!("models_ui.rs"),
            include_str!("resources_ui.rs"),
            include_str!("guide_ui.rs"),
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
