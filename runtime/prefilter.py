#!/usr/bin/env python3
"""
prefilter.py — selezione dei candidati executor (Metnos v1.1 POC).

Implementazione bag-of-words sull'`affinity` dichiarato nei manifest. La forma
con embedding MiniLM (decisa in F1=b) e' rimandata a una iterazione successiva
del POC: il bag-of-words e' un placeholder funzionante per validare la forma
del flusso "user query -> top-K -> LLM con catalogo ristretto".

Bootstrap statico (F5=c): solo affinity tags day-1; quando il mnestoma esistera'
il punteggio sara' affinity_match + history_boost.

K adattivo (deciso 26/4/2026 sera dopo stress test D-tools):
    Quando il prefilter ha alta confidenza (top-1 nettamente sopra gli altri),
    K si abbassa a K_min. Quando ha bassa confidenza (score molto vicini fra
    primi e successivi), K si alza fino a K_max. Razionale: ottimizzare il
    trade-off recall/precision contro la latenza/distrazione del LLM.

API:
    rank(query, catalog, k=10) -> list[Executor]                     (legacy)
    rank_adaptive(query, catalog, k_min=5, k_max=40) -> (list, info) (preferita)
"""
import re

from logging_setup import get_logger
log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+", re.UNICODE)


def tokenize(text):
    return set(_WORD_RE.findall((text or "").lower()))


# Mappa verbi IT/EN (forme coniugate) → verbo canonico del vocabolario chiuso.
# Usato per identificare l'intent della query e dare un boost forte agli
# executor il cui nome inizia con quel verbo.
_VERB_TO_CANONICAL = {
    # move
    "sposta": "move", "sposto": "move", "spostare": "move", "sposti": "move",
    "muovi": "move", "muovo": "move", "muovere": "move", "muove": "move",
    "trasferisci": "move", "trasferisco": "move", "trasferire": "move",
    "rinomina": "move", "rinomino": "move", "rinominare": "move",
    "move": "move", "moves": "move", "rename": "move", "rn": "move", "mv": "move",
    "transfer": "move",
    # delete
    "cancella": "delete", "cancello": "delete", "cancellare": "delete",
    "elimina": "delete", "elimino": "delete", "eliminare": "delete",
    "rimuovi": "delete", "rimuovo": "delete", "rimuovere": "delete",
    "delete": "delete", "remove": "delete", "rm": "delete",
    "drop": "delete",
    # NB: niente `del` (preposizione articolata IT in "del sistema/del server"
    # confondeva il verb-boost). Bug live 15/5/2026: "mostrami il quadro
    # del sistema" → boost +10 a tutti i delete_*. Falso positivo.
    # read
    "leggi": "read", "leggo": "read", "leggere": "read", "letto": "read",
    "mostra": "read", "mostro": "read", "mostrami": "read", "mostrare": "read",
    "controlla": "read", "controllo": "read", "controllare": "read",
    "apri": "read", "apro": "read", "aprire": "read",
    "read": "read", "open": "read", "show": "read", "view": "read", "cat": "read",
    "check": "read",
    # write
    "scrivi": "write", "scrivo": "write", "scrivere": "write",
    "salva": "write", "salvo": "write", "salvare": "write",
    "metti": "write", "metto": "write", "mettere": "write",
    "put": "write", "place": "write",
    "write": "write", "save": "write",
    # find
    "trova": "find", "trovo": "find", "trovare": "find",
    "cerca": "find", "cerco": "find", "cercare": "find",
    "find": "find", "search": "find", "locate": "find", "where": "find",
    # list
    "elenca": "list", "elenco": "list", "elencare": "list",
    "lista": "list", "list": "list", "ls": "list",
    # create
    "crea": "create", "creo": "create", "creare": "create",
    "create": "create", "mkdir": "create", "make": "create", "new": "create",
    # send (canonici + enclitici IT + sinonimi EN)
    "invia": "send", "invio": "send", "inviare": "send",
    "inviami": "send", "inviagli": "send",  # enclitico IT (oggetto indiretto)
    "manda": "send", "mando": "send", "mandare": "send",
    "mandami": "send", "mandagli": "send",
    "spedisci": "send", "spedisco": "send", "spedire": "send",
    "spediscimi": "send",
    "scrivimi": "send", "scrivigli": "send",  # write+to_me semantica = send
    "notifica": "send", "notificami": "send", "notificagli": "send",
    "avvisa": "send", "avvisami": "send", "avvisagli": "send",
    "send": "send", "notify": "send", "alert": "send", "ping": "send",
    # NB: `email`/`mail`/`text`/`message` esclusi qui — sono piu' spesso
    # nomi che verbi nelle query naturali IT+EN (es. «email e telefono di X»).
    # Quando l'utente vuole l'azione, usa send/invia/manda/notify esplicito.
    # set (update events labels, contacts, signatures, persons, credentials).
    # "fissa"/"prenota" sono i verbi idiomatici per booking di un evento
    # del calendario in IT; "book"/"schedule" in EN. Post ADR 0128 (12/5/2026):
    # l'executor canonico Google Workspace per la creazione e' `create_events`
    # (NON piu' `set_events`). Quindi il verbo semantico e' `create` (terminal
    # resource creation), distinto da `set` (idempotent state upsert).
    "fissa": "create", "fisso": "create", "fissare": "create",
    "prenota": "create", "prenoto": "create", "prenotare": "create",
    "book": "create", "schedule": "create", "schedules": "create",
    # http get (verbo `fetch` rimosso 3/5/2026: HTTP GET = `get_urls`)
    "scarica": "get", "scarico": "get", "scaricare": "get",
    "fetch": "get", "download": "get", "wget": "get", "curl": "get",
    # compress
    "comprimi": "compress", "comprimo": "compress", "comprimere": "compress",
    "zippa": "compress", "compress": "compress", "zip": "compress", "tar": "compress",
    # extract
    "estrai": "extract", "estraggo": "extract", "estrarre": "extract",
    "extract": "extract", "unzip": "extract", "untar": "extract",
    # change — forma/parametri (resize/convert/rotate/crop, vocab §2.2).
    # Senza queste entry, _all_query_verbs_satisfied non riconosce «converti
    # X in formato Y» come mutating, auto-final transformative non triggera
    # e il PLANNER ri-emette la stessa call (loop fino al duplicate-detect).
    "cambia": "change", "cambio": "change", "cambiare": "change",
    "modifica": "change", "modifico": "change", "modificare": "change",
    "trasforma": "change", "trasformo": "change", "trasformare": "change",
    "converti": "change", "converto": "change", "convertire": "change",
    "ridimensiona": "change", "ridimensiono": "change", "ridimensionare": "change",
    "ruota": "change", "ruoto": "change", "ruotare": "change",
    "ritaglia": "change", "ritaglio": "change", "ritagliare": "change",
    "normalizza": "change", "normalizzo": "change", "normalizzare": "change",
    "ricodifica": "change", "ricodifico": "change", "ricodificare": "change",
    "change": "change", "modify": "change", "transform": "change",
    "convert": "change", "resize": "change", "rotate": "change",
    "crop": "change", "normalize": "change", "reformat": "change",
    "transcode": "change", "encode": "change",
    # filter
    "filtra": "filter", "filtro": "filter", "filtrare": "filter",
    "filter": "filter",
    # NB: «where» riservato a `find` (linea 63): «where is X?» = trova X
    # in EN naturale. Mai SQL-context nelle query utente.
    # describe
    "descrivi": "describe", "descrivo": "describe", "descrivere": "describe",
    "riassumi": "describe", "riassumo": "describe", "riassumere": "describe",
    "describe": "describe", "summarize": "describe", "summary": "describe",
    # describe — suggestion semantics (P5, 12/5/2026): «proponi/suggerisci/
    # raccomanda» chiedono di presentare informazione strutturata (N opzioni,
    # alternative, slot, orari). Sono READ-ONLY: NON creano/modificano niente.
    # Mappati a `describe` perche' presentano dati aggregati gia' disponibili
    # (es. slot liberi computati da read_events). Enclitici IT consistent
    # con `send` (mandami/inviami): proponimi/suggeriscimi/raccomandami.
    "proponi": "describe", "propongo": "describe", "proporre": "describe",
    "proponimi": "describe", "proponici": "describe",
    "suggerisci": "describe", "suggerisco": "describe", "suggerire": "describe",
    "suggeriscimi": "describe", "suggeriscici": "describe",
    "raccomanda": "describe", "raccomando": "describe", "raccomandare": "describe",
    "raccomandami": "describe", "raccomandaci": "describe",
    "propose": "describe", "proposes": "describe", "proposing": "describe",
    "suggest": "describe", "suggests": "describe", "suggesting": "describe",
    "recommend": "describe", "recommends": "describe", "recommending": "describe",
    # get (enrichment)
    "arricchisci": "get", "arricchisco": "get", "arricchire": "get",
    "ottieni": "get", "ottengo": "get", "ottenere": "get",
    "get": "get", "give": "get", "tell": "get",
    "dimmi": "get", "dammi": "get",  # spesso enrichment-like
    # compute
    "calcola": "compute", "calcolo": "compute", "calcolare": "compute",
    "compute": "compute", "calculate": "compute", "hash": "compute",
}

# Verbi-superficie polisemici per CONTENITORE (§7.3, fix 3/6/2026). Il
# vocabolario context-free _VERB_TO_CANONICAL e' 1:1 e quindi LOSSY: "metti/
# salva" -> "write" soltanto, cosi' i producer `create_*` perdono il verb-boost
# e il planner sceglie write inventando un path (bug spreadsheet 2-3/6). Qui
# dichiariamo i SIBLING di lifecycle di un canonical: oltre al primario,
# surfacciamo anche questi producer (recall pieno) e lasciamo decidere ai
# layer a valle (write=UPSERT, SCOPO manifest, verifier L6). Auto-limitante:
# il sibling boosta solo se quel producer esiste DAVVERO per l'oggetto (es.
# write_files_spreadsheet <-> create_files_spreadsheet); per oggetti con un
# solo producer (create_events, niente write_events) e' un no-op.
_VERB_ALSO_CANONICAL: dict[str, tuple[str, ...]] = {
    "write": ("create",),   # scrivi-in-esistente <-> crea-nuovo
    "create": ("write",),
}

# Boost ridotto per il sibling: entra nel pool ma sotto il primario (che resta
# preferito a parita' di object/qualifier).
_VERB_SIBLING_BOOST = 7


def detect_canonical_verb(qtokens):
    """Ritorna il primo verbo canonico (move/delete/read/...) trovato fra i
    token della query, o None. Importante per boost del prefilter."""
    for tok in sorted(qtokens):
        v = _VERB_TO_CANONICAL.get(tok)
        if v:
            return v
    return None


# Italian clitic suffixes per pronominal forms (universal §7.9):
# "mettili" = "metti" + "li", "inviamelo" = "invia" + "melo", ecc.
# Ordine matter: prima i più lunghi.
_IT_CLITIC_SUFFIXES = (
    "celo", "cela", "celi", "cele", "cene", "cisi",
    "melo", "mela", "meli", "mele", "mene",
    "telo", "tela", "teli", "tele", "tene",
    "selo", "sela", "seli", "sele", "sene",
    "gli", "mi", "ti", "si", "ci", "vi", "ne",
    "lo", "la", "li", "le",
)


def _strip_italian_clitic(tok: str) -> str | None:
    """Universal §7.9: rimuovi clitico pronome IT. Ritorna stem o None."""
    for suf in _IT_CLITIC_SUFFIXES:
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            return tok[:-len(suf)]
    return None


def detect_canonical_verbs_all(qtokens) -> list[str]:
    """Ritorna TUTTI i verbi canonici distinti trovati fra i token, in ordine
    di apparizione. Usato per detection multi-step (es. «fissa appuntamento e
    mandami email» -> ['set', 'send']). Lista vuota se nessun verbo.
    Generale: deriva dai sinonimi vocab IT+EN gia' presenti in
    `_VERB_TO_CANONICAL`, non hardcoded a un caso d'uso specifico.

    Italian clitic stripping (§7.9 universal): "mettili"→"metti", "inviamelo"
    →"invia". Cattura clitici pronominali standard IT.
    """
    seen = []
    for tok in sorted(qtokens):
        v = _VERB_TO_CANONICAL.get(tok)
        if not v:
            # Try clitic stripping (mettili → metti, inviamelo → invia)
            stem = _strip_italian_clitic(tok)
            if stem:
                v = _VERB_TO_CANONICAL.get(stem)
        if v and v not in seen:
            seen.append(v)
    return seen


# Mappa parole IT/EN -> oggetto canonico (suffisso di executor name).
# Permette di disambiguare fra `move_files` e `move_messages` quando il verbo
# si applica a entrambi.
_OBJECT_HINTS = {
    "messages": ["mail", "email", "posta", "messaggio", "messaggi", "imap",
                 "inbox", "indesiderata", "junk", "spam", "trash", "cestino",
                 "archivio", "archive", "mittente", "oggetto",
                 "destinatario", "subject", "from"],
    "files":    ["file", "files",
                 "pdf", "csv", "xlsx", "txt", "documento", "documenti"],
    # Termini semantici + estensioni image (vocabolario chiuso e stabile
    # da ~30 anni: jpg/png/heic/webp sono universalmente "image", non
    # hardcoding anti-pattern — decisione Roberto 5/5/2026).
    "images":   ["foto", "photo", "photos", "image", "images", "immagine",
                 "immagini", "picture", "pictures",
                 "jpg", "jpeg", "png", "heic", "webp"],
    "dirs":     ["directory", "directories", "cartella", "cartelle", "folder",
                 "folders", "dir", "subdir", "subfolder"],
    "location": ["location", "posizione", "dove", "luogo", "geolocation",
                 "coordinate", "lat", "lon", "longitudine", "latitudine"],
    "places":   ["place", "luogo", "luoghi", "geo", "city", "country",
                 "citta", "comune", "indirizzo"],
    "now":      ["adesso", "now", "ora", "orario", "tempo"],
    # Solo identificatori tecnici stabili e linguisticamente neutri.
    # I domini web (es. `scuola.edu.it`, `metnos.com`) sono identificati
    # strutturalmente via regex in `_detect_domain_in_query` — generale per
    # qualsiasi TLD presente o futuro. Termini come `notizie/news/articolo/
    # blog/post/pagina/feed/rss` rimossi: sostantivi specifici, language-
    # bound, misleading su sostantivi simili (es. "pagina 5 del PDF" non
    # implica object=urls). Decisione Roberto 5/5/2026.
    "urls":     ["url", "urls", "uri", "link", "https", "http"],
    "numbers":  ["numero", "numeri", "number", "numbers", "media", "stddev",
                 "minimo", "massimo", "statistica"],
    # `lines` non e' piu' un oggetto separato (3/5/2026): e' qualifier
    # di granularita' su `texts`. Sinonimi mappati su `texts`.
    "texts":    ["testo", "testi", "text", "texts", "riga", "righe",
                  "line", "lines", "log"],
    "packages": ["package", "pacchetto", "pip", "apt", "package"],
    "events":   ["evento", "eventi", "event", "calendar", "calendario",
                  "appuntamento", "appuntamenti", "appointment", "appointments",
                  "riunione", "riunioni", "meeting", "meetings",
                  "agenda", "incontro", "incontri",
                  "scadenza", "scadenze", "deadline",
                  "fissa", "prenota", "book", "schedule",
                  # P5 (12/5/2026): hint per query suggestion-style come
                  # «proponi 3 orari per appuntamento» o «slot liberi mattina».
                  # Mantieni allineato con vocab classes; "orari/fasce/slot"
                  # sono universali per il dominio calendar.
                  "orari", "orario", "fascia", "fasce", "slot", "slots",
                  "mattina", "pomeriggio", "morning", "afternoon"],
    # Calendars (3/6): il CONTENITORE-calendario. "calendario/calendar" restano
    # anche hint di `events` (ambiguo: "leggi il calendario"=eventi) — la
    # disambiguazione create-container vs evento la fa il SCOPO del manifest +
    # entrambi i producer nel pool.
    "calendars": ["calendario", "calendari", "calendar", "calendars"],
    "contacts": ["contatto", "contatti", "contact", "rubrica"],
    "processes": ["processo", "processi", "process", "processes", "ps",
                   "task", "pid", "cpu", "ram", "memoria", "memory",
                   "istanza", "istanze", "instance", "instances",
                   "running", "esecuzione", "daemon", "demone",
                   "kill", "uccidi", "termina", "stop"],
    "signatures": ["signature", "signatures", "policy", "policies",
                    "blacklist", "whitelist", "graylist", "forbidden",
                    "safety"],
    "proposals": ["proposta", "proposte", "proposal", "proposals",
                   "introvertiva", "introvertive", "introvertivo",
                   "review", "candidato", "candidati", "candidate",
                   "candidates", "dedupe", "generalize", "specialize",
                   "pending"],
}


## Domain (web) detector: regex strutturale (non hardcoded TLD list).
#
# Riconosce un dominio web in una query libera (`scuola.edu.it`,
# `repubblica.it`, `metnos.com`) senza enumerare TLD. Anti-pattern §7.3:
# se domani arriva `.health` o `.school`, niente lista da aggiornare.
#
# Pattern: `<label>.<tld>[.<tld2>]` con label alfanumerico + TLD 2-24 char
# alfabetici. Filtri di scarto:
#   - estensioni filesystem comuni → e' un path, non un dominio
#   - version-numbers (`v1.2`, `1.2.3`) → label e/o TLD numerico
#   - timestamp date (`2026.01.15`) → label numerico
#
# Caso live 5/5/2026: `cerca in scuola.edu.it organico` deve → urls;
# `leggi /tmp/file.jpg` NON deve → urls.
_FS_EXTENSIONS = frozenset({
    "txt", "md", "py", "rs", "go", "c", "h", "cpp", "hpp", "js", "ts", "tsx",
    "jsx", "java", "kt", "swift", "rb", "php", "sh", "bash", "zsh", "fish",
    "json", "yaml", "yml", "toml", "ini", "cfg", "conf", "env",
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "csv", "tsv",
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "heic", "svg",
    "mp3", "wav", "flac", "ogg", "m4a", "aac",
    "mp4", "mkv", "mov", "avi", "webm", "wmv",
    "zip", "gz", "tar", "bz2", "xz", "7z", "rar",
    "log", "bak", "tmp", "swp", "lock",
    "html", "htm", "xml", "css", "scss", "less",
    "sql", "db", "sqlite", "sqlite3",
    "pyc", "pyo", "o", "so", "dll", "dylib", "exe",
})

# Pattern: parola alfanumerica con almeno una lettera + . + label-tld.
# Negative lookbehind `(?<![/\w\-])` evita match su path tipo `/tmp/x.jpg`
# (stop dopo `/`) e su parti di identifier `foo.x.jpg`. Lookbehind `[\w\-]`
# include `_-` perche' il domain pattern non li ha mai come carattere finale.
_DOMAIN_RE = re.compile(
    r"(?<![/\w\-])"                                # not preceded by path-sep or word char
    r"(?P<label>[a-z0-9][a-z0-9\-]{0,62})"         # primary label
    r"\."
    r"(?P<tld>[a-z]{2,24})"                        # primary TLD: alfabetico puro
    r"(?:\.(?P<tld2>[a-z]{2,24}))?"                # optional ccTLD (es. ".edu.it")
    r"(?![\w\-])",                                  # not followed by word char
    re.IGNORECASE,
)


def _detect_domain_in_query(query: str) -> bool:
    """True se la query contiene una sequenza che ha forma di dominio web.

    Filtri di scarto (no false positive):
      - label puramente numerica (es. `1.2.3`, `2026.01`)
      - TLD = estensione filesystem nota (es. `file.jpg`, `notes.md`)
      - dominio 'localhost' senza punto (escluso dal pattern stesso)
    """
    if not query:
        return False
    for m in _DOMAIN_RE.finditer(query):
        label = m.group("label")
        tld = m.group("tld").lower()
        tld2 = (m.group("tld2") or "").lower()
        # filtro 1: label tutta numerica → version o date
        if label.isdigit():
            continue
        # filtro 2: TLD primario in lista estensioni filesystem
        # (`.jpg`, `.md`, `.py`, ...). Il caso `.com` / `.it` / `.edu` non
        # e' tra le estensioni e quindi passa.
        # Nota: se c'e' un secondo TLD (`.edu.it`), il primo TLD `edu` non
        # e' un'estensione → passa, e per scrupolo verifichiamo anche tld2.
        if tld in _FS_EXTENSIONS:
            continue
        if tld2 and tld2 in _FS_EXTENSIONS:
            continue
        return True
    return False


def detect_canonical_object(qtokens, query: str | None = None):
    """Ritorna l'oggetto canonico (es. 'messages', 'files') matchato nei
    token della query. Conta le occorrenze e ritorna quello con piu' hit.

    Layer 2 (5/5/2026): se la query contiene un dominio web (`scuola.edu.it`,
    `metnos.com`), object='urls' viene OVERRIDE. Generale per qualsiasi TLD,
    sostituisce la vecchia lista hardcoded di TLD in `_OBJECT_HINTS["urls"]`.
    """
    scores = {}
    for obj, hints in _OBJECT_HINTS.items():
        hit = sum(1 for h in hints if h in qtokens)
        if hit:
            scores[obj] = hit
    # Override strutturale: presenza di dominio web → urls.
    if query and _detect_domain_in_query(query):
        # Add+force: se urls non era detected via token, lo introduciamo;
        # se lo era, ne aumentiamo lo score per assicurarci che vinca il
        # tie-break con altri object eventualmente menzionati.
        scores["urls"] = scores.get("urls", 0) + 5
    if not scores:
        return None
    # Tie-break: il primo nel dict order (Python 3.7+ preserva ordine).
    return max(scores.keys(), key=lambda o: scores[o])


_STOPWORDS_IT = {
    "il","lo","la","i","gli","le","un","uno","una",
    "di","a","da","in","con","su","per","tra","fra",
    "e","o","ma","se","che","non","ne","ci","si","mi","ti","vi",
    "del","della","dei","delle","dello","degli","al","alla","ai","alle",
    "dal","dalla","dai","dalle","nel","nella","nei","nelle","sul","sulla","sui","sulle",
    "sono","è","ho","ha","hanno","essere","avere",
    "questo","quella","questi","quelle","quello",
    "miei","tuoi","suoi","mia","tua","sua",
    "oggi","ieri","domani",
    "anche","poi","cosi","cosi'","molto","piu","piu'","pero","pero'",
}
_STOPWORDS_EN = {
    "the","a","an","of","in","on","at","to","for","with","by","from",
    "and","or","but","if","not","is","are","was","were","be","been",
    "this","that","these","those","my","your","their","our",
    "today","yesterday","tomorrow",
}
_STOPWORDS = _STOPWORDS_IT | _STOPWORDS_EN


def affinity_score(query_tokens, executor, *,
                   query_canonical_verb=None, query_canonical_object=None,
                   query_raw=None):
    """Score con preferenza forte al VERBO CANONICO della query, all'oggetto
    canonico, e all'affinity (verbi/azioni dichiarati nel manifest);
    soft-match cap-ato sui token rari.

    Pesi:
    - VERB BOOST: +10 se il nome dell'executor e' `<canonical>_*` (es. query
      "sposta..." → boost a tutti i `move_*`).
    - OBJECT BOOST: +6 se il nome contiene `<canonical_object>` (suffisso o
      qualifier). Disambigua fra move_files e move_messages.
    - hard match (token query ∈ tag affinity): peso 4 per token.
    - soft match (token query ∈ tokenize(description) escluse stopwords e i
      token gia' contati come hard): peso 1, cappato a 3 (per evitare che
      description verbose dominino).

    §7.3 Task #41 (28/5/2026) — opt-in METNOS_PREFILTER_RULES=1 attiva 4 rule
    aggiuntive portate da e2e/simulator (path-promote, query-pattern, producer
    compat, rare-token penalty). Bench 446q baseline: prefilter top-1 47% →
    atteso 65-75% post-rules. Vedi runtime/prefilter_rules.py.

    Riferimento al caso live 29/4/2026: query "sposta in Posta indesiderata le
    mail" privilegiava read_messages (description ricca + affinity over-tagged)
    su move_messages. Il verb-boost+object-boost risolve.
    """
    aff_tokens = set()
    for tag in executor.affinity:
        aff_tokens.update(tokenize(tag))
    desc_tokens = tokenize(executor.description)
    hard_matches = query_tokens & aff_tokens
    hard = len(hard_matches) * 4
    soft_pool = (query_tokens & desc_tokens) - hard_matches - _STOPWORDS
    soft = min(len(soft_pool), 3)
    verb_boost = 0
    if query_canonical_verb:
        _first = executor.name.split("_", 1)[0]
        if _first == query_canonical_verb:
            verb_boost = 10
        elif _first in _VERB_ALSO_CANONICAL.get(query_canonical_verb, ()):
            verb_boost = _VERB_SIBLING_BOOST  # producer sibling di lifecycle
    object_boost = 0
    if query_canonical_object:
        # Match se l'object canonico e' parte del nome dell'executor (es.
        # "messages" in "move_messages" o "files" in "read_files_csv").
        name_parts = executor.name.split("_")
        if query_canonical_object in name_parts:
            object_boost = 6
    base = hard + soft + verb_boost + object_boost

    # §7.3 opt-in rule porting da simulator
    import os
    if os.environ.get("METNOS_PREFILTER_RULES", "0") == "1" and query_raw:
        try:
            from prefilter_rules import (compute_rule_boost,
                                          compute_rare_penalty)
            rule_boost = compute_rule_boost(
                query_raw, query_tokens, query_canonical_verb, executor)
            rare_pen = compute_rare_penalty(query_tokens, executor)
            base += rule_boost + rare_pen
        except Exception as _e:  # §2.8 no silent failure
            log.warning("prefilter_rules failed for %s: %s",
                        executor.name, _e)

    return base


def rank(query, catalog, k=10, min_score=1):
    """Forma legacy (K fisso). Usata da test esistenti."""
    catalog = _filter_dormant(catalog)
    qtokens = tokenize(query)
    if not qtokens:
        return list(catalog)[:k]
    canonical_verb = detect_canonical_verb(qtokens)
    canonical_object = detect_canonical_object(qtokens, query)
    # §7.3 Task #41: init rare-tokens cache (idempotente) per rule penalty
    import os as _os_pref
    if _os_pref.environ.get("METNOS_PREFILTER_RULES", "0") == "1":
        try:
            from prefilter_rules import init_rare_tokens
            init_rare_tokens(catalog)
        except Exception as _e:  # §2.8 no silent failure
            log.warning("init_rare_tokens failed: %s", _e)
    scored = [(affinity_score(qtokens, e,
                              query_canonical_verb=canonical_verb,
                              query_canonical_object=canonical_object,
                              query_raw=query), e)
              for e in catalog]
    scored.sort(key=lambda p: (-p[0], getattr(p[1], "name", "")))
    above = [e for s, e in scored if s >= min_score]
    if above:
        return above[:k]
    return [e for _, e in scored[:k]]


def _confidence(scores):
    """
    Misura della confidenza del prefilter, in [0, 1].
    1 = top-1 domina nettamente; 0 = scores ravvicinati o tutti zero.
    Heuristica:
        - se top-1 == 0 (nessun match): confidenza = 0
        - se top-1 > 0 e top-2 == 0: confidenza = 1 (dominio assoluto)
        - altrimenti: (top-1 - top-2) / top-1
    """
    if not scores or scores[0] == 0:
        return 0.0
    if len(scores) < 2 or scores[1] == 0:
        return 1.0
    return max(0.0, (scores[0] - scores[1]) / scores[0])


def adaptive_k(scores, k_min=5, k_max=40):
    """
    Calcola K dato il vettore di scores ordinato decrescente.

    Politica:
        confidenza alta (>= 0.7)  -> K = k_min   (top-1 chiaro, basta poco)
        confidenza media (0.3-0.7)-> K interpolato linearmente fra min e max
        confidenza bassa (< 0.3)  -> K = k_max   (scegli largo, lascia decidere al LLM)

    In aggiunta, K non puo' eccedere il numero di score >= 1 (no padding di tools
    irrilevanti); se TUTTI gli score sono zero, ritorna k_min comunque.
    """
    conf = _confidence(scores)
    if conf >= 0.7:
        K = k_min
    elif conf <= 0.3:
        K = k_max
    else:
        # interpola: conf=0.7 -> k_min, conf=0.3 -> k_max
        frac = (0.7 - conf) / 0.4
        K = int(k_min + frac * (k_max - k_min))
    n_useful = sum(1 for s in scores if s >= 1)
    if n_useful > 0:
        # Cap su n_useful: non riempire con tools a score zero (dispersivo).
        K = min(K, n_useful)
    else:
        # Nessun match: floor a k_min ma non oltre la dimensione del catalogo.
        K = min(k_min, len(scores))
    K = min(K, len(scores))
    return K, conf


# Vocabolario classificato — importato da vocab.py (single source of truth).
# Aggiungere/togliere un verbo dalle classi si fa in vocab.py.
from vocab import (
    PRECURSOR_VERBS as _PRECURSOR_VERBS,
    PRODUCER_VERBS as _PRODUCER_VERBS,
)
# Tool di manipolazione dati: utili come step intermedi in QUASI tutti i pipeline.
# Vengono sempre inclusi nei candidati se nel catalog, non aumentano il rumore
# perche' sono semanticamente neutri (filter, classify_entries era gia' synth-injected).
_PIPELINE_HELPERS = ("filter_entries",)

# Cross-tool dependencies query-driven: alcuni tool hanno una semantica che
# richiede UN ALTRO tool come precursor SOLO se la query ha un certo marker.
# Esempio: find_places funziona stand-alone per query con luogo esplicito
# ("ristoranti a Brescia"), ma per query location-relative ("vicino a me",
# "qui", "intorno") richiede get_location prima per risolvere il "me" in
# coordinate. Il PLANNER prompt §5 prescrive get_location per "DOVE-SONO",
# ma se get_location non e' nei candidati esposti il modello non puo' chiamarlo.
# Regole come tuple (consumer_name, provider_name, query_markers).
# Tool "stella" per oggetto: per ogni OBJECT del vocabolario c'e' un
# executor primario che va INCLUSO nei candidati anche se l'intent del
# turno ha picked un verbo diverso. Es. per "places" il primario e'
# find_places: vale sia per query "trova/cerca/find" sia per query
# ellittiche o senza verbo ("farmacia piu vicina") che l'intent extractor
# mappa a verbo generico (get/list). Senza injection il PLANNER vedrebbe
# solo i get_* del dominio, non riconoscerebbe find_places, e attiverebbe
# request_new_executor su un nome che esiste gia' (caso live 1/5/2026).
_OBJECT_PRIMARY_TOOLS = {
    # Iniezione automatica nel pool dei top-K quando l'object e' detectato
    # nella query: garantisce che il PLANNER veda l'executor canonico
    # PRIMA di scivolare a `request_new_executor` (caso ricorrente 4/5/2026:
    # "Quante istanze di claudio sono running" non includeva get_processes
    # → synt scattava su executor gia' esistente).
    "places":    ("find_places",),
    "processes": ("get_processes",),
    "messages":  ("read_messages", "send_messages",
                   "move_messages", "find_messages"),
    "persons":   ("get_persons", "set_persons",
                   "find_persons_indices", "delete_persons"),
    "tasks":     ("list_tasks", "read_tasks", "create_tasks",
                   "delete_tasks", "set_tasks", "read_tasks_history"),
    "files":     ("find_files", "read_files"),
    "dirs":      ("list_dirs", "find_dirs"),
    "urls":      ("find_urls", "get_urls", "read_urls_html", "read_urls_pdf"),
    # Calendar events (Google Workspace skill, importati 10/5/2026,
    # rinominato ADR 0128 12/5/2026: set_events -> create_events).
    # create_events (crea), read_events (lettura), delete_events (cancella).
    "events":    ("create_events", "read_events", "delete_events"),
    "calendars": ("create_calendars", "delete_calendars"),
    # Contatti Google Workspace (read_contacts dal skill):
    "contacts":  ("read_contacts",),
    "images":    ("find_images_indices", "change_images", "find_files"),
    "packages":  ("find_packages",),  # canonical handcrafted name (no get_packages)
    "numbers":   (),  # niente primary, lascia al ranker
    "texts":     ("read_files", "filter_texts_lines"),
    "signatures": ("get_signatures",),
    # ADR 0090 (4-5/5/2026): get_inputs e' il motore UI dichiarativo per
    # raccolta valori dall'utente (dialog/form/voice). Iniezione su query
    # come "chiedimi", "dialogo", "form", "modulo": il PLANNER vede subito
    # il tool canonico invece di scivolare a request_new_executor.
    "inputs":    ("get_inputs",),
}

_QUERY_DEPENDENT_PRECURSORS = (
    ("find_places", "get_location", (
        # IT: marker location-relative — TUTTE le forme di genere/numero
        # ("vicino" maschile non matcha "vicina/vicini/vicine" col substring
        # match: la lista deve enumerarle esplicitamente).
        "vicino a me", "vicino", "vicina", "vicini", "vicine",
        "piu vicino", "piu vicina", "piu vicini", "piu vicine",
        "piu' vicino", "piu' vicina",  # con apostrofo
        "qui", "qua", "intorno", "intorno a me", "intorno a noi",
        "nelle vicinanze", "nei dintorni", "in zona", "in giro",
        "vicinissimo", "vicinissima",
        # EN
        "near me", "around me", "around here", "around us",
        "nearby", "nearest", "closest", "close by", "close-by",
        "in the area", "in proximity",
    )),
)


# EXIF-intent markers (4/6/2026): `get_files` (azione_oggetto = get+files, il
# tool EXIF/dates/place/gps/device) va iniettato nel pool SOLO quando la query
# riguarda i METADATI di scatto di una foto, NON per query generiche su file
# (es. "elenca i file con la dimensione" → get_files NON serve, find_files ha
# gia' size → evita il misroute get_files(fields=["size"]) che e' enum-invalid).
# Deterministico §7.9: substring match. Vedi core-rule §5 EXIF→get_files.
_EXIF_MARKERS = (
    # IT
    "exif", "scattat", "metadati foto", "metadati della foto", "geotag",
    "luogo di scatto", "dove e' stata fatta", "dove è stata fatta",
    "dove e' stata scattata", "dove è stata scattata",
    "quando e' stata scattata", "quando è stata scattata",
    "con che camera", "con quale camera", "con che fotocamera",
    "che fotocamera", "modello di fotocamera", "coordinate gps",
    # EN
    "with what camera", "which camera", "where was it taken",
    "when was it taken", "where was this photo", "when was this photo",
    "capture date", "gps coordinates",
)


# Shell-intent hints (ADR 0088): query con questi marker triggerano
# l'iniezione automatica di `admin` nel pool top-K. Il pianificatore
# vede admin come tool ordinario, lo seleziona, il vaglio always-on
# emette la carta dialog manager.
#
# Detection: word-boundary regex (12/5/2026). Match per substring naive
# generava falsi positivi disastrosi (es. "ferma" matchava "conferma",
# "afferma", "fermata"; "share" matchava "shared" e cosi' via; "kill"
# matchava "skill"). Con `\b...\b` la condizione e' "token intero".
# Gli hint multi-word (ip route, comando shell, log di sistema) restano
# match come frase intera grazie a `\b` alle estremita'. Determinismo
# §7.9: regex compilata, niente LLM.
_TIME_INTENT_HINTS = (
    # Italiano
    "che ora", "che ore", "ora corrente", "data corrente", "che giorno",
    "che data", "adesso", "in questo momento",
    # Inglese
    "what time", "what date", "current time", "current date",
    "what day", "today is", "what's the time", "right now",
)


_TIME_INTENT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(h) for h in _TIME_INTENT_HINTS) + r")\b",
    re.IGNORECASE,
)


def _detect_time_intent(qlow: str) -> bool:
    """True se la query chiede ora/data corrente. Match word-boundary."""
    return bool(_TIME_INTENT_RE.search(qlow or ""))


_SHELL_INTENT_HINTS = (
    # mount / umount
    "mount", "monta", "monto", "montare", "umount", "smonta", "smontare",
    "share", "nas", "cifs", "smb", "nfs",
    # kill / process
    "kill", "uccidi", "termina", "killa", "ammazza",
    # systemctl / services
    "systemctl", "service", "servizio", "restart", "riavvia", "riavviare",
    "start", "avvia", "avviare", "stop", "ferma", "fermare", "fermo",
    # permissions
    "chmod", "chown", "permessi", "permission",
    # network — basics
    "ifconfig", "ip route", "iptables", "rete", "network",
    # packages
    "apt", "apt-get", "pacchetto", "package", "installa", "installare",
    # logs
    "journalctl", "syslog", "log di sistema",
    # generic shell verb
    "comando shell", "shell command", "esegui",
    # ── long-tail sysinfo fallback (22/5/2026): query che `get_processes`
    # non copre (porte, socket, kernel module, GPU, ecc.). Trigger inietta
    # admin nel pool: il LLM (che conosce i comandi Linux) propone p.es.
    # `ss -tlnp` o `lsmod` e la whitelist v3 li auto-approva.
    "porta", "porte", "port", "ports",
    "socket", "sockets",
    "listening", "ascolta", "ascoltante",
    "tcp", "udp",
    "modulo kernel", "moduli kernel", "kernel module",
    "scheda video", "gpu", "video card",
    "lsof", "lsblk", "lsmod", "lspci", "lsusb",
    "dmesg", "sensors", "sensor",
)


_SHELL_INTENT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(h) for h in _SHELL_INTENT_HINTS) + r")\b",
    re.IGNORECASE,
)


def _detect_shell_intent(qlow: str) -> bool:
    """True se la query contiene un marker shell-related come TOKEN INTERO
    (word boundary). Case-insensitive."""
    return bool(_SHELL_INTENT_RE.search(qlow or ""))


def _query_has_marker(qlow, markers):
    """True se la query (lowercase) contiene almeno uno dei marker.
    Match per substring (i marker sono frasi corte, sufficiente per IT/EN)."""
    return any(m in qlow for m in markers)


def rank_with_intent(query, catalog, intent, *, k=3):
    # Skip dormant: come rank_adaptive, vedi _filter_dormant.
    catalog = _filter_dormant(catalog)
    """Ranking quando un intent_extractor ha gia' identificato verb+object.

    Filtra il catalog per `name.startswith(verb_)`; fra i match preferisce
    quelli con object nei name_parts. Cap a `k` (default 3 — coerente con
    "max 3 candidates" — Roberto 29/4/2026).

    Per verbi CONSUMER (qualunque verbo non in _PRODUCER_VERBS — quindi
    move/delete/send/write/extract/create + describe/classify/filter/sort/
    group/render/set/compress/compute/compare) include automaticamente UN
    precursor (read/find/get/list per lo stesso object) perche' il planner
    ha bisogno di leggere/cercare prima di agire o riassumere/filtrare.
    Senza il precursor il planner chiamerebbe il verbo-finale con from_step=0
    (caso live 29/4/2026 sera + regressione 30/4/2026 mattina su
    "riassumi le mail importanti": verb=describe non era destructive,
    nessun precursor → describe_entries con history vuota).
    """
    verb = (intent or {}).get("verb")
    obj = (intent or {}).get("object")
    if not verb:
        return None  # caller fa fallback lexicon
    qtokens = tokenize(query) if query else set()
    # §7.3 opt-in: rule_boost wire-in nel path intent-driven (era applicato
    # solo nel fallback BoW). Gating env METNOS_PREFILTER_RULES=1.
    import os as _os_intent
    _rules_on = (_os_intent.environ.get("METNOS_PREFILTER_RULES", "0") == "1"
                  and query)
    _rule_fn = None
    if _rules_on:
        try:
            from prefilter_rules import compute_rule_boost, init_rare_tokens
            init_rare_tokens(catalog)
            _rule_fn = compute_rule_boost
        except Exception as _e:  # §2.8 no silent failure
            log.warning("prefilter_rules init in rank_with_intent: %s", _e)
            _rule_fn = None
    primary = []
    for e in catalog:
        parts = e.name.split("_")
        _first = parts[0] if parts else ""
        if _first == verb:
            s = 10
        elif _first in _VERB_ALSO_CANONICAL.get(verb, ()):
            s = _VERB_SIBLING_BOOST  # sibling di lifecycle: nel pool, sotto il primario
        else:
            continue
        if obj and obj in parts:
            s += 6
        # Qualifier bonus SOLO se il qualifier matcha un token nella query
        # (es. query "leggi il csv" → bonus per read_files_csv). Altrimenti
        # il generico `read_files` deve poter battere i qualified per query
        # senza qualifier-keyword (regressione 30/4/2026: "leggi /etc/hostname"
        # picked read_files_csv per cap k=3).
        if len(parts) >= 3:
            qualifiers = parts[2:]
            if qtokens and any(q in qtokens for q in qualifiers):
                s += 2  # forte bonus se il qualifier matcha la query
            # else: nessun bonus — il generico (parts=[verb,obj]) puo' battere
        if _rule_fn is not None:
            try:
                s += _rule_fn(query, qtokens, verb, e)
            except Exception as _e:
                log.warning("rule_boost in rank_with_intent for %s: %s",
                             e.name, _e)
        primary.append((s, e))
    primary.sort(key=lambda p: (-p[0], getattr(p[1], "name", "")))

    # Se nessun executor matcha il verbo dell'intent (es. query compound dove
    # l'estrattore ha pickato un verbo intermedio come "group" senza alcun
    # group_* in catalog), l'intent e' "debole". Ricadi su bag-of-words
    # tornando None: il fallback sceglie un set piu' ampio basato su token
    # match (regressione 30/4/2026 UC3: "Trova ... raggruppa ..." → intent
    # group → solo read_files come precursor → planner privo di find_files).
    #
    # ECCEZIONE (1/5/2026): per i universal-helper verbs (classify, filter,
    # sort, describe, compute) il tool e' iniettato in-process da
    # agent_runtime, non vive nel catalog manifest. NON ritornare None:
    # procedi al precursor injection (caller comporra' classify_entries +
    # read_messages tramite il path universal-helper).
    _UNIVERSAL_HELPER_VERBS = ("classify", "filter", "sort", "describe", "compute")
    if not primary and verb not in _UNIVERSAL_HELPER_VERBS:
        return None

    # Precursor injection PRIMA del check incoherent: se il verb e' consumer
    # e c'e' obj, aggiungi i producer (read/find/list/get) dell'obj. Solo
    # cosi' un intent come `compute, dirs` (compute_* non ha dirs ma find_dirs
    # si) viene riconosciuto come coerente; senza injection prima, il check
    # successivo lo rejecterebbe.
    if verb not in _PRODUCER_VERBS and obj:
        # Verbo CONSUMER (describe, classify, filter, move, delete, send,
        # compress, write, compute, ...) → aggiungi TUTTI i precursor
        # producer del medesimo oggetto disponibili in catalog (uno per pverb
        # in _PRECURSOR_VERBS). Cap k_max=8 li accomoda.
        seen = {e.name for _, e in primary}
        for pverb in _PRECURSOR_VERBS:
            for e in catalog:
                parts = e.name.split("_")
                if parts[0] == pverb and obj in parts and e.name not in seen:
                    primary.append((5, e))   # score < primary, ma >0
                    seen.add(e.name)
                    break  # uno per pverb, non tutti i qualified

    # Object primary tools: per ogni OBJECT esistono executor "stella" che
    # vanno SEMPRE inclusi nei candidati per quell'object, anche se l'intent
    # ha picked un verbo diverso (caso 1/5/2026 "Farmacia piu vicina":
    # intent get/places → solo get_* nei candidati → find_places escluso →
    # PLANNER attiva request_new_executor su nome esistente).
    if obj and obj in _OBJECT_PRIMARY_TOOLS:
        seen_for_obj = {e.name for _, e in primary}
        for primary_name in _OBJECT_PRIMARY_TOOLS[obj]:
            if primary_name in seen_for_obj:
                continue
            primary_exec = next((e for e in catalog if e.name == primary_name), None)
            if primary_exec is not None:
                primary.append((8, primary_exec))  # score sopra precursor generici (5) sotto match diretto (10+)
                seen_for_obj.add(primary_name)

    # Cross-tool query-driven precursors (vedi _QUERY_DEPENDENT_PRECURSORS).
    # Aggiunge un provider quando il consumer e' nei candidati E la query
    # contiene un marker semantico che lo rende necessario. Es. find_places
    # + "vicino a me" → inietta get_location.
    qlow = (query or "").lower()
    seen_names = {e.name for _, e in primary}
    for cons_name, prov_name, markers in _QUERY_DEPENDENT_PRECURSORS:
        if cons_name not in seen_names:
            continue
        if not _query_has_marker(qlow, markers):
            continue
        if prov_name in seen_names:
            continue
        prov_exec = next((e for e in catalog if e.name == prov_name), None)
        if prov_exec is not None:
            primary.append((7, prov_exec))  # score sopra precursor generici
            seen_names.add(prov_name)

    # EXIF injection condizionale (4/6/2026): get_files entra nel pool SOLO se
    # la query ha marker EXIF (scatto/gps/camera/...) e l'object e' files/images.
    # Cosi' le query EXIF-by-path lo vedono (core-rule §5), ma le query generiche
    # su file ("dimensione/elenco") NON sono tentate da get_files (che e' EXIF-only
    # → get_files(fields=["size"]) = enum-invalid, misroute 4/6). Deterministico §7.9.
    if obj in ("files", "images") and "get_files" not in seen_names \
            and _query_has_marker(qlow, _EXIF_MARKERS):
        gf = next((e for e in catalog if e.name == "get_files"), None)
        if gf is not None:
            primary.append((9, gf))  # alta priorita': intento EXIF esplicito
            seen_names.add("get_files")

    # Admin shell injection (ADR 0088, 4/5/2026): query con shell-intent
    # marker (mount/kill/systemctl/...) → admin a priorità massima.
    # Permette al PLANNER di sceglierlo invece di scivolare a
    # `request_new_executor` o produrre final_answer di resa.
    # 22/5/2026: shell-intent esteso a long-tail sysinfo (port/socket/lsmod/
    # gpu/...). Se admin gia' in seen_names (matched per affinity), lo
    # promuoviamo comunque al top — il PLANNER deve vederlo come prima
    # opzione, non al 6° posto.
    if _detect_shell_intent(qlow):
        admin_exec = next((e for e in catalog if e.name == "admin"), None)
        if admin_exec is not None:
            primary = [(s, e) for s, e in primary if e.name != "admin"]
            primary.insert(0, (15, admin_exec))
            seen_names.add("admin")

    # Time intent injection (6/5/2026): "che ore sono", "what time", etc.
    # → inietta get_now con priorità massima. Senza questo il prefilter
    # ranking BoW fallisce perche' il vocabolario obj non ha "time"
    # come oggetto canonico, e l'affinity (ora/orario) non matcha tutti
    # i casi morfologici (ore plurale, "che ora").
    if _detect_time_intent(qlow) and "get_now" not in seen_names:
        get_now_exec = next((e for e in catalog if e.name == "get_now"), None)
        if get_now_exec is not None:
            primary.insert(0, (15, get_now_exec))
            seen_names.add("get_now")

    # Check coerenza: dopo l'injection, deve esserci almeno un candidato
    # con obj in name. Se ancora nessuno (es. obj=messages ma no executor
    # con messages), fallback BoW per pesare i token della query.
    if obj and not any(obj in e.name.split("_") for _, e in primary):
        return None

    primary.sort(key=lambda p: (-p[0], getattr(p[1], "name", "")))
    # Layer 1 (5/5/2026): force-include dei primary tools dell'object oltre
    # il cap top-K. La tupla `_OBJECT_PRIMARY_TOOLS[obj]` dichiara TUTTI gli
    # executor canonici per il dominio (es. urls → find_urls, get_urls,
    # read_urls_html). Senza force-include, il top-K=8 puo' tagliare uno di
    # quelli (es. read_urls_html score 8 vs 8 verbi-match score 10) → step
    # successivo non vede il consumer. Generale: vale per ogni object detected.
    primary_names = set(_OBJECT_PRIMARY_TOOLS.get(obj, ())) if obj else set()
    head = [e for _, e in primary[:k]]
    head_names = {e.name for e in head}
    forced = []
    for _, e in primary:
        if e.name in primary_names and e.name not in head_names:
            forced.append(e)
            head_names.add(e.name)
    return head + forced


def _filter_dormant(catalog):
    """Skip executor dormant (ADR 15/5/2026): importati da skill senza
    credenziali (es. *_google_workspace pre-OAuth). Visibili in
    `metnos-skills list` per introspezione, nascosti al PLANNER. Pattern
    deterministico §7.9: attributo `dormant: bool` settato dal loader."""
    return [e for e in catalog if not getattr(e, "dormant", False)]


def rank_adaptive(query, catalog, k_min=5, k_max=8, *, llm_call=None,
                   prefer_intent=True):
    """Dispatcher modulare (17/5/2026): delega alla strategy selezionata da
    env `METNOS_PREFILTER` (default `legacy` = comportamento storico).

    Strategy registrate in `runtime/prefilter_strategies/__init__.py`.
    Per backward compat, in assenza di env var o per `METNOS_PREFILTER=
    legacy|token_flat` ritorna esattamente il comportamento di
    `_rank_adaptive_legacy` originale.

    Telemetria opt-in (`METNOS_PREFILTER_TELEMETRY=1`): logga ogni call su
    `~/.local/share/metnos/prefilter_telemetry.jsonl` per A/B compare.

    Compare mode (`METNOS_PREFILTER=compare:a,b`): esegue entrambi A e B in
    sequenza, ritorna A, ma logga B come confronto.
    """
    import os as _os
    chosen_env = _os.environ.get("METNOS_PREFILTER", "").strip().lower()
    if not chosen_env or chosen_env in ("legacy", "token_flat"):
        # Fast path: nessun overhead per il default.
        result = _rank_adaptive_legacy(
            query, catalog, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        if _os.environ.get("METNOS_PREFILTER_TELEMETRY", "0") == "1":
            _log_telemetry("legacy", query, result)
        return result
    # Modular path
    from prefilter_strategies import select_strategy
    import time as _time
    primary_name = chosen_env
    secondary_name = None
    if chosen_env.startswith("compare:"):
        parts = chosen_env.split(":", 1)[1].split(",")
        primary_name = parts[0].strip()
        if len(parts) > 1:
            secondary_name = parts[1].strip()
    primary = select_strategy(primary_name)
    t0 = _time.perf_counter()
    result = primary.rank(query, catalog, k_min=k_min, k_max=k_max,
                          llm_call=llm_call, prefer_intent=prefer_intent)
    elapsed_ms = int((_time.perf_counter() - t0) * 1000)
    _log_telemetry(primary.name, query, result, elapsed_ms=elapsed_ms)
    # Compare mode: lancia secondary, logga ma non ritorna.
    if secondary_name:
        try:
            secondary = select_strategy(secondary_name)
            t1 = _time.perf_counter()
            sec_result = secondary.rank(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
            sec_elapsed_ms = int((_time.perf_counter() - t1) * 1000)
            _log_telemetry(
                secondary.name, query, sec_result,
                elapsed_ms=sec_elapsed_ms, compare_against=primary.name,
            )
        except Exception as ex:
            import logging
            logging.getLogger(__name__).warning(
                "compare-mode secondary %r failed: %s", secondary_name, ex)
    return result


def _log_telemetry(strategy_name: str, query: str, result, *,
                    elapsed_ms: int | None = None,
                    compare_against: str | None = None) -> None:
    """Append JSONL telemetry record. Best-effort, fail-silent."""
    try:
        import json
        import hashlib
        import time
        from pathlib import Path
        candidates, route_info = result if isinstance(result, tuple) else (result, {})
        top3 = []
        for e in (candidates or [])[:3]:
            n = getattr(e, "name", None) or str(e)[:60]
            top3.append(n)
        rec = {
            "ts": time.time(),
            "strategy": strategy_name,
            "query_hash": hashlib.sha256(query.encode()).hexdigest()[:12],
            "query_len": len(query),
            "n_candidates": len(candidates or []),
            "top3": top3,
            "confidence": (route_info or {}).get("confidence"),
            "reason": (route_info or {}).get("reason"),
        }
        if elapsed_ms is not None:
            rec["elapsed_ms"] = elapsed_ms
        if compare_against:
            rec["compare_against"] = compare_against
        import config as _C  # §7.11 (local import per evitare circular)
        p = _C.PATH_USER_DATA / "prefilter_telemetry.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _rank_adaptive_legacy(query, catalog, k_min=5, k_max=8, *, llm_call=None,
                           prefer_intent=True):
    """
    Forma adattiva (preferita v1.1) con intent extractor LLM-based opzionale.

    Pipeline (Roberto 29/4/2026):
        1. Se llm_call disponibile e prefer_intent=True: chiama
           intent_extractor.extract_intent → {verb, object}.
        2. Se intent valido produce candidati >0: ritorna max-3 ranked
           per verb+object boost.
        3. Altrimenti fallback al ranking bag-of-words (verb_boost+object_boost
           via lexicon, soft-match cap-ato).

    Vantaggio dell'intent extractor: robusto a variazioni di linguaggio
    ("archivia", "svuota cestino", "metti in spam") che il lexicon manuale
    non copre. Latenza tipica ~350ms con gemma 4 26B middle tier.

    Filtro relativo (28/4 sera): tieni solo score >= max(1, top_score / 2).
    Evita di passare al planner tool con affinity bassa che fanno rumore (Gemma
    sotto-pesa le description e si attacca a nomi calamita). Cap superiore a
    k_max comunque.
    """
    # Skip dormant (skill_credentials check, ADR 15/5/2026): il PLANNER
    # non deve vedere executor inattivi per mancanza di OAuth/token.
    catalog = _filter_dormant(catalog)
    # 1. Intent extractor (LLM-based) se disponibile
    if llm_call is not None and prefer_intent:
        try:
            from intent_extractor import extract_intent
            intent = extract_intent(query, llm_call)
        except Exception:
            intent = None
        if intent and intent.get("verb"):
            picked = rank_with_intent(query, catalog, intent, k=k_max)
            if picked:
                return picked, {
                    "chosen_k": len(picked),
                    "confidence": 1.0,  # LLM-determined intent → high confidence
                    "reason": "intent",
                    "intent": intent,
                }
    # 2. Fallback bag-of-words
    qtokens = tokenize(query)
    if not qtokens:
        executors = list(catalog)
        return executors[:k_min], {"chosen_k": min(k_min, len(executors)),
                                    "confidence": 0.0, "scores_top": [], "reason": "empty_query"}
    canonical_verb = detect_canonical_verb(qtokens)
    canonical_object = detect_canonical_object(qtokens, query)
    # §7.3 Task #41 (28/5/2026): pass query_raw per attivare typed-rules
    # (input_coverage, schema_field) gated da METNOS_PREFILTER_RULES=1.
    # Init rare-tokens cache (idempotente, no-op se env=0).
    import os as _os_pref
    if _os_pref.environ.get("METNOS_PREFILTER_RULES", "0") == "1":
        try:
            from prefilter_rules import init_rare_tokens
            init_rare_tokens(catalog)
        except Exception:
            pass
    scored = [(affinity_score(qtokens, e,
                              query_canonical_verb=canonical_verb,
                              query_canonical_object=canonical_object,
                              query_raw=query), e)
              for e in catalog]
    scored.sort(key=lambda p: (-p[0], getattr(p[1], "name", "")))
    scores = [s for s, _ in scored]
    top_score = scores[0] if scores else 0
    semantic_reason = ""
    # Semantic fallback (BGE-M3) quando hard match troppo debole: query con
    # typo, sinonimi semantici, declinazioni irregolari, cross-lingua. Costa
    # ~25ms quando attivato; skip quando hard match e' gia' confident.
    try:
        from affinity_semantic import (
            is_enabled as _sem_enabled, threshold as _sem_threshold,
            alpha as _sem_alpha, build_or_load_cache as _sem_build,
            semantic_max_per_executor as _sem_max,
        )
        if _sem_enabled() and top_score < _sem_threshold():
            _cache = _sem_build(list(catalog))
            if _cache is not None:
                _semmap = _sem_max(query, _cache)
                if _semmap:
                    _a = _sem_alpha()
                    scored = [(s + _a * _semmap.get(e.name, 0.0), e)
                              for s, e in scored]
                    scored.sort(key=lambda p: (-p[0], getattr(p[1], "name", "")))
                    scores = [s for s, _ in scored]
                    top_score = scores[0] if scores else 0
                    semantic_reason = "semantic_fallback"
    except Exception:
        pass  # fallback silente: il hard match ranking resta valido
    rel_cutoff = max(1, top_score // 2)
    relevant = [(s, e) for s, e in scored if s >= rel_cutoff]
    if len(relevant) < k_min:
        # garantisci almeno k_min, anche pescando sotto la soglia
        relevant = scored[:k_min]
    K = min(len(relevant), k_max)
    K = max(K, k_min) if scored else 0
    selected = [e for _, e in scored[:K]]
    # Layer 1 (5/5/2026): force-include dei primary tools per l'object
    # detectato anche nel fallback BoW. Pareggia il comportamento di
    # rank_with_intent: la tupla `_OBJECT_PRIMARY_TOOLS[obj]` entra TUTTA
    # nel pool oltre il top-K, garantita visibile al PLANNER.
    if canonical_object and canonical_object in _OBJECT_PRIMARY_TOOLS:
        sel_names = {e.name for e in selected}
        for primary_name in _OBJECT_PRIMARY_TOOLS[canonical_object]:
            if primary_name in sel_names:
                continue
            primary_exec = next((e for e in catalog if e.name == primary_name), None)
            if primary_exec is not None:
                selected.append(primary_exec)
                sel_names.add(primary_name)
    # Admin shell injection (ADR 0088) anche nel fallback BoW: se la query
    # ha shell-intent marker e admin esiste nel catalog, garantisce che
    # sia in cima al pool (head-injection). 22/5/2026: anche se gia'
    # presente per affinity, lo promuoviamo a position 0.
    qlow_bow = (query or "").lower()
    if _detect_shell_intent(qlow_bow):
        admin_exec = next((e for e in catalog if e.name == "admin"), None)
        if admin_exec is not None:
            selected = [e for e in selected if e.name != "admin"]
            selected = [admin_exec] + selected[:max(0, k_max - 1)]
    _, conf = adaptive_k(scores, k_min, k_max)
    return selected, {
        "chosen_k": K,
        "confidence": round(conf, 3),
        "top_score": top_score,
        "rel_cutoff": rel_cutoff,
        "scores_top": scores[:max(K, 5)],
        "reason": (semantic_reason or
                   (f"rel_cutoff>={rel_cutoff}" if rel_cutoff > 1
                    else "k_min_floor")),
    }


def explain(query, catalog, k=5):
    qtokens = tokenize(query)
    print(f"query='{query}'  tokens={sorted(qtokens)}")
    scored = sorted(
        ((affinity_score(qtokens, e), e) for e in catalog),
        key=lambda p: p[0], reverse=True,
    )
    print(f"top {k}:")
    for s, e in scored[:k]:
        aff_match = qtokens & set(t for tag in e.affinity for t in tokenize(tag))
        print(f"  score={s:3d}  {e.name:14s}  match={sorted(aff_match)}")
    K, conf = adaptive_k([s for s, _ in scored])
    print(f"  --> adaptive_k={K} (confidence={conf:.2f})")


if __name__ == "__main__":
    from loader import load_catalog
    cat = load_catalog()
    queries = [
        "che ora e?",
        "leggi il file ~/notes/diary.md",
        "scarica https://httpbin.org/get",
        "salva il documento sul disco",
        "scrivi una nota e mandala via mail",
        "fai qualcosa di carino",  # query vaga
    ]
    for q in queries:
        print()
        explain(q, cat)
