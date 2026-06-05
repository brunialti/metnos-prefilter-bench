#!/usr/bin/env python3
"""config — costanti e path centralizzati di Metnos.

Fonte UNICA di:
  - paths assoluti (root install, workspace, user data, state, config)
  - costanti di tuning (cap, soglie, decay, timeout) con rationale inline
  - env override (METNOS_*) per deploy diversi

Pattern: ogni modulo importa `from config import C`. Niente magic number
nei moduli applicativi; niente path hardcoded sparsi.

Convenzioni:
  - C.PATH_*           Path assoluti (Path objects, non str)
  - C.DB_*             percorsi DB sqlite (sotto user state/data)
  - C.CAP_*            limiti runtime (loop break, max items)
  - C.WEIGHT_*         pesi e soglie mnestoma (decay, reinforce, archive)
  - C.TIMEOUT_*        timeout in secondi (task, request, push)
  - C.DEFAULT_*        default dei tier (lang, channel, llm tier)

Env override (tutti opzionali):
  METNOS_HOME           override root install (default <install_root>)
  METNOS_USER_DATA      override ~/.local/share/metnos
  METNOS_USER_STATE     override ~/.local/state/metnos
  METNOS_USER_CONFIG    override ~/.config/metnos
  METNOS_LANG           lingua interfaccia (default 'it')
  METNOS_CAP_STEPS      max step per turno (default 30)
  METNOS_LOG_LEVEL      logger root (DEBUG|INFO|WARNING|ERROR; default INFO)
  METNOS_LOG_FILE       path file log (default journal-only)
  METNOS_INDEX_ROOT     override <USER_DATA>/index (storage indici di dominio)
  METNOS_DRY_RUN        "1" → executor write short-circuitano (no side effects)
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v else default
    except (ValueError, TypeError):
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Root paths -----------------------------------------------------------

# Install root (codice + executor canonici + decisions).
# Default auto-derived da `Path(__file__).resolve().parents[1]` — questo file
# vive in <PATH_ROOT>/runtime/config.py, quindi `parents[1]` ricava la root
# senza ipotesi sul nome del path. Cambiare il filesystem layout (es.
# rinomina <install_root> → /opt/metnos) non richiede nessuna config: il codice
# si trova da solo.
# Override esplicito via env METNOS_INSTALL_ROOT (preferito, ADR 0148);
# l'alias METNOS_HOME e' deprecato — viene letto solo se METNOS_INSTALL_ROOT
# non e' settato, per back-compat finche' tutti gli script downstream non
# sono passati al nuovo nome.
_AUTO_ROOT         = Path(__file__).resolve().parents[1]
PATH_ROOT          = _env_path(
    "METNOS_INSTALL_ROOT",
    _env_path("METNOS_HOME", _AUTO_ROOT),
)
PATH_RUNTIME       = PATH_ROOT / "runtime"
PATH_EXECUTORS     = PATH_ROOT / "executors"
PATH_WORKSPACE     = PATH_ROOT / "workspace"
PATH_DECISIONS     = PATH_ROOT / "decisions"
PATH_DOCS          = PATH_ROOT / "docs"

# User XDG paths (per-utente, scrivibili senza sudo)
PATH_USER_DATA     = _env_path("METNOS_USER_DATA",
                                Path.home() / ".local" / "share" / "metnos")
PATH_USER_STATE    = _env_path("METNOS_USER_STATE",
                                Path.home() / ".local" / "state" / "metnos")
PATH_USER_CONFIG   = _env_path("METNOS_USER_CONFIG",
                                Path.home() / ".config" / "metnos")

# Synth executors (synth on-the-fly, ADR 0066)
PATH_SYNTH_EXECUTORS = PATH_USER_DATA / "executors"
# Skill imported (ADR 0123; ADR 0160 rename _imports/ → skills/).
# Loader scansiona ENTRAMBI: i write nuovi vanno in PATH_SKILLS_USER (new),
# le installazioni legacy in PATH_SKILLS_USER_LEGACY restano leggibili.
PATH_SKILLS_USER         = PATH_USER_DATA / "executors" / "skills"
PATH_SKILLS_USER_LEGACY  = PATH_USER_DATA / "executors" / "_imports"
PATH_SKILLS_BUILTIN      = PATH_EXECUTORS / "skills"
# Audit log dir (introvertiva, vaglio, synt; ADR 0067)
PATH_AUDIT         = PATH_USER_DATA / "introvertiva"
# History turns (TurnLog jsonl daily files)
PATH_TURNS         = PATH_USER_DATA / "turns"
# Cost tracking (LLM provider costs)
PATH_COST          = PATH_USER_DATA / "cost"
# Index root (indici di dominio: image/scene, image/persons, image/gps, ...).
# Override via METNOS_INDEX_ROOT per isolare i test E2E dry-run dallo storage
# di produzione (8/5/2026: il bug "47 sha8 orfane con base_path=/tmp/..." nasce
# proprio da test che scrivevano sotto ~/.local/share/metnos/index/ globale).
PATH_INDEX_ROOT    = _env_path("METNOS_INDEX_ROOT", PATH_USER_DATA / "index")
# Index image (storage canonico ADR 0086 + 0113)
PATH_INDEX_IMAGE   = PATH_INDEX_ROOT / "image"

# Dry-run globale: se True, gli executor "write" (create/delete/move/write/
# change/send/set) short-circuitano la parte distruttiva e ritornano un
# payload `{ok, dry_run:true, would_*}` senza side-effect. Read-only invariati.
DRY_RUN            = _env_str("METNOS_DRY_RUN", "0") == "1"

# --- Database paths ------------------------------------------------------

# Mnestoma (mnest grafo + events)
DB_MNESTOMA        = PATH_WORKSPACE / ".mnestoma" / "mnest.sqlite"
# Scheduler (system tasks + recurring user tasks state)
DB_SCHEDULER       = PATH_WORKSPACE / ".scheduler" / "state.sqlite"
# i18n testi multilingua
DB_I18N            = PATH_USER_DATA / "i18n.sqlite"
# Scratchpad (handle observation grandi, ADR 0050)
DB_SCRATCHPAD      = PATH_USER_DATA / "scratchpad.db"
# Pairings (multi-device, ADR 0035)
DB_PAIRINGS        = PATH_USER_STATE / "pairings.db"
# Recurring user tasks (registered via PLANNER create_tasks)
DB_RECURRING_TASKS = PATH_USER_STATE / "recurring_tasks.db"
# Approvals (autonomy_level + grant pending)
DB_APPROVALS       = PATH_USER_STATE / "approvals.db"
# Devices (multi-device pairing extensions)
DB_DEVICES         = PATH_USER_STATE / "devices.db"
# Policy (autonomy_level matrix)
DB_POLICY          = PATH_USER_STATE / "policy.db"
# Observability (run history, dashboard data)
DB_OBSERVABILITY   = PATH_USER_STATE / "observability.db"
# Multi-tool fast-path memoization (ADR 0150): canonical_query → tools sequence
# memoizzata, TTL N giorni di attivita' effettiva.
DB_MULTI_TOOL_PATHS = PATH_USER_DATA / "multi_tool_paths.sqlite"
# Change intents (ADR 0158): single source of truth per il ciclo di vita
# proposed → accepted → applied → observed → finalized (o rolled_back).
# Sostituisce 9 storage frammentati (telos jsonl, introvertiva sqlite,
# synt jsonl, multi_tool sqlite, canonical_query_log, executor_history, ...).
DB_CHANGE_INTENTS  = PATH_USER_STATE / "change_intents.sqlite"
# Audit JSONL (append-only, no schema; non-DB ma simile)
LOG_LOCATIONS_JSONL = PATH_USER_DATA / "locations.jsonl"

# --- Tuning costanti ------------------------------------------------------

# Loop / step cap (agent_runtime). Override via METNOS_CAP_STEPS.
# Razionale: 30 step e' soglia oltre cui il PLANNER molto raramente converge
# senza loop_break. Cap sotto = utenti smart query bloccati.
CAP_STEPS              = _env_int("METNOS_CAP_STEPS", 30)
# Stesso executor in fila prima di marcare loop. 10 = soglia conservativa
# pre-cap-expand prompt.
CAP_SAME_EXECUTOR      = _env_int("METNOS_CAP_SAME_EXECUTOR", 10)
# Observation > soglia → offload a scratchpad invece di passare inline.
CAP_OBSERVATION_BYTES  = _env_int("METNOS_OBS_BYTES", 4096)

# --- Mnestoma tuning ------------------------------------------------------

# Reinforce per ogni passing osservato. 0.15/passing → mnest a uses=4
# raggiunge weight ~0.7 (sopra soglia synth_trigger).
WEIGHT_REINFORCE       = 0.15
# Weight iniziale di un nuovo mnest (mai osservato prima).
WEIGHT_BOOTSTRAP       = 0.30
# Decay esponenziale per giorno. 0.018/giorno → dimezzamento ~38 giorni.
# Razionale: pattern visto 1 mese fa pesa meta' di uno visto oggi.
WEIGHT_DECAY_LAMBDA    = 0.018
# Sotto questa soglia, mnest viene "decayed" (state=decaying invece active).
WEIGHT_DECAY_THRESHOLD = 0.20
# Sotto questa soglia + age, mnest archiviato (state=archived).
WEIGHT_ARCHIVE_THRESHOLD = 0.05
# Eta' minima in giorni per archive (evita archive di mnest nuovi che
# decadono brevemente per inattivita' temporanea).
ARCHIVE_AGE_DAYS       = 90
# Proto-mnest mai promossi sotto soglia → purgati.
PROTO_PURGE_THRESHOLD  = 0.05

# --- Synth trigger --------------------------------------------------------

# Numero uses minimo per synth-trigger (mnest deve essere visto N volte).
SYNTH_TRIGGER_USES     = 3
# Weight minimo (oltre il quale il mnest e' ritenuto "stabile").
SYNTH_TRIGGER_WEIGHT   = 0.30

# --- Timeout (secondi) ----------------------------------------------------

# Default task fire timeout (scheduler). Oltre → status='timeout'.
TIMEOUT_TASK_S         = 300
# Push canale (Telegram send) per fire ricorrente.
TIMEOUT_PUSH_S         = 30
# Approval pending TTL (cap_pending dialog).
TIMEOUT_APPROVAL_S     = 600
# Location request timeout (utente non risponde a prompt 📍).
TIMEOUT_LOCATION_S     = 300
# Scratchpad TTL (1 ora).
TIMEOUT_SCRATCHPAD_S   = 3600

# --- Default app ---------------------------------------------------------

DEFAULT_LANG           = _env_str("METNOS_LANG", "it")
DEFAULT_TIMEZONE       = _env_str("METNOS_TZ", "Europe/Rome")
DEFAULT_CHANNEL        = "telegram"
DEFAULT_ACTOR          = "host"

# --- Recurring tasks quota -----------------------------------------------

MAX_TASKS_PER_ACTOR    = _env_int("METNOS_MAX_TASKS_PER_ACTOR", 50)

# --- Geo provider --------------------------------------------------------

# Chain provider (CSV): prima primary, poi fallback. Es. "google,photon".
GEO_PROVIDERS_CHAIN    = _env_str("METNOS_GEO_PROVIDERS", "google,photon")

# --- Logging ------------------------------------------------------------

LOG_LEVEL              = _env_str("METNOS_LOG_LEVEL", "INFO").upper()
LOG_FILE               = _env_path("METNOS_LOG_FILE",
                                     PATH_USER_STATE / "metnos.log")
LOG_FORMAT             = "%(asctime)s %(name)s %(levelname)s %(message)s"
LOG_DATE_FORMAT        = "%Y-%m-%dT%H:%M:%S"

# --- Helper ensure dirs ---------------------------------------------------

def ensure_dirs() -> None:
    """Crea i path user (idempotente). Non crea PATH_ROOT (deve esistere
    a deploy time)."""
    for p in (PATH_USER_DATA, PATH_USER_STATE, PATH_USER_CONFIG,
              PATH_SYNTH_EXECUTORS, PATH_AUDIT, PATH_TURNS, PATH_COST,
              DB_PAIRINGS.parent, DB_RECURRING_TASKS.parent,
              DB_MNESTOMA.parent, DB_SCHEDULER.parent):
        p.mkdir(parents=True, exist_ok=True)


# Auto-ensure al primo import (idempotente, low cost).
ensure_dirs()


# --- Backward-compat aliases (deprecabili gradualmente) ------------------

# Mnestoma.py compatibility (vecchi import). Deprecabili dopo refactor.
REINFORCE_DELTA              = WEIGHT_REINFORCE
BOOTSTRAP_WEIGHT             = WEIGHT_BOOTSTRAP
DECAY_LAMBDA_DEFAULT         = WEIGHT_DECAY_LAMBDA
DECAY_THRESHOLD              = WEIGHT_DECAY_THRESHOLD
ARCHIVE_THRESHOLD            = WEIGHT_ARCHIVE_THRESHOLD
