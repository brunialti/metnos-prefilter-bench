"""executor_typing — minimal lookup for executor IO typing in production.

Porting selettivo da e2e/simulator/registry.py. Espone SOLO le funzionalita'
necessarie alla prefilter rules: lookup per nome → dict con:
  - inputs: { name: {required, role, semantic_type, ...} }
  - output: { type, schema: { field_name: type } }
  - consumes: list of "kind:key" strings

§7.11 rename-resilient: il default dir si auto-deriva via __file__. Cache
mtime-based: se il file source viene aggiornato, ricarica. Senza override
env, legge da `<PATH_ROOT>/e2e/simulator/typing_cache/` (read-only, gia'
estratti 84 typing JSON), poi fallback `<PATH_USER_DATA>/executor_typing/`
se l'utente vuole estensioni custom.

API:
  - get_typing(name) -> dict | None
  - has_typing(name) -> bool
  - all_typed_names() -> list[str]

§7.9 deterministico: nessuna call LLM. Pure lettura JSON con cache.
§2.1 helper comune in runtime/, non per-executor.

Wiring: usato da prefilter_rules.compute_input_coverage_score e
compute_schema_field_score se METNOS_PREFILTER_RULES=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# Cache: name → (mtime, dict). Invalidato per-name al cambio mtime.
_TYPING_CACHE: dict[str, tuple[float, dict]] = {}
_NAMES_CACHE: Optional[set[str]] = None  # scoperti via listdir + env
_DIRS_CACHE: Optional[list[Path]] = None


def _candidate_dirs() -> list[Path]:
    """Directories da cui caricare typing JSON, in ordine di priorita'.

    Override env `METNOS_TYPING_DIR` (path:path) altrimenti default:
      1. <PATH_USER_DATA>/executor_typing/  (user custom, se esiste)
      2. <PATH_ROOT>/e2e/simulator/typing_cache/  (built-in, gia' estratti)

    §7.11: nessun path hardcoded; deriva da config.PATH_*.
    """
    global _DIRS_CACHE
    if _DIRS_CACHE is not None:
        return _DIRS_CACHE
    dirs: list[Path] = []
    env_override = os.environ.get("METNOS_TYPING_DIR", "").strip()
    if env_override:
        for raw in env_override.split(":"):
            p = Path(raw).expanduser()
            if p.is_dir():
                dirs.append(p)
    if not dirs:
        try:
            from . import config as _C  # type: ignore[no-redef]
        except Exception:
            try:
                import config as _C  # type: ignore[no-redef]
            except Exception:
                _C = None
        if _C is not None:
            user_dir = _C.PATH_USER_DATA / "executor_typing"
            if user_dir.is_dir():
                dirs.append(user_dir)
            builtin = _C.PATH_ROOT / "e2e" / "simulator" / "typing_cache"
            if builtin.is_dir():
                dirs.append(builtin)
    if not dirs:
        # Last-resort: derive root via __file__ (parents[1] = repo root).
        root = Path(__file__).resolve().parents[1]
        builtin = root / "e2e" / "simulator" / "typing_cache"
        if builtin.is_dir():
            dirs.append(builtin)
    _DIRS_CACHE = dirs
    return dirs


def _resolve_path(name: str) -> Optional[Path]:
    """Trova il file <name>.json nella prima dir disponibile.

    Fallback `_<name>.json`: i builtin runtime (classify_entries, describe_entries,
    admin, *_tasks, final_answer, get_inputs, undo_last_turn) sono salvati nel
    typing_cache con prefisso underscore (`_classify_entries.json`). Senza questo
    fallback le typed rules sarebbero silenziosamente no-op per quei tool.
    """
    for d in _candidate_dirs():
        p = d / f"{name}.json"
        if p.exists():
            return p
        pu = d / f"_{name}.json"
        if pu.exists():
            return pu
    return None


def get_typing(name: str) -> Optional[dict]:
    """Ritorna dict typing per executor `name`, o None se non typed.

    Cache mtime-aware: refresha se il file source e' stato aggiornato.
    Errori (JSON malformato, IO) → ritorna None, no raise (§2.8 fail-honest
    ma fail-graceful per rules opzionali).
    """
    if not name:
        return None
    p = _resolve_path(name)
    if p is None:
        return None
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    cached = _TYPING_CACHE.get(name)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    _TYPING_CACHE[name] = (mtime, data)
    return data


def has_typing(name: str) -> bool:
    return _resolve_path(name) is not None


def all_typed_names() -> set[str]:
    """Lista di tutti gli executor con typing disponibile (union delle dirs).

    Cache: scansiona una sola volta. Reset via reset_cache().
    """
    global _NAMES_CACHE
    if _NAMES_CACHE is not None:
        return _NAMES_CACHE
    names: set[str] = set()
    for d in _candidate_dirs():
        try:
            for f in d.iterdir():
                if f.is_file() and f.suffix == ".json":
                    names.add(f.stem)
        except OSError:
            continue
    _NAMES_CACHE = names
    return names


def reset_cache() -> None:
    """Resetta cache (test/debug)."""
    global _TYPING_CACHE, _NAMES_CACHE, _DIRS_CACHE
    _TYPING_CACHE = {}
    _NAMES_CACHE = None
    _DIRS_CACHE = None


# ── Helpers per rules (extract typed views, no semantic logic) ────────────

def required_source_inputs(typing: dict) -> list[dict]:
    """Estrae inputs con required=True AND role=='source'.

    Rule 10.5 base: solo questi sono "obbligatori dalla query" — gli altri
    sono filtri opzionali (role=filter) o meta (role=meta) bindabili a
    default. Coerente con simulator graph_search_v2:1221-1222.
    """
    if not isinstance(typing, dict):
        return []
    inputs = typing.get("inputs", {}) or {}
    out = []
    for arg_name, meta in inputs.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("required") and meta.get("role") == "source":
            out.append({
                "name": arg_name,
                "semantic_type": meta.get("semantic_type", ""),
            })
    return out


def output_schema_keys(typing: dict) -> set[str]:
    """Estrae chiavi top-level di output.schema (lowercased).

    Rule 11.0 base: i field name esposti dall'executor (schema record).
    Usati per match con token query (es. query mention "size" → boost a
    executor che espongono `size` nel schema).
    """
    if not isinstance(typing, dict):
        return set()
    output = typing.get("output", {}) or {}
    schema = output.get("schema", {}) or {}
    if not isinstance(schema, dict):
        return set()
    out: set[str] = set()
    for k in schema.keys():
        if isinstance(k, str):
            out.add(k.lower())
    return out


def output_type(typing: dict) -> str:
    """Output type string (es. 'message_entry[]', 'json_object'). Lowercased."""
    if not isinstance(typing, dict):
        return ""
    output = typing.get("output", {}) or {}
    t = output.get("type", "")
    return t.lower() if isinstance(t, str) else ""
