#!/usr/bin/env python3
"""logging_setup — configurazione logger root di Metnos.

Idempotente: chiamabile N volte senza duplicare handler. Tutti i moduli
usano `logger = logging.getLogger("metnos.<modulo>")` e il root metnos
e' configurato qui.

Output di default:
  - stdout (sempre, livello INFO o env METNOS_LOG_LEVEL)
  - file (config.LOG_FILE se settato, default ~/.local/state/metnos/metnos.log)
  - journal (se invocato sotto systemd, automatico via stdout)

Pattern d'uso nei moduli:
    from logging_setup import get_logger
    log = get_logger(__name__)
    log.info("avviato")
    log.warning("...")
    log.exception("crash")  # include traceback
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import config as _C  # noqa: E402

_ROOT_NAME = "metnos"
_initialized = False


def setup_logging(level: str | None = None,
                   log_file: Path | None = None,
                   to_stdout: bool = True) -> logging.Logger:
    """Configura il logger root 'metnos'. Idempotente.

    Argomenti opzionali sovrascrivono config defaults.
    Ritorna il root logger 'metnos' configurato.
    """
    global _initialized
    root = logging.getLogger(_ROOT_NAME)
    if _initialized:
        return root
    lvl_name = (level or _C.LOG_LEVEL).upper()
    lvl = getattr(logging, lvl_name, logging.INFO)
    root.setLevel(lvl)
    fmt = logging.Formatter(_C.LOG_FORMAT, datefmt=_C.LOG_DATE_FORMAT)
    # Stderr handler (catturato da systemd journal automaticamente).
    # Bug 8/5/2026: i logger scrivevano su stdout, contaminando il JSON
    # output degli executor invocati come subprocess (build_orchestrator
    # ERROR mescolato con find_images_indices JSON → parse fail nel
    # dispatcher). Stderr e' il canale corretto per i log: systemd cattura
    # entrambi i canali, ma il subprocess capture stdout resta pulito.
    if to_stdout:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(lvl)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    # File handler con rotation (10MB, 5 backup).
    target = log_file or _C.LOG_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            target, maxBytes=10 * 1024 * 1024, backupCount=5,
        )
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except (OSError, PermissionError) as e:
        root.warning("logging_setup: file handler disabled (%s): %s", target, e)
    # Don't propagate to root logger (evita doppio output se root configurato).
    root.propagate = False
    _initialized = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Ritorna logger figlio di 'metnos'. Setup automatico se non gia' fatto.

    Esempio: `get_logger(__name__)` dentro un modulo `runtime.recurring_tasks`
    ritorna logger 'metnos.recurring_tasks'.
    """
    if not _initialized:
        setup_logging()
    # Normalizza name: stripping prefisso path se modulo e' importato strano.
    short = name.split(".")[-1] if name else "root"
    return logging.getLogger(f"{_ROOT_NAME}.{short}")
