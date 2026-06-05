"""Plugin architecture per il prefilter Metnos (17/5/2026).

Permette di scegliere a runtime quale strategia di prefilter usare via env
`METNOS_PREFILTER`. Comportamento default: `legacy` (backward compatible,
chiama direttamente `prefilter.rank_adaptive` originale).

Strategie disponibili:
- `legacy`: il prefilter token-flat attuale (`rank_adaptive`). Baseline.
- `token_flat`: alias di `legacy`, esplicito.
- `selective_semantic`: token-flat + BGE-M3 fallback (estende ADR 0134).
- `verb_first`: dispatch deterministico §2.2 + token-rank sui survivors.
- `compare:<a>,<b>`: lancia A e B in parallelo, ritorna A ma logga B per A/B.

Pattern §7.3 generale: aggiungere una nuova strategy = aggiungere un file +
una entry in `REGISTRY`. Niente if/elif sparsi nel codice.

Telemetria: ogni invocazione logga
`~/.local/share/metnos/prefilter_telemetry.jsonl` (strategy_name, query_hash,
latency_ms, n_candidates, top3_names, confidence). Tool CLI
`runtime/prefilter_stats.py` aggrega per A/B.

API:
    select_strategy(name: str) -> PrefilterStrategy
    rank_adaptive_modular(...) -> (candidates, route_info)
"""
from __future__ import annotations

import os
from typing import Callable, Protocol


class PrefilterStrategy(Protocol):
    """Interfaccia per una strategia di prefilter.

    Ogni implementazione deve esporre `name: str` e `rank(...)` con la
    stessa signature di `prefilter.rank_adaptive` per compatibilita'
    drop-in. Il dispatcher gestisce telemetria e fallback.
    """

    name: str

    def rank(
        self,
        query: str,
        catalog,
        k_min: int = 5,
        k_max: int = 8,
        *,
        llm_call: Callable | None = None,
        prefer_intent: bool = True,
    ) -> tuple[list, dict]:
        ...


# Registry: name -> factory (no-arg, lazy import per evitare cicli).
REGISTRY: dict[str, Callable[[], PrefilterStrategy]] = {}


def register(name: str, factory: Callable[[], PrefilterStrategy]) -> None:
    """Aggiunge una strategy al registry. Idempotente."""
    REGISTRY[name] = factory


def list_strategies() -> list[str]:
    """Nomi delle strategy registrate."""
    return sorted(REGISTRY.keys())


def select_strategy(name: str | None = None) -> PrefilterStrategy:
    """Ritorna l'istanza della strategy richiesta o di quella di default.

    `name` priorita': arg esplicito > env METNOS_PREFILTER > "legacy".
    Strategy sconosciuta → fallback a "legacy" + log warning.
    """
    chosen = (name or os.environ.get("METNOS_PREFILTER", "")
              or "legacy").strip().lower()
    # `compare:a,b` → in fase 1 ritorniamo solo `a` (compare-mode handled
    # dal dispatcher esterno, vedi rank_adaptive_modular).
    if chosen.startswith("compare:"):
        chosen = chosen.split(":", 1)[1].split(",", 1)[0].strip()
    factory = REGISTRY.get(chosen)
    if factory is None:
        import logging
        logging.getLogger(__name__).warning(
            "prefilter strategy %r non registrata, fallback a 'legacy'", chosen)
        factory = REGISTRY.get("legacy")
        if factory is None:
            raise RuntimeError(
                "nessuna strategy 'legacy' registrata: registry corrotto")
    return factory()


# Auto-registrazione al primo import. Strategy concrete in moduli separati.
def _auto_register() -> None:
    """Registra le strategy built-in. Chiamato all'import del package."""
    from . import token_flat
    register("legacy", token_flat.make)
    register("token_flat", token_flat.make)
    try:
        from . import selective_semantic
        register("selective_semantic", selective_semantic.make)
    except ImportError:
        pass  # BGE-M3 infrastructure non disponibile: skip
    try:
        from . import verb_first
        register("verb_first", verb_first.make)
    except ImportError:
        pass
    try:
        from . import fts5
        register("fts5", fts5.make)
    except ImportError:
        pass
    try:
        from . import bloom
        register("bloom", bloom.make)
    except ImportError:
        pass
    try:
        from . import trie
        register("trie", trie.make)
    except ImportError:
        pass
    try:
        from . import constraint
        register("constraint", constraint.make)
    except ImportError:
        pass
    # Ottimizzazioni v2 dei top-3 (17/5/2026 sera)
    try:
        from . import trie_v2
        register("trie_v2", trie_v2.make)
    except ImportError:
        pass
    try:
        from . import token_flat_v2
        register("token_flat_v2", token_flat_v2.make)
    except ImportError:
        pass
    try:
        from . import selective_semantic_v2
        register("selective_semantic_v2", selective_semantic_v2.make)
    except ImportError:
        pass
    # Pensiero laterale (17/5/2026 sera): RRF, length-adaptive, cache, cascade
    for mod_name in ("rrf_ensemble", "length_adaptive",
                      "cached_token_flat", "hybrid_cascade"):
        try:
            mod = __import__(f"prefilter_strategies.{mod_name}",
                              fromlist=["make"])
            register(mod_name, mod.make)
        except ImportError:
            pass


_auto_register()
