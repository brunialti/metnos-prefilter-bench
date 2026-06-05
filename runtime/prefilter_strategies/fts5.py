"""Strategy `fts5`: SQLite FTS5 inverted index su (name, description,
affinity tokens). Rank BM25 nativo SQLite.

Scala a 10K+ documenti senza degrado. Stesso pattern usato da Notion
search, GitHub issues search, VS Code symbol search. Zero embedding,
zero LLM, deterministico §7.9.

Index storage: `~/.cache/metnos/prefilter_fts5.sqlite`. Rebuild
incrementale al cambio del catalog (rilevato via signature aggregata
di nomi+digest). Per ora full rebuild on-miss; ottimizzazioni
incrementali in future.

Best for: pool grande (>200 tool), persistent rank deterministico,
query con termini rari (sparse-friendly).
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Callable

_INDEX_PATH = Path.home() / ".cache" / "metnos" / "prefilter_fts5.sqlite"


def _catalog_signature(catalog_list) -> str:
    """Hash deterministico del catalog: rileva quando il pool cambia."""
    h = hashlib.sha256()
    for e in sorted(catalog_list, key=lambda x: getattr(x, "name", "")):
        name = getattr(e, "name", "") or ""
        digest = (getattr(e, "code_digest", "")
                  or getattr(e, "digest", "") or "")
        h.update(name.encode())
        h.update(digest.encode())
    return h.hexdigest()[:16]


def _build_index(catalog_list) -> tuple[sqlite3.Connection, str]:
    """Costruisce/riusa indice FTS5. Ritorna (conn, signature)."""
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    sig = _catalog_signature(catalog_list)
    conn = sqlite3.connect(str(_INDEX_PATH))
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS sig (key TEXT PRIMARY KEY, value TEXT)")
    cur.execute("SELECT value FROM sig WHERE key='catalog'")
    row = cur.fetchone()
    current_sig = row[0] if row else None
    if current_sig == sig:
        return conn, sig
    # Rebuild
    cur.execute("DROP TABLE IF EXISTS tools_fts")
    cur.execute("""
        CREATE VIRTUAL TABLE tools_fts USING fts5(
            name UNINDEXED,
            verb,
            obj,
            text,
            tokenize="unicode61 remove_diacritics 2"
        )
    """)
    for e in catalog_list:
        name = getattr(e, "name", "") or ""
        if not name:
            continue
        parts = name.split("_", 2)
        verb = parts[0] if parts else ""
        obj = parts[1] if len(parts) > 1 else ""
        affinity = " ".join(getattr(e, "affinity", []) or [])
        desc = (getattr(e, "description", "") or "")[:1000]
        text = f"{name} {verb} {obj} {affinity} {desc}".lower()
        cur.execute(
            "INSERT INTO tools_fts(name, verb, obj, text) VALUES (?, ?, ?, ?)",
            (name, verb, obj, text),
        )
    cur.execute("INSERT OR REPLACE INTO sig(key, value) VALUES ('catalog', ?)", (sig,))
    conn.commit()
    return conn, sig


def _fts_query(query: str) -> str:
    """Sanitizza query per FTS5 MATCH: token AND con OR fallback."""
    import re
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
    if not tokens:
        return ""
    # Strategia: prima 5 token come OR (più permissivo: high recall)
    return " OR ".join(tokens[:8])


class Fts5Strategy:
    name = "fts5"

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
        catalog_list = list(catalog)
        if not catalog_list:
            return [], {"chosen_k": 0, "confidence": 0.0,
                        "reason": "empty_catalog"}
        from prefilter import _filter_dormant
        catalog_list = _filter_dormant(catalog_list)
        try:
            conn, sig = _build_index(catalog_list)
        except Exception as ex:
            # Fallback to legacy se FTS5 fail (sqlite senza FTS5 compile-time).
            import logging
            logging.getLogger(__name__).warning(
                "FTS5 unavailable, fallback legacy: %s", ex)
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        match = _fts_query(query)
        if not match:
            # Empty query → fallback legacy
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT name, bm25(tools_fts) AS score FROM tools_fts "
                "WHERE tools_fts MATCH ? ORDER BY score LIMIT ?",
                (match, k_max * 2),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # Query malformata per FTS5 → fallback
            rows = []
        name_to_executor = {getattr(e, "name", ""): e for e in catalog_list}
        candidates = []
        scores = []
        for name, score in rows:
            ex = name_to_executor.get(name)
            if ex is not None:
                candidates.append(ex)
                # bm25 ritorna score NEGATIVO (più negativo = meglio).
                # Inverti per coerenza con altre strategy.
                scores.append(-score)
        # Garantisci k_min con primary_tools se troppo pochi
        if len(candidates) < k_min:
            from prefilter import detect_canonical_object, tokenize
            try:
                obj = detect_canonical_object(tokenize(query), query)
                from prefilter import _OBJECT_PRIMARY_TOOLS
                primaries = _OBJECT_PRIMARY_TOOLS.get(obj, ())
                for pname in primaries:
                    pe = name_to_executor.get(pname)
                    if pe is not None and pe not in candidates:
                        candidates.append(pe)
                        scores.append(0.0)
                    if len(candidates) >= k_min:
                        break
            except Exception:
                pass
        candidates = candidates[:k_max]
        return candidates, {
            "chosen_k": len(candidates),
            "confidence": min(1.0, scores[0] / 10.0) if scores else 0.0,
            "reason": "fts5_bm25",
            "scores_top": scores[:3],
        }


def make() -> Fts5Strategy:
    return Fts5Strategy()
