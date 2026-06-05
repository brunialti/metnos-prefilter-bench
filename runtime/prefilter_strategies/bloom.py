"""Strategy `bloom`: Bloom filter pre-screen + token-rank sui survivors.

Per ogni tool del catalog costruisce un Bloom filter sui token
caratteristici (name parts + verbi + oggetti + affinity tokens).
Query → check Bloom in O(1) per tool: tool con Bloom miss escluso senza
token-rank.

No false negatives su token esatti (Bloom garantisce). Falsi positivi
~1-5% si filtrano comunque al rank successivo.

Memory: ~100 byte/tool (256 bit per filter, 3 hash funcs). A 250 tool =
~25 KB. Cache LRU in memoria, rebuild su catalog signature change.

Determinismo §7.9: hash funcs deterministiche (md5 con seed fisso).
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable

_BLOOM_BITS = 256
_BLOOM_HASHES = 3
_CACHE: dict[str, tuple[str, dict]] = {}  # sig -> (sig, {name: bloom_int})


def _hash_tokens(text: str) -> list[int]:
    """Ritorna posizioni bit per ogni token rilevante."""
    positions = []
    tokens = [t for t in re.findall(r"\w+", (text or "").lower()) if len(t) > 1]
    for tok in tokens:
        # 3 funzioni hash deterministiche (md5 con diversi prefissi).
        for prefix in ("a", "b", "c")[:_BLOOM_HASHES]:
            h = hashlib.md5(f"{prefix}:{tok}".encode()).digest()
            bit = int.from_bytes(h[:4], "big") % _BLOOM_BITS
            positions.append(bit)
    return positions


def _build_bloom(tool) -> int:
    """Costruisce Bloom filter come int (bitmask) per un tool."""
    name = getattr(tool, "name", "") or ""
    parts = name.split("_")
    affinity = " ".join(getattr(tool, "affinity", []) or [])
    desc = (getattr(tool, "description", "") or "")[:500]
    corpus = " ".join(parts) + " " + affinity + " " + desc
    filter_int = 0
    for pos in _hash_tokens(corpus):
        filter_int |= (1 << pos)
    return filter_int


def _catalog_signature(catalog_list) -> str:
    h = hashlib.sha256()
    for e in sorted(catalog_list, key=lambda x: getattr(x, "name", "")):
        h.update((getattr(e, "name", "") or "").encode())
    return h.hexdigest()[:16]


def _get_filters(catalog_list) -> dict:
    sig = _catalog_signature(catalog_list)
    cached = _CACHE.get("current")
    if cached and cached[0] == sig:
        return cached[1]
    filters = {
        getattr(t, "name", ""): _build_bloom(t)
        for t in catalog_list
        if getattr(t, "name", "")
    }
    _CACHE["current"] = (sig, filters)
    return filters


def _bloom_overlap(query_filter: int, tool_filter: int) -> int:
    """Numero di bit comuni (popcount AND). Heuristic per ranking."""
    return bin(query_filter & tool_filter).count("1")


class BloomStrategy:
    name = "bloom"

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
            return [], {"chosen_k": 0, "confidence": 0.0, "reason": "empty_catalog"}
        from prefilter import _filter_dormant
        catalog_list = _filter_dormant(catalog_list)
        filters = _get_filters(catalog_list)
        # Bloom della query
        query_filter = 0
        for pos in _hash_tokens(query):
            query_filter |= (1 << pos)
        if query_filter == 0:
            # Query degenere: fallback
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        # Score ogni tool per overlap
        scored = []
        for tool in catalog_list:
            name = getattr(tool, "name", "")
            if not name or name not in filters:
                continue
            overlap = _bloom_overlap(query_filter, filters[name])
            if overlap > 0:
                scored.append((overlap, tool))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Fallback se troppo pochi survivors
        if len(scored) < k_min:
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        # Token-rank legacy sui survivors top-2*k_max (refine)
        survivors = [t for _, t in scored[:k_max * 2]]
        from prefilter import _rank_adaptive_legacy
        candidates, info = _rank_adaptive_legacy(
            query, survivors, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        info = dict(info or {})
        info["bloom_survivors"] = len(scored)
        info["bloom_reduction"] = (
            1.0 - len(scored) / max(1, len(catalog_list))
        )
        return candidates, info


def make() -> BloomStrategy:
    return BloomStrategy()
