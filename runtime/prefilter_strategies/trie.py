"""Strategy `trie`: trie sui prefissi verb→object→qualifier→provider.

I tool Metnos seguono naming `verb_object[_qualifier][_provider]` §2.2.
Costruisce trie con 4 livelli max. Query → estrae verb canonico
(intent_extractor o detect_canonical_verb), object canonico, lookup
O(L) nel trie.

Vocabolario chiuso §2.2 amplifica: 23 verbi × 19 oggetti = 437 nodi
max al 2° livello. Memory trascurabile.

Best for: routing deterministico quando verb+object sono chiari.
Complementare a verb_first ma piu' fine: discrimina anche su qualifier.
"""
from __future__ import annotations

import re
from typing import Callable


class TrieNode:
    __slots__ = ("children", "tools")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.tools: list = []


_TRIE_CACHE: dict[str, tuple[str, TrieNode]] = {}


def _build_trie(catalog_list) -> TrieNode:
    import hashlib
    sig = hashlib.sha256(
        "|".join(sorted(getattr(e, "name", "") for e in catalog_list)).encode()
    ).hexdigest()[:16]
    cached = _TRIE_CACHE.get("current")
    if cached and cached[0] == sig:
        return cached[1]
    root = TrieNode()
    for tool in catalog_list:
        name = getattr(tool, "name", "") or ""
        if not name:
            continue
        parts = name.split("_")
        node = root
        # Insert con path verb/obj/qual/provider
        for p in parts:
            if p not in node.children:
                node.children[p] = TrieNode()
            node = node.children[p]
            # Tool e' associato a ogni nodo del cammino (cosi' lookup
            # parziale come "find" o "find_files" trova tutti i discendenti)
            node.tools.append(tool)
    _TRIE_CACHE["current"] = (sig, root)
    return root


def _query_path(query: str) -> list[str]:
    """Estrae il path di lookup dalla query: verb + object canonici."""
    from prefilter import (
        detect_canonical_verb, detect_canonical_object, tokenize,
    )
    qtokens = tokenize(query)
    if not qtokens:
        return []
    verb = detect_canonical_verb(qtokens)
    obj = detect_canonical_object(qtokens, query)
    path = []
    if verb:
        path.append(verb)
        if obj:
            path.append(obj)
    return path


class TrieStrategy:
    name = "trie"

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
        root = _build_trie(catalog_list)
        path = _query_path(query)
        if not path:
            # Path vuoto → fallback
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        # Naviga il trie: usa il nodo piu' profondo che ha tool > k_min.
        node = root
        depth = 0
        for p in path:
            child = node.children.get(p)
            if child is None or len(child.tools) < k_min:
                break
            node = child
            depth += 1
        if depth == 0 or not node.tools:
            # Nessun match utile → fallback
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        # Token-rank sui survivors del trie
        # Dedup by name (Executor non sempre hashable in produzione).
        _seen = set()
        survivors = []
        for t in node.tools:
            n = getattr(t, "name", "")
            if n and n not in _seen:
                _seen.add(n)
                survivors.append(t)
        from prefilter import _rank_adaptive_legacy
        candidates, info = _rank_adaptive_legacy(
            query, survivors, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        info = dict(info or {})
        info["trie_depth"] = depth
        info["trie_path"] = path[:depth]
        info["trie_survivors"] = len(survivors)
        return candidates, info


def make() -> TrieStrategy:
    return TrieStrategy()
