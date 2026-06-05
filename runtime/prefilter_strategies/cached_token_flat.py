"""Strategy `cached_token_flat`: token_flat + LRU cache su query hash.

Pensiero laterale: nei turn log Metnos, le query si ripetono spesso
("che ore sono", "trova foto", "stato del sistema" ricorrono ogni
giorno). Cache hash → tools, latency near-zero per repeat. Cache
size 256 (LRU).

Cache invalidation: signature catalog (hash dei nomi). Se cambia →
cache flushed (al primo miss).

Deterministico §7.9, zero overhead lookup O(1).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Callable


_CACHE_SIZE = 256
_CACHE: OrderedDict[str, tuple[list[str], dict]] = OrderedDict()
_CACHE_SIG: list[str] = [""]  # box for mutable


def _catalog_sig(catalog_list) -> str:
    h = hashlib.sha256()
    for e in sorted(catalog_list, key=lambda x: getattr(x, "name", "")):
        h.update((getattr(e, "name", "") or "").encode())
    return h.hexdigest()[:16]


class CachedTokenFlatStrategy:
    name = "cached_token_flat"

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
        # Check catalog signature
        sig = _catalog_sig(catalog_list)
        if _CACHE_SIG[0] != sig:
            _CACHE.clear()
            _CACHE_SIG[0] = sig
        # Cache key: query normalized + k_max
        query_norm = (query or "").strip().lower()
        key = hashlib.sha256(f"{query_norm}|{k_max}".encode()).hexdigest()[:16]
        if key in _CACHE:
            names, info = _CACHE.pop(key)
            _CACHE[key] = (names, info)  # LRU touch
            name_to_tool = {getattr(t, "name", ""): t for t in catalog_list}
            tools = [name_to_tool[n] for n in names if n in name_to_tool]
            new_info = dict(info)
            new_info["cache_hit"] = True
            return tools, new_info
        # Miss: delega a legacy
        from prefilter import _rank_adaptive_legacy
        candidates, info = _rank_adaptive_legacy(
            query, catalog, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        # Cache (solo nomi per economia memoria)
        names = [getattr(t, "name", "") for t in candidates]
        info_copy = dict(info or {})
        info_copy["cache_hit"] = False
        _CACHE[key] = (names, info_copy)
        if len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)  # evict oldest
        return candidates, info_copy


def make() -> CachedTokenFlatStrategy:
    return CachedTokenFlatStrategy()
