"""Strategy `hybrid_cascade`: trie (cheap reducer) → token_flat (refine).

Pensiero laterale: il trie riduce search space O(N) → O(N/branching),
ma sui survivors fa solo legacy. Cascade vero: stage 1 trie depth-first
per identificare il SUB-SET semantico (es. tutti i tool con verb=find);
stage 2 token_flat sul sub-set per ranking fine, MA con boost
addizionale per i tool che il trie ha identificato come "deepest match".

Differenza vs trie v1: il trie v1 fa legacy sui survivors trie, ma usa
TUTTI i survivors (deepest + shallower). Cascade weighta i deepest piu'
forte, includendo solo i deepest se ce ne sono >=k_min.

Stage 3 (boost finale): se il match trie e' molto specifico (depth=3+,
es. tool=verb_obj_qualifier_provider matchato sui 4 livelli), bypass
token-rank → return diretto del trie pure.
"""
from __future__ import annotations

from typing import Callable


class HybridCascadeStrategy:
    name = "hybrid_cascade"

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
        from prefilter_strategies.trie import _build_trie, _query_path
        from prefilter import _filter_dormant, _rank_adaptive_legacy
        catalog_list = _filter_dormant(list(catalog))
        if not catalog_list:
            return [], {"chosen_k": 0, "confidence": 0,
                        "reason": "empty_catalog"}
        root = _build_trie(catalog_list)
        path = _query_path(query)
        # Stage 1: trie navigation
        node = root
        depth = 0
        for p in path:
            child = node.children.get(p)
            if child is None or len(child.tools) < k_min:
                break
            node = child
            depth += 1
        if depth == 0:
            # Trie nullo → fallback diretto legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        # Dedup by name (Executor non sempre hashable in produzione).
        _seen = set()
        survivors = []
        for t in node.tools:
            n = getattr(t, "name", "")
            if n and n not in _seen:
                _seen.add(n)
                survivors.append(t)
        # Stage 2: se survivors == k_min ESATTO o si sono pochi tool molto
        # specifici, bypass token rank → return diretto (saving rank cost)
        if depth >= 2 and len(survivors) <= k_max:
            return survivors[:k_max], {
                "chosen_k": len(survivors),
                "confidence": 0.95,
                "reason": f"hybrid_cascade_direct(depth={depth})",
                "cascade_depth": depth,
                "cascade_bypass_rank": True,
            }
        # Stage 3: token-rank sui survivors
        candidates, info = _rank_adaptive_legacy(
            query, survivors, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        info = dict(info or {})
        info["cascade_depth"] = depth
        info["cascade_survivors"] = len(survivors)
        info["cascade_bypass_rank"] = False
        return candidates, info


def make() -> HybridCascadeStrategy:
    return HybridCascadeStrategy()
